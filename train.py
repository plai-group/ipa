import os, sys

from torch.nn.parallel.data_parallel import DataParallel
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


if 'NO_MPI' not in os.environ:
    from mpi4py import MPI
import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import set_up_data
from utils import get_cpu_stats_over_ranks
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema, setup_save_dirs, set_seed_if_new, reload_ckpt, is_stable_is_failed
from vae_helpers import sample_part_images
from vae_helpers import RNG, rng_decorator
import wandb
import matplotlib.pyplot as plt
import shutil
from torch.nn.parallel.distributed import DistributedDataParallel


PROJECT_NAME = 'ipa'


def run_net(H, vae, x, target, iterate, differentiate):

    if H.unconditional:
        stats = vae.forward(x, target)
        if differentiate:
            (stats['elbo']/H.grad_accumulations).backward()
    else:
        part_obs = sample_part_images(H, x)
        stats = vae.forward(part_obs, x, target, obj=H.kl, iterate=iterate)
        if differentiate:
            (stats['loss']/H.grad_accumulations).backward()
    return stats


def enforce_obs(H, samples, part_obs, orig=None):
    if torch.is_tensor(part_obs):
        part_obs = part_obs.cpu().numpy()
    if orig is None:
        orig = part_obs[..., :-1]
    mask = part_obs[..., -1:]
    return samples * (1-mask) + orig * mask

def training_step(H, x, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    optimizer.zero_grad()
    for x_chunk, target_chunk in zip(torch.chunk(x, chunks=H.grad_accumulations, dim=0),
                                     torch.chunk(target, chunks=H.grad_accumulations, dim=0)):
        stats = run_net(H, vae, x_chunk, target_chunk, iterate=iterate, differentiate=True)
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    if H.kl == 'sym':
        distortion_nans = 0
        rate_nans = 0
    else:
        distortion_nans = torch.isnan(stats['distortion']).sum()
        rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(
        dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        if not H.no_ema:
            update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
    return stats


def eval_step(H, data_input, target, ema_vae, i=0):
    with torch.no_grad():
        stats = run_net(H, ema_vae, data_input, target, iterate=np.inf, differentiate=False)

    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    if dataset in ['ffhq_1024', 'xray', 'shoes', 'bags', 'shoes64', 'bags64']:
        orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
    else:
        orig_image = x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed

def loader(H, data, is_train, epoch=None, bs=None):
    sampler = None if "NO_MPI" in os.environ else  DistributedSampler(data, num_replicas=H.mpi_size, rank=H.rank)
    num_workers = H.n_workers if "NO_MPI" in os.environ else 0
    if is_train and sampler is not None:
        sampler.set_epoch(epoch)
    if bs is None:
        bs = H.n_batch*H.grad_accumulations if is_train else H.n_batch
    return DataLoader(data, batch_size=bs, drop_last=True, pin_memory=True, sampler=sampler, num_workers=num_workers,
                      shuffle=(sampler is None))

def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint,
               starting_epoch, iterate):
    optimizer, scheduler = load_opt(H, vae, logprint,
                                    init_cond_from_uncond=H.load_pretrained)

    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):

        for epoch_iter, x in enumerate(loader(H, data_train, is_train=True, epoch=epoch)):
            if epoch_iter > 0 and H.rank == 0:
                wandb.log({'iteration': iterate}, commit=True)
            data_input, target = preprocess_fn(x)
            training_stats = training_step(H, data_input, target, vae, ema_vae, optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()

            if H.no_ema:
                ema_vae = vae.module if isinstance(vae, DistributedDataParallel) else vae

            # log losses
            if iterate % H.iters_per_log == 0 or (iters_since_starting in early_evals):
                if H.rank == 0:
                    wandb.log(dict(epoch=epoch, **accumulate_stats(stats, H.iters_per_log)), commit=False)
            if iterate % 10000 == 0 or (iters_since_starting in early_evals):
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_log))

            # log images
            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                log_images(H, ema_vae, viz_batch_original, viz_batch_processed)

            # check if we need to reload whenever logging
            if iterate % H.iters_per_log == 0:
                _, failed = is_stable_is_failed(stats, H.iters_per_log)
                if failed:
                    print('reloading due to update skipping')
                    if H.rank == 0:
                        api = wandb.Api()
                        run = api.run(f'{os.environ["WANDB_ENTITY"]}/{PROJECT_NAME}/{H.wandb_id}')
                        print(run.summary)
                        if 'last_stable_save' in run.summary:
                            last_stable_save = run.summary['last_stable_save']
                            stable_ckpt_dir = os.path.join(H.save_dir, f'iter-{last_stable_save}')
                            wandb.log({'reloading_from': last_stable_save}, commit=False)
                        else:
                            return 'failed'
                    else:
                        stable_ckpt_dir = None
                    if 'NO_MPI' not in os.environ:
                        stable_ckpt_dir = MPI.COMM_WORLD.bcast(stable_ckpt_dir, root=0)
                    if stable_ckpt_dir is not None:
                        print('reloading with stable_ckpt_dir =', stable_ckpt_dir)
                        reload_ckpt(H, stable_ckpt_dir, vae, ema_vae, optimizer, logprint)
            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['loss']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_log))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H, create_dir=False)
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H, create_dir=True)
                wandb.log(dict(epoch=epoch, last_save=iterate), commit=False)
                stable, _ = is_stable_is_failed(stats, H.iters_per_log)
                print('saving', iterate)
                if stable:
                    wandb.log(dict(epoch=epoch, last_stable_save=iterate), commit=False)
                    print('and it stable',)

            if H.num_iters is not None and iterate >= H.num_iters:
                assert H.num_epochs == 1
                break

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)
            if H.rank == 0:
                valid_stats = {f'valid-{k}': v for k, v in valid_stats.items()}
                wandb.log(valid_stats, commit=False)

        if H.rank == 0:
            wandb.log({'iteration': iterate}, commit=True)


def evaluate(H, ema_vae, data_valid, preprocess_fn, is_train=False):
    stats_valid = []
    for i, x in enumerate(loader(H, data_valid, is_train=is_train)):
        data_input, target = preprocess_fn(x)
        stats_valid.append(eval_step(H, data_input, target, ema_vae, i=i))
        # print(stats_valid[-1]['distortion'])
    vals = [a['loss'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_loss=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def log_reconstructions(H, ema_vae, first_latents_from, last_latents_from,
                        viz_batch_original, caption, part_obs=None,
                        full_activations=None, part_activations=None,
                        only_top_level=False, n_upper_samples=1,
                        n_lower_samples=1, lower_t=0.1):

    zss = []
    for _ in range(n_upper_samples):
        _, stats = ema_vae.decoder.run(sample_from=first_latents_from,
                                       full_activations=full_activations,
                                       part_activations=part_activations,
                                       get_ents=H.plot_ent, get_latents=True)
        zs = [s['z'].cuda().clone() for s in stats]
        zss.append(zs)
    reconstructions = [viz_batch_original[..., -H.image_size:, :].numpy()]
    if part_obs is not None:
        if H.conditioning == 'image':
            masked = []
        else:
            masked = [enforce_obs(H, viz_batch_original.numpy()*0., part_obs)]
        reconstructions.extend(masked)
    if only_top_level:
        lv_points = np.array([int(only_top_level),])
    else:
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    lv_points = np.tile(np.repeat(lv_points, n_lower_samples), n_upper_samples)
    upper_indices = np.repeat(np.arange(n_upper_samples), (1 if only_top_level else H.num_variables_visualize)*n_lower_samples)

    for i, level in zip(upper_indices, lv_points):
        px_z, _ = ema_vae.decoder.run(sample_from=last_latents_from,
                                      full_activations=full_activations,
                                      part_activations=part_activations,
                                      manual_latents=zss[i][:level], t=lower_t)
        reconstructions.append(ema_vae.decoder.out_net.sample(px_z))
    reconstructions = np.array(reconstructions)
    for col in range(min(H.num_reconstructions_visualize, reconstructions.shape[1])):
        recon = reconstructions[:, col]
        combined = np.concatenate(list(recon), axis=1)
        name = f'{caption} {col}'
        wandb.log({name: wandb.Image(combined, caption=name)}, commit=False)

    return stats

@rng_decorator(0)
@torch.no_grad()
def log_images(H, ema_vae, viz_batch_original, viz_batch_processed):
    if H.rank != 0:
        return

    full_activations = ema_vae.encode_full_image(viz_batch_processed)
    if H.conditional:
        with RNG(H.viz_seed):
            viz_batch_masked = sample_part_images(H, viz_batch_processed)
        with RNG(H.viz_seed):
            unnormed_masked = sample_part_images(H, viz_batch_original.float())
        part_activations = ema_vae.part_encoder(viz_batch_masked)
    else:
        part_activations = None

    stats = log_reconstructions(H, ema_vae, 'full', 'prior',
                                viz_batch_original, caption='Full then prior',
                                full_activations=full_activations,
                                part_activations=part_activations)
    if H.conditional:
        _ = log_reconstructions(H, ema_vae, 'part', 'prior',
                                viz_batch_original, caption='Part then prior',
                                part_obs=unnormed_masked,
                                full_activations=full_activations,
                                part_activations=part_activations)

        if H.plot_ent:
            for img_i in range(H.num_reconstructions_visualize):
                fig, axes = plt.subplots(nrows=2, ncols=len(stats), figsize=(8, 1))
                fig.suptitle(f'Reduction in entropy for image {img_i} relative to prior')
                for i, name in enumerate(['part_enc', 'full_enc']):
                    axes[i, 0].set_ylabel(name)
                    for j, layer in enumerate(stats):
                        ents = (layer['ents'][i] - layer['ents'][-1])
                        ents = torch.nn.functional.interpolate(ents.unsqueeze(1), size=(H.image_size, H.image_size)).squeeze(1)
                        pcm = axes[i, j].imshow(ents.detach().cpu().numpy()[img_i],
                                                interpolation='none', vmin=-20, vmax=20, cmap='seismic')
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])
                plt.subplots_adjust(left=0, bottom=0.25, right=1, top=0.85, wspace=0.1, hspace=0.1)
                cbar_ax = fig.add_axes([0.09, 0.2, 0.84, 0.06])
                fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
                wandb.log({f'ents-{img_i}': fig}, commit=False)

        if H.conditional:
            for t in [1.0, 0.7, 0.9, 0.8][:H.num_temperatures_visualize]:

                all_samples = [[] for _ in range(viz_batch_original.shape[0])]
                for _ in range(H.num_samples_visualize):
                    sample_px_z, _ = ema_vae.decoder.run(sample_from='part', part_activations=part_activations)
                    sample_batch = ema_vae.decoder.out_net.sample(sample_px_z)
                    for i, sample in enumerate(sample_batch):
                        all_samples[i].append(sample)

                if H.conditioning == 'image':
                    obses = viz_batch_original[..., :H.image_size, :]
                else:
                    obses = enforce_obs(H, viz_batch_original.numpy()*0., unnormed_masked)
                for i, samples in enumerate(all_samples):
                    final = np.concatenate([obses[i]] + samples, axis=1)
                    caption = f"Samples {i} T={t}"
                    wandb.log({caption: wandb.Image(final, caption=caption)}, commit=False)

    if H.train_encoder_decoder != "" or not H.logged_unconditional:
        # log unconditional samples
        uncond_px_z, _ = ema_vae.decoder.run(sample_from='prior', n=10)
        uncond_samples = ema_vae.decoder.out_net.sample(uncond_px_z)
        uncond_samples = np.concatenate(list(uncond_samples), axis=1)  # put images side-by-side
        caption = 'Unconditional samples'
        wandb.log({caption : wandb.Image(uncond_samples, caption=caption)}, commit=False)
        H.logged_unconditional = True
    plt.close("all")


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_fn, is_train=H.eval_with_train_set)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    H.resuming = H.wandb_id is not None

    # wandb things
    if H.rank == 0:
        wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
                config=H, tags=H.tags, id=H.wandb_id, resume=H.resuming)
        H.wandb_id = wandb.run.id

    if H.resuming:
        api = wandb.Api()
        run = api.run(f'{os.environ["WANDB_ENTITY"]}/{PROJECT_NAME}/{H.wandb_id}')
        save_steps = [r for r in run.scan_history() if 'last_save' in r]
        last_save = save_steps[-1]
        cur_iters = last_save['iteration']
        cur_epoch = last_save['epoch']
    else:
        cur_iters = 0
        cur_epoch = 0

    setup_save_dirs(H)
    fresh_weights = H.pretrained_load_dir is None and not H.resuming
    H.load_pretrained = H.pretrained_load_dir is not None and not H.resuming
    H.logged_unconditional = False
    if H.load_pretrained:
        vae, ema_vae = load_vaes(H, logprint, init_cond_from_uncond=True)
    else:
        vae, ema_vae = load_vaes(H, logprint, init_cond_from_uncond=False)

    if H.train_encoder_decoder != "all":
        # Freeze weights of the unconditional parts of the VAE.
        for name, param in vae.named_parameters():
            if 'part_encoder' not in name and 'part_enc' not in name:
                if H.train_encoder_decoder == "":
                    param.requires_grad = False
                elif H.train_encoder_decoder == 'slightly':
                    param.requires_grad = ('decoder.out_net.' in name)
                else:
                    raise NotImplementedError

    if H.no_ema:
        assert H.ema_rate == 0
    set_seed_if_new(H)
    if H.rank == 0:
        n_params = sum([p.numel() for p in vae.parameters()])
        n_params_learnable = sum([p.numel() for p in vae.parameters() if p.requires_grad])
        logprint(n_params=f"{n_params:,}", n_params_learnable=f'{n_params_learnable:,}')
        wandb.log({"n_params": n_params,
                   "n_params-learnable": n_params_learnable})
    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
    else:
        return_val = 'failed'
        while return_val == 'failed':
            return_val = train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint,
                                    starting_epoch=cur_epoch, iterate=cur_iters)


if __name__ == "__main__":
    main()
