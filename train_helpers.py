import torch
import numpy as np
import socket
import argparse
import os
from functools import partial
if 'NO_MPI' not in os.environ:
    from mpi4py import MPI
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
from torch.optim import AdamW as BasicAdamW
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE, ConditionalVAE
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import DataParallel


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def save_model(path, vae, ema_vae, optimizer, H, create_dir):
    if create_dir:
        if os.path.exists(path):
            print('\n\n WARNING: path already exists. perhaps restarting after interrupted save. \n')
        else:
            os.mkdir(path)
    torch.save(vae.state_dict(), os.path.join(path, 'model.th'))
    torch.save(ema_vae.state_dict(), os.path.join(path, 'model-ema.th'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'opt.th'))
    torch.save(dict(H), os.path.join(path, 'config.th'))


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if 'nans' in k or 'skip' in k:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'elbo':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['elbo'] = np.mean(vals)
            z['elbo_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = stats[-1][k] if len(stats) < frequency else np.mean([a[k] for a in stats[-frequency:]])
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:] if k in a])
    return z


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


def setup_mpi(H):
    H.mpi_size = mpi_size()
    H.local_rank = local_mpi_rank()
    H.rank = mpi_rank()
    if 'NO_MPI' not in os.environ:
        os.environ["MASTER_ADDR"] = MPI.COMM_WORLD.bcast(socket.gethostname(), root=0)
        os.environ["MASTER_PORT"] = str(H.port) if H.port is not None else str(np.random.randint(29500, 29999))
        os.environ["RANK"] = str(H.rank)
        os.environ["WORLD_SIZE"] = str(H.mpi_size)
        # os.environ["NCCL_LL_THRESHOLD"] = "0"
        torch.cuda.set_device(H.local_rank)
        dist.init_process_group(backend='nccl', init_method=f"env://")


def distributed_maybe_download(path, local_rank, mpi_size):
    if not path.startswith('gs://'):
        return path
    filename = path[5:].replace('/', '-')
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp


@contextmanager
def first_rank_first(local_rank, mpi_size):
    if mpi_size > 1 and local_rank > 0:
        dist.barrier()
    try:
        yield
    finally:
        if mpi_size > 1 and local_rank == 0:
            dist.barrier()


def setup_save_dirs(H):
    if H.wandb_id is None:
        H.wandb_id = 'none'
    H.save_dir = os.path.join(H.save_dir, H.wandb_id)
    if H.rank == 0:
        mkdir_p(H.save_dir)
        mkdir_p(os.path.join(H.save_dir, 'latest'))
    H.logdir = os.path.join(H.save_dir, 'log')

    if H.resuming:
        if H.ckpt_load_dir is None:
            H.ckpt_load_dir = os.path.join(H.save_dir, 'latest')
        print(f'Using ckpt_load_dir {H.ckpt_load_dir}.')
    elif H.ckpt_load_dir is not None:
        print(f"Warning: not resuming but loading from checkpoint at {H.ckpt_load_dir}")


def set_up_hyperparams(s=None, do_print=True):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_mpi(H)
    logprint = logger(None)  # H.logdir)
    if do_print:
        for i, k in enumerate(sorted(H)):
            logprint(type='hparam', key=k, value=H[k])
        logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint


def set_seed_if_new(H):
    if H.resuming:
        return
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)


def restore_params(H, model, path, local_rank, mpi_size, init_cond_from_uncond, map_ddp=True, map_cpu=False):
    state_dict = torch.load(distributed_maybe_download(path, local_rank, mpi_size), map_location='cpu' if map_cpu else None)
    if map_ddp:
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    make_part_encoder_initialisation(H, state_dict, init_cond_from_uncond)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print('\nKeys missing from state dict. Ensure this is intentional.\n')
        model.load_state_dict(state_dict, strict=False)


def restore_log(path, local_rank, mpi_size):
    loaded = [json.loads(l) for l in open(distributed_maybe_download(path, local_rank, mpi_size))]
    try:
        cur_eval_loss = min([z['elbo'] for z in loaded if 'type' in z and z['type'] == 'eval_loss'])
    except ValueError:
        cur_eval_loss = float('inf')
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return cur_eval_loss, iterate, starting_epoch


def make_part_encoder_initialisation(H, state_dict, init_cond_from_uncond):
    if (H.pretrained_partial_encoder == "") or (not H.init_cond_from_uncond):
        return
    for k in list(state_dict.keys()):
        ks = k.split('.')
        if ks[0] == 'encoder':
            new_k = '.'.join(['part_encoder'] + ks[1:])
            if k == 'encoder.in_conv.weight':
                # add extra input channel
                v = state_dict[k]
                v = torch.cat([v, torch.zeros_like(v[:, :1])], dim=1)
            else:
                v = state_dict[k]
            state_dict[new_k] = v
        elif (H.pretrained_partial_encoder == "all") and (ks[0] == 'decoder' and len(ks) >= 4 and ks[3] == 'enc'):
            new_k = k.replace('enc', 'part_enc')
            state_dict[new_k] = state_dict[k]
        else:
            continue


def load_vaes(H, logprint, init_cond_from_uncond=False, ckpt_dir=None):
    print('loading vaes', init_cond_from_uncond)
    if ckpt_dir is not None:
        load_dir = ckpt_dir
    elif init_cond_from_uncond:
        load_dir = H.pretrained_load_dir
    else:
        load_dir = H.ckpt_load_dir
    VAE_type = ConditionalVAE if H.conditional else VAE
    vae = VAE_type(H)
    if load_dir is not None:
        if init_cond_from_uncond:
            # use pretrained model with ema
            vae_path = os.path.join(load_dir, 'model-ema.th')
        else:
            vae_path = os.path.join(load_dir, 'model.th')
        logprint(f'Restoring vae from {vae_path}')
        restore_params(H, vae, vae_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size,
                       init_cond_from_uncond=init_cond_from_uncond)

    ema_vae = VAE_type(H)
    if load_dir is not None:
        ema_path = os.path.join(load_dir, 'model-ema.th')
        print(ema_path)
        logprint(f'Restoring ema vae from {ema_path}')
        restore_params(H, ema_vae, ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size,
                       init_cond_from_uncond=init_cond_from_uncond)
    else:
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)

    vae = vae.cuda(H.local_rank)
    ema_vae = ema_vae.cuda(H.local_rank)

    if "NO_MPI" not in os.environ:
        vae = DistributedDataParallel(vae, device_ids=[H.local_rank], output_device=H.local_rank, find_unused_parameters=True)  # ideally would not need find_unused_parameters
    if len(list(vae.named_parameters())) != len(list(vae.parameters())):
        raise ValueError('Some params are not named. Please name all params.')
    total_params = 0
    for name, p in vae.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae


def load_opt(H, vae, logprint, init_cond_from_uncond=False):
    optim_type = BasicAdamW# if 'NO_MPI' in os.environ else AdamW
    optimizer = optim_type([p for p in vae.parameters() if p.requires_grad], weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))
    if init_cond_from_uncond:
        load_dir = H.pretrained_load_dir
        assert load_dir is not None
        return optimizer, scheduler
    else:
        load_dir = H.ckpt_load_dir
    if load_dir is not None and not H.not_load_opt:
        opt_path = os.path.join(load_dir, 'opt.th')
        print(f'Restoring opt from {opt_path}.')
        optimizer.load_state_dict(
            torch.load(distributed_maybe_download(opt_path, H.local_rank, H.mpi_size), map_location='cpu'))
    return optimizer, scheduler


def reload_ckpt(H, ckpt_dir, vae, ema_vae, optimizer, logprint):
    opt_path = os.path.join(ckpt_dir, 'opt.th')
    optimizer.load_state_dict(
        torch.load(distributed_maybe_download(opt_path, H.local_rank, H.mpi_size), map_location='cpu'))
    vae_path = os.path.join(ckpt_dir, 'model.th')
    ema_path = os.path.join(ckpt_dir, 'model-ema.th')
    vae_module = vae if 'NO_MPI' in os.environ else vae.module
    restore_params(H, vae_module, vae_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, init_cond_from_uncond=False)
    restore_params(H, ema_vae, ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, init_cond_from_uncond=False)


def reinit(H, vae, ema_vae, optimizer, logprint):    # really shitty function but may be good enough
    # vae.build()
    vae.decoder.build()
    vae.encoder.build()
    vae.decoder = vae.decoder.cuda(H.local_rank)
    vae.encoder = vae.encoder.cuda(H.local_rank)

def is_stable_is_failed(stats, horizon):
    recent_stats = stats[-horizon:]
    prop_skipped_updates = sum(s['skipped_updates'] for s in recent_stats) / len(recent_stats)
    stable = (prop_skipped_updates < 0.25) and (stats[-1]['skipped_updates'] == 0)   # hard-coded hyperparameters :)
    failed = (prop_skipped_updates == 1) and (len(recent_stats) == horizon)
    return stable, failed
