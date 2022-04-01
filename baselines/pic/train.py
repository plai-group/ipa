import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
#from util.visualizer import Visualizer
from data import cifar10, ffhq256, imagenet64
from vae_helpers import sample_part_images
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import os, sys
import wandb
from tqdm import tqdm
from vae_helpers import RNG, rng_decorator
from util import util
import tempfile
import imageio
import shutil


def tensor2png(img_tensor, path):
    img_numpy = img_tensor.cpu().permute(1,2,0).numpy()
    img_numpy = (img_numpy * 255).astype(np.uint8)
    imageio.imwrite(path, img_numpy)


def sample_mask_fid(opt, img, categories=None, seed=None, *args, **kwargs):
    # opt shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    if isinstance(categories, int):
        categories = torch.ones(len(img)).int() * categories
    def f():
        x = sample_part_images(
            opt, img.permute(0, 2, 3, 1),
            categories=categories, *args, **kwargs
            )[..., -1]
        return x.contiguous()
    if seed is not None:
        with RNG(seed):
            return f()
    else:
        return f()


class ScaledTensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index] / 255., 0

    def __len__(self):
        return len(self.tensor)


@torch.no_grad()
def fid(args, dataset, model, N):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ScaledTensorDataset(dataset)

    base_dir = os.environ['TMP_DIR'] if 'TMP_DIR' in os.environ else '.'
    base_dir = os.path.join(base_dir, "FID")
    os.makedirs(base_dir, exist_ok=True)

    # generate and save a bunch of images
    sample_dir = tempfile.mkdtemp(dir=base_dir, prefix=f"valid_")
    n_categories = {'patches': 6, 'patches-missing': 6, 'blank': 1, 'foveal': 6, 'freeform': 5}[args.conditioning]
    for c in range(n_categories):
        os.makedirs(os.path.join(sample_dir, str(int(c))))
    sample_i = 0

    # make dataset into format in which fid can be calculated
    tmp = os.path.join(base_dir, f"{args.dataset}_all.npz")
    assert os.path.exists(tmp) # Assumes that the pre-computed dataset stats are copied over.
    if os.path.exists(tmp):
        dataset_stats = tmp
        dataset_dir = None
    else:
        dataset_stats = os.path.join(base_dir, f"{args.dataset}_valid.npz")
        dataset_dir = os.path.join(base_dir, f"{args.dataset}_valid")
    print(f"[FID] dataset_dir = {dataset_dir}")
    print(f"[FID] dataset_stats = {dataset_stats}")

    stats2, dir2 = dataset_stats, dataset_dir
    if not os.path.exists(stats2):
        if os.path.exists(dir2):
            print(f"{stats2} doesn't exist but {dir2} does")
            raise Exception
        os.makedirs(dir2)
        i2 = 0

    for batch in tqdm(DataLoader(dataset, batch_size=args.batchSize, shuffle=False), desc="Data generation"):
        img_batch = batch[0]
        img_batch = img_batch.to(device)
        ## save dataset images to dir2 (if its stats are not already computed in stats2)
        if not os.path.exists(stats2):
            for img in img_batch:
                tensor2png(img, os.path.join(dir2, f"{i2}.png"))
                i2 += 1

        # save image completions
        if N is None or sample_i < N:
            for cat_idx, categories in enumerate(range(n_categories)):
                seed = sample_i * n_categories + cat_idx
                mask = sample_mask_fid(args, img_batch, categories=categories, seed=seed)
                mask = mask.to(device)
                samples = model.inpaint(img_batch, mask)
                for b, img in enumerate(samples):
                    path = os.path.join(sample_dir, str(int(categories)), f"{sample_i + b}.png")
                    tensor2png(img, path)
            sample_i += len(img_batch)
            if N is not None and sample_i == N and os.path.exists(stats2):
                break

    path2 = stats2 if os.path.exists(stats2) else dir2

    scores = {}
    for category in tqdm(range(n_categories), desc="FID computation"):
        cat_dir = os.path.join(sample_dir, str(int(category)))
        scores[f'fid-{int(category)}-{len(os.listdir(cat_dir))}'] = calculate_fid_given_paths([path2, cat_dir],
                                                     cache_path1=True, cache_path2=False)
        for fname in os.listdir(cat_dir):
            shutil.move(os.path.join(cat_dir, fname), os.path.join(sample_dir, f"{category}_{fname}"))  # move files from category dir to main dir
    scores[f'fid'] = calculate_fid_given_paths([path2, sample_dir], cache_path1=True, cache_path2=False)

    if dir2 is not None and os.path.exists(dir2):
        shutil.rmtree(dir2)
    shutil.rmtree(sample_dir)

    return scores


@rng_decorator(0)
@torch.no_grad()
def log_images(opt, model, viz_batch_processed):
    model.eval()
    viz_mask = sample_mask(opt, viz_batch_processed)
    data = {"img": viz_batch_processed, "mask": viz_mask}
    model.set_input(data)
    img_list_all = model.test_viz(opt.num_samples_visualize)
    img_list_all = torch.cat(img_list_all, dim=-1)
    log_dict = {}
    for i, img in enumerate(img_list_all):
        caption = f"Sample {i}"
        log_dict.update({caption: wandb.Image(util.tensor2im(img), caption=caption)})
    wandb.log(log_dict)
    model.train()


PROJECT_NAME = 'pluralistic-image-completion'
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


class MyDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return len(self.data)


def get_dataset(opt):
    transform = [T.ToTensor()]
    if opt.dataset == 'cifar10':
        transform.append(T.Resize(256))
        (trX, _), (vaX, _), (teX, _) = cifar10(opt.data_root, one_hot=False)
    elif opt.dataset == 'ffhq256':
        trX, vaX, teX = ffhq256(opt.data_root)
    elif opt.dataset == "imagenet64":
        transform.append(T.Resize(256))
        trX, vaX, teX = imagenet64(opt.data_root)
    else:
        raise ValueError(f"Unexpected dataset parameter {opt.dataset}")
    transform = T.Compose(transform)
    return [MyDataset(d, transform) for d in [trX, vaX, teX]]


def sample_mask(opt, batch):
    # opt shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(opt, batch.permute(0, 2, 3, 1))[..., -1]
    return x.unsqueeze(1).expand(x.shape[0], 3, *x.shape[1:]).contiguous()


class MaskDataset(Dataset):
    def __init__(self, args, img_shape):
        super().__init__()
        self.example_batch = torch.zeros(1, *img_shape)
        self.args = args

    def __len__(self):
        return 1000 * 1000 * 1000

    def __getitem__(self, idx):
        return sample_mask(self.args, self.example_batch).numpy()[0]


def _infinite_loader(loader):
        while True:
            for x in loader:
                yield x


class MaskLoader:
    def __init__(self, mask_dataset, num_workers):
        self.loader = _infinite_loader(DataLoader(mask_dataset,
                                                  batch_size=5,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  num_workers=num_workers,
                                                  prefetch_factor=5))
    
    def get_batch(self, batch_size):
        b = 0
        mask_parts = []
        while b < batch_size:
            m = next(self.loader)
            mask_parts.append(m)
            b += len(m)
        return torch.cat(mask_parts, dim=0)[:batch_size]


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    resume = False
    if opt.resume:
        resume = True
        wandb_id = opt.name
    wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
                config=opt, tags=opt.tags, resume=resume,
                id=opt.name if resume else None)
    if opt.name=="":
        opt.name = wandb.run.id
    TrainOptions.save_options(opt)
    # create a dataset
    # dataset = dataloader(opt)
    # dataset_size = len(dataset) * opt.batchSize
    train_set, valid_set, test_set = get_dataset(opt)
    train_loader = DataLoader(train_set, batch_size=opt.batchSize,
                              shuffle=not opt.no_shuffle,
                              num_workers=int(opt.nThreads))
    valid_set_fid = torch.as_tensor(valid_set.data).permute(0, 3, 1, 2) # A hacky way of getting the FID dataset ready
    viz_batch_processed = next(iter(DataLoader(valid_set, batch_size=opt.num_images_visualize)))
    mask_dataset = MaskDataset(opt, train_set[0].shape)
    mask_loader = MaskLoader(mask_dataset, opt.num_workers)
    print('training images = %d' % len(train_set))
    # create a model
    model = create_model(opt)
    n_params = model.num_parameters()
    print(f"Number of parameters: {n_params:,}")
    wandb.log({"num_parameters": n_params})
    # create a visualizer
    #visualizer = Visualizer(opt)
    # training flag
    keep_training = True
    #max_iteration = opt.niter+opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count

    log_images(opt, model, viz_batch_processed)
    # training process
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(train_loader):#enumerate(tqdm(train_loader)):
            iter_start_time = time.time()
            total_iteration += 1
            img = next(iter(train_loader))
            mask = mask_loader.get_batch(len(img))
            mask = mask.cuda()
            data = {"img": img, "mask": mask}
            model.set_input(data)
            model.optimize_parameters()

            # display images on visdom and save images
            # if total_iteration % opt.display_freq == 0:
            #     # img_dict = model.get_current_visuals()
            #     # final = np.concatenate(list(img_dict.values()), axis=1)
            #     # caption = ','.join(model.get_current_visuals().keys())
            #     # wandb.log({caption: wandb.Image(final, caption=caption)})
            # #     visualizer.display_current_results(model.get_current_visuals(), epoch)
            # #     visualizer.plot_current_distribution(model.get_current_dis())

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                losses["iter_time"] = t * opt.batchSize
                losses["epoch"] = epoch
                wandb.log(losses)
                # visualizer.print_current_errors(epoch, total_iteration, losses, t)
                # if opt.display_id > 0:
                #     visualizer.plot_current_errors(total_iteration, losses)

            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')
        
        log_images(opt, model, viz_batch_processed)
        # save the model after each epoch to the disk
        print('saving the model of iterations %d, epoch %d' % (total_iteration, epoch))
        save_path = model.save_networks(epoch)

        # if total_iteration > max_iteration:
        #     keep_training = False
        #     break

        model.update_learning_rate()

    print('\nEnd training')
