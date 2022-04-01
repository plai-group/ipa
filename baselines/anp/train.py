import json
import numpy as np
import os
import sys
import torch
from torch.distributions.kl import kl_divergence
import data_handlers
from neural_process import model_dispatcher_args, model_dispatcher, save_model
import wandb
import socket
from functools import partial
import time
import shutil
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from argparse import ArgumentParser

import test as test_module
from data import cifar10, ffhq256
from vae_helpers import sample_part_images, rng_decorator
from vae_helpers.baseline_utils import update_args
from utils import to_rgb, img_mask_to_np_input


TQDM_MIN_INTERVAL = 60
WANDB_PROJECT_NAME = "nps"

if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


def sample_mask(opt, batch):
    # opt shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(opt, batch.permute(0, 2, 3, 1))[..., -1].bool()
    return x.contiguous()


@rng_decorator(0)
@torch.no_grad()
def log_images(opt, model, viz_batch_processed):
    model.eval()
    log_dict = {}
    context_mask = sample_mask(opt, viz_batch_processed)
    for idx in range(len(viz_batch_processed)):
        to_plot = [to_rgb(context_mask[idx].unsqueeze(0) * viz_batch_processed[idx]),
                   to_rgb(viz_batch_processed[idx])]
        for _ in range(opt.num_samples_visualize):
            img = viz_batch_processed[idx].unsqueeze(0).to(opt.device)
            mask = context_mask[idx].unsqueeze(0).to(opt.device)
            to_plot.append( to_rgb(model.inpaint_img(img, mask, enforce_obs=False)).squeeze(0).cpu().numpy() )
        to_plot = np.concatenate(to_plot, axis=-1)
        to_plot = to_plot.transpose(1,2,0)
        caption = f"Sample {idx}"
        log_dict.update({caption: wandb.Image(to_plot, caption=caption)})
    wandb.log(log_dict)
    model.train()


def collate_fn_pathces(batch, args):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    # In this implementation of NPs, the context/target set size should be
    # the same along the batch dimension. Therefore, we will sample only
    # one mask and use it for the whole batch.
    context_mask = sample_mask(args, images[:1])
    if args.img_size[-1] > 32:
        subsampling_factor = (args.img_size[-1] // 32) * (args.img_size[-2] // 32)
        mask_shape = context_mask.shape
        context_mask = context_mask.view(-1)
        nonzero_indices = torch.nonzero(context_mask).view(-1)
        subsampled_indices = nonzero_indices[torch.randperm(len(nonzero_indices))[:len(nonzero_indices)//subsampling_factor]]
        context_mask = torch.zeros_like(context_mask)
        context_mask.scatter_(0, subsampled_indices, 1)
        # Construct target mask
        target_size = context_mask.numel() // subsampling_factor
        extra_target_points = max(0, target_size - context_mask.sum())
        context_zero_indices = torch.nonzero(~context_mask).view(-1)
        subsampled_indices = context_zero_indices[torch.randperm(len(context_zero_indices))[:extra_target_points]]
        target_mask = torch.clone(context_mask)
        target_mask.scatter_(0, subsampled_indices, 1)
        context_mask = context_mask.reshape(mask_shape)
        target_mask = target_mask.reshape(mask_shape)
        context_mask = context_mask.expand(len(images), *context_mask.shape[1:])
        target_mask = target_mask.expand(len(images), *target_mask.shape[1:])
    else:
        context_mask = context_mask.expand(len(images), *context_mask.shape[1:])
        target_mask = torch.ones_like(context_mask)
    x_context, y_context = img_mask_to_np_input(images, context_mask)
    x_target, y_target = img_mask_to_np_input(images, target_mask)
    # Returns two respresentations of the data: x-y pairs and image-masks.
    return (x_context, y_context, x_target, y_target), (images, context_mask, target_mask)


class MyDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), 0 # Returns the image and a dummy label

    def __len__(self):
        return len(self.data)


def get_dataset(args):
    transform = [T.ToTensor()]
    if args.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(args.data_root, one_hot=False)
    elif args.dataset == 'ffhq256':
        transform.append(T.Resize(64))
        trX, vaX, teX = ffhq256(args.data_root)
    else:
        raise ValueError(f"Unexpected dataset parameter {args.dataset}")
    transform = T.Compose(transform)
    return [MyDataset(d, transform) for d in [trX, vaX, teX]]


def loss_np(q_context, q_target, p_y_pred, y_target):
    # Log likelihood has shape (batch_size, num_target, y_dim).
    # Take the mean over batch and sum over number of targets
    # and dimensions of y
    log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
    return -log_likelihood + kl


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.
    """

    def __init__(self, args, model, optimizer):
        for k in ["device", "num_context_range", "num_extra_target_range", "log_loss_every"]:
            assert hasattr(args, k)
        self.args = args
        self.model = model
        self.optimizer = optimizer
        # Check if neural process is for images
        self.steps = 0
        self.loss_fn = loss_np

    def train_step(self, data):
        # Create context and target points and apply neural process
        x_context, y_context, x_target, y_target = data
        x_context = x_context.to(self.args.device)
        y_context = y_context.to(self.args.device)
        x_target = x_target.to(self.args.device)
        y_target = y_target.to(self.args.device)

        q_context, q_target, p_y_pred = self.model(x_context, y_context, x_target, y_target)
        loss = self.loss_fn(q_context, q_target, p_y_pred, y_target)
        return loss

    def train_epoch(self, data_loader):
        """
        Trains Neural Process for one epoch.
        Returns the average training loss for the epoch

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance
        """
        epoch_loss = 0.
        for i, data in enumerate(tqdm(data_loader, mininterval=TQDM_MIN_INTERVAL)):
            data = data[0]
            self.optimizer.zero_grad()
            # Forward pass
            loss = self.train_step(data)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Update stats
            epoch_loss += loss.item()
            self.steps += 1
            # Logging
            if self.steps % self.args.log_loss_every == 0:
                wandb.log({"loss": loss.item(),
                           "iteration": self.steps})
        return epoch_loss / len(data_loader)


def main(args):
    ### Initialize the dataset ###
    collate_fn = partial(collate_fn_pathces, args=args)
    train_set, valid_set, test_set = get_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=8,
                              prefetch_factor=4)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=0)
    viz_batch_processed = next(iter(DataLoader(valid_set, batch_size=args.num_images_visualize)))[0]
    ### Initialize the model ###
    model = model_dispatcher_args(args).to(args.device)
    print(f"Number of parameters = {model.num_parameters():,}")
    wandb.log({"Parameters": model.num_parameters()}, step=0)
    ### Initialize the optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ### Initialize the trainer ###
    trainer = NeuralProcessTrainer(args=args, model=model,
                                   optimizer=optimizer)
    ### Prepare the paths to save the trained models ###
    checkpoints_dir = Path("checkpoints") / wandb.run.id
    checkpoints_dir.mkdir(parents=True)
    model_path_latest = checkpoints_dir / "model_latest.pt"
    model_path_best = checkpoints_dir / 'model_best.pt'
    valid_loss_best = float("inf")
    best_epoch = 0

    ### Training loop ###
    log_images(args, model, viz_batch_processed)
    for epoch in tqdm(range(args.epochs), leave=False, desc="Epoch"):
        t_0 = time.time()
        avg_loss = trainer.train_epoch(train_loader)
        epoch_time = time.time() - t_0
        # Save the latest model after each epoch
        save_model(model, model_path_latest,
                   config=args.get_dict(), epoch=epoch)
        if epoch % args.save_every == 0:
            shutil.copy2(model_path_latest, checkpoints_dir / f"model_{epoch}.pt")
        # Load the saved model for evaluation
        if epoch % args.eval_every == 0:
            model.eval()
            # Evaluate the model qualitatively, save it as an image and log it to wandb
            log_images(args, model, viz_batch_processed)
            # Evaluate the model quantitavely (NLL, log_prob)
            valid_loss_dict = test_module.quantitative(model, dataloader=valid_loader, device=args.device)
            wandb.log({**valid_loss_dict}, commit=False)
            # Update the best model, if achieved a better validation loss
            if valid_loss_dict["valid_loss.NLL"] < valid_loss_best:
                shutil.copy2(model_path_latest, model_path_best)
                best_epoch = epoch
                valid_loss_best = valid_loss_dict["valid_loss.NLL"]
            model.train()
        # Log the epoch average loss
        wandb.log({"epoch_loss": avg_loss,
                   "epoch_time": epoch_time,
                   "epoch": epoch,
                   "epoch_best": best_epoch,
                   "valid_loss_best": valid_loss_best})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, required=True)
    # Model arguments
    parser.add_argument("--r_dim", type=int, default=512)
    parser.add_argument("--s_dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--likelihood_std", type=float, default=None,
                        help="If None, will learn the likelihood standard deviation as a part of the model."\
                             "Otherwise (it should be a float) fixes the likelihood standard deviation to the given value")
    parser.add_argument("--likelihood_std_bias", type=float, default=0.1)
    parser.add_argument("--posterior_std_bias", type=float, default=0.1)
    parser.add_argument("--self_attentions", type=int, default=2,
                        help="Specifies the number of self-attention layers (in latent and deterministic paths).")
    parser.add_argument("--cross_attentions", type=int, default=2,
                        help="Specifies the number of cross-attention layers (only in deterministic path).")
    parser.add_argument("--attention_type", default="transformer")
    parser.add_argument("--deterministic_path", default=True)
    # Training arguments
    parser.add_argument("--num_context_range", type=int, nargs=2, default=[1, 200])
    parser.add_argument("--num_extra_target_range", type=int, nargs=2,  default=[0, 200])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--log_loss_every", type=int, default=200,
                        help="Frequency of logging the training loss (after graient updates)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Frequency of keeping the saved models at the trained models at the end of epochs (after epochs)")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Frequency of evaluating the model (after epochs)")
    parser.add_argument("--eval_m_list", type=int, nargs="*", default=[3, 5, 15, 30, 90, 1024])
    parser = update_args(parser)
    args = parser.parse_args()

    if args.dataset == "ffhq256":
        args.img_size = [3, 64, 64]
        args.num_context_range = [x*2 for x in args.num_context_range]
        args.num_extra_target_range = [x*2 for x in args.num_extra_target_range]
    elif args.dataset == "cifar10":
        args.img_size = [3, 32, 32]
    else:
        raise Exception(f"Unrecognized dataset {args.dataset}. Please use ffhq256 or cifar10.")

    wandb_run = wandb.init(project=WANDB_PROJECT_NAME,
                           entity=os.environ['WANDB_ENTITY'],
                           config=args)
    main(args)