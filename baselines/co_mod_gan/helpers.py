# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

from vae_helpers import rng_decorator
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from data import cifar10, ffhq256
from vae_helpers import sample_part_images
import numpy as np
from argparse import Namespace
import wandb


def sample_mask(args, batch):
    # args shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(args, batch.permute(0, 2, 3, 1))[..., -1]
    x = x.unsqueeze(1)
    return x.contiguous()


@rng_decorator(0)
def log_images(args, model, viz_batch_processed):
    def inpaint(gt_images, masks):
        masked_images = gt_images * masks
        recon, mask = model(masked_images, masks, rounds=-1)
        inpainted = recon * (1 - masks) + gt_images * masks
        return inpainted.cpu().numpy()
    model.eval()
    masks = sample_mask(args, viz_batch_processed)
    log_dict = {}
    for idx in range(len(viz_batch_processed)):
        to_plot = [(viz_batch_processed[idx] * masks[idx]).cpu().numpy(),
                   viz_batch_processed[idx].cpu().numpy()]
        for _ in range(args.num_samples_visualize):
            to_plot.append(np.clip(inpaint(viz_batch_processed[idx].unsqueeze(0), masks[idx].unsqueeze(0)).squeeze(0), 0, 1))
        to_plot = np.concatenate(to_plot, axis=-1)
        to_plot = to_plot.transpose(1,2,0)
        caption = f"Sample {idx}"
        log_dict.update({caption: wandb.Image(to_plot, caption=caption)})
    wandb.log(log_dict)
    model.train()


#######******************************##############
class NewDataset:
    def __init__(self, args):
        self.args = args
        if args.dataset == "cifar10":
            (trX, _), (vaX, _), (teX, _) = cifar10(args.data_root, one_hot=False)
        elif args.dataset == "ffhq256":
            trX, vaX, teX = ffhq256(args.data_root)
        
        self.batch_size = None
        self.train_set = TensorDataset(torch.as_tensor(np.transpose(trX, [0, 3, 1, 2])))
        self.valid_set = TensorDataset(torch.as_tensor(np.transpose(vaX, [0, 3, 1, 2])))
        self.train_iterator = None #self._iterator(self.train_loader)
        self.valid_iterator = None #self._iterator(self.valid_loader)
        self.example_batch = None #next(iter(self.train_loader))

        self.shape = list(self.train_set[0][0].shape)
        self.resolution = self.shape[-1]

    def _iterator(self, dataloader):
        while True:
            for x in dataloader:
                yield x[0].contiguous().numpy()
    
    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0, hole_range=[0,1]):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 #and lod in self._tf_datasets
        assert lod == 0
        if self.batch_size != minibatch_size:
            self.batch_size = minibatch_size
            train_loader = DataLoader(self.train_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True)
            valid_loader = DataLoader(self.valid_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True)
            self.train_iterator = self._iterator(train_loader)
            self.valid_iterator = self._iterator(valid_loader)
            self.example_batch = next(iter(train_loader))

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        assert self.batch_size is not None
        return next(self.train_iterator), self.get_random_labels_tf(self.batch_size)
        
    def get_minibatch_val_tf(self): # => images, labels
        assert self.batch_size is not None
        return next(self.valid_iterator), self.get_random_labels_tf(self.batch_size)

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        return self.get_minibatch_tf()
            
    def get_minibatch_val_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        return self.get_minibatch_val_tf()

    # Get next minibatch as TensorFlow expressions.
    def get_random_masks_tf(self): # => images, labels
        return sample_mask(self.args, self.example_batch).numpy()
        #tf.convert_to_tensor

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        # with tf.name_scope('Dataset'):
        #     return tf.zeros([minibatch_size, 0], self.label_dtype)
        return np.zeros([minibatch_size, 0])


def load_dataset_new(dataset, data_root,
                        conditioning,
                        max_patches,
                        patch_size_frac):
    args = Namespace(dataset=dataset,
                        data_root=data_root,
                        conditioning=conditioning,
                        max_patches=max_patches,
                        patch_size_frac=patch_size_frac)
    return NewDataset(args)
    
#######******************************##############


#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train CoModGAN using the FFHQ dataset
  python %(prog)s --data-dir=~/datasets --dataset=ffhq --metrics=ids10k --num-gpus=8

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train CoModGAN.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--dataset', choices=['cifar10', 'ffhq256'], required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='ids10k', type=_parse_comma_sep)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--resume-with-new-nets', default=False, action='store_true')
    parser.add_argument('--disable-style-mod', default=False, action='store_true')
    parser.add_argument('--disable-cond-mod', default=False, action='store_true')
    ### New arguments
    parser.add_argument('--tags', type=str, nargs='*', default=[])
    parser.add_argument('--conditioning', type=str,
                            choices=['patches', 'patches-missing', 'blank'], default='patches')
    parser.add_argument('--data_root', type=str, default='../../')
    parser.add_argument('--max_patches', type=int, default=5)
    parser.add_argument('--patch_size_frac', type=float, default=0.35,
                        help="Patch width as fraction of image width.")
    parser.add_argument('--num_images_visualize', type=int, default=5)
    parser.add_argument('--num_samples_visualize', type=int, default=5)
    parser.add_argument('--img_size', type=int)

    args = parser.parse_args()

    dataset = NewDataset(args)
    dataset.configure(minibatch_size=12)
    

    import pdb; pdb.set_trace()
    print("done")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

