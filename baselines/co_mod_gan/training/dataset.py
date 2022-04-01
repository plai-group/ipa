import argparse
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from data import cifar10, ffhq256, xray, imagenet64
from vae_helpers import sample_part_images
import numpy as np
from argparse import Namespace
import wandb
import tensorflow as tf
from torchvision.transforms import functional as vF


def sample_mask(args, batch):
    # args shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(args, batch.permute(0, 2, 3, 1))[..., -1]
    x = x.unsqueeze(1)
    return x.contiguous()


class MaskDataset(Dataset):
    def __init__(self, args, img_shape):
        super().__init__()
        self.example_batch = torch.zeros(1, *img_shape)
        self.args = args

    def __len__(self):
        return 1000 * 1000 * 1000

    def __getitem__(self, idx):
        return sample_mask(self.args, self.example_batch).numpy()[0]


def infinite_loader(loader):
    while True:
        for x in loader:
            yield x


class ResizedDataset(TensorDataset):
    def __init__(self, data, size):
        self.size = size
        self.data = data
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = self.data[idx]
        return vF.resize(tmp, self.size), 0



class NewDataset:
    def __init__(self, args):
        self.dtype              = 'uint8'
        self.label_size         = 0
        self.label_dtype        = 'float32'
        self.dynamic_range      = [0, 255]
        self.pix2pix            = False
        self.args = args
        if args.dataset == "cifar10":
            (trX, _), (vaX, _), _ = cifar10(args.data_root, one_hot=False)
        elif args.dataset == "ffhq256":
            trX, vaX, _ = ffhq256(args.data_root)
        elif args.dataset == "xray":
            trX, vaX, _ = xray(args.data_root)
        elif args.dataset == "imagenet64":
            trX, vaX, _ = imagenet64(args.data_root)
        else:
            assert False
        
        self.batch_size = None
        if args.dataset == "xray":
            self.train_set = ImageFolder(trX, transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*255)]))
            self.valid_set = ImageFolder(vaX, transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*255)]))
        else:
            self.train_set = TensorDataset(torch.as_tensor(np.transpose(trX, [0, 3, 1, 2])))
            self.valid_set = TensorDataset(torch.as_tensor(np.transpose(vaX, [0, 3, 1, 2])))
        self.train_iterator = None
        self.valid_iterator = None
        self.example_batch = None
        self.hole_range = None

        self.shape = list(self.train_set[0][0].shape)
        self.resolution = self.shape[-1]

        self.resolution = self.shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.mask_loader = None
        if self.args.num_workers is not None and self.args.num_workers > 0:
            mask_dataset = MaskDataset(self.args, self.train_set[0][0].shape)
            self.mask_loader = infinite_loader(DataLoader(mask_dataset,
                                                          batch_size=4,
                                                          shuffle=False,
                                                          drop_last=True,
                                                          num_workers=self.args.num_workers,
                                                          prefetch_factor=4))

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
        minibatch_size = int(minibatch_size)
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
            self.example_batch = next(iter(train_loader))[0]
        self.hole_range = hole_range

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        assert self.batch_size is not None
        imgs, labels = self.get_minibatch_np()
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(labels)
        
    def get_minibatch_val_tf(self): # => images, labels
        assert self.batch_size is not None
        imgs, labels = self.get_minibatch_val_np()
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(labels)

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size=-1, lod=0): # => images, labels
        if minibatch_size == -1:
            minibatch_size = self.batch_size
        self.configure(minibatch_size, lod)
        imgs = next(self.train_iterator)
        return imgs, np.zeros([minibatch_size, 0])
            
    def get_minibatch_val_np(self, minibatch_size=-1, lod=0): # => images, labels
        if minibatch_size == -1:
            minibatch_size = self.batch_size
        self.configure(minibatch_size, lod)
        imgs = next(self.valid_iterator)
        return imgs, np.zeros([minibatch_size, 0])

    # Get next minibatch as TensorFlow expressions.
    def get_random_masks_tf(self): # => images, labels
        return tf.convert_to_tensor(self.get_random_masks_np())

    # Get next minibatch as NumPy arrays.
    def get_random_masks_np(self, minibatch_size=-1, hole_range=[0,1]):
        if minibatch_size == -1:
            minibatch_size = self.batch_size
        self.configure(minibatch_size)
        if self.mask_loader is not None:
            # This is a hacky way of utilizing pytorch dataloaders to prefetch the masks and get a better run-time.
            b = 0
            mask_parts = []
            while b < minibatch_size:
                m = next(self.mask_loader)
                mask_parts.append(m)
                b += len(m)
            return torch.cat(mask_parts, dim=0)[:minibatch_size].numpy()
        else:
            return sample_mask(self.args, self.example_batch).numpy()


    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size=-1): # => labels
        if minibatch_size == -1:
            minibatch_size = self.batch_size
        # with tf.name_scope('Dataset'):
        #     return tf.zeros([minibatch_size, 0], self.label_dtype)
        return tf.zeros([minibatch_size, 0], self.label_dtype)
        #return np.zeros([minibatch_size, 0], self.label_dtype)
