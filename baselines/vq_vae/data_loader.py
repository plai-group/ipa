import os
import numpy as np
import torch
from data import cifar10, ffhq256
from vae_helpers import sample_part_images
import numpy as np
import tensorflow as tf
from torchvision.transforms import functional as vF
import torchvision.transforms as transforms


def sample_mask(args, batch):
    # args shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(args, batch)[..., -1]
    x = x.unsqueeze(1)
    return x.contiguous()


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, args, img_shape):
        super().__init__()
        self.example_batch = torch.zeros(1, *img_shape)
        self.args = args

    def __len__(self):
        return 1000 * 1000 * 1000

    def __getitem__(self, idx):
        mask = sample_mask(self.args, self.example_batch)[0]
        mask = mask.permute(1, 2, 0).contiguous().numpy()
        return 1 - mask # Masks in this paper are inverted


def infinite_loader(loader):
    while True:
        for x in loader:
            yield x


class ResizedDataset(torch.utils.data.Dataset):
    def __init__(self, data, size):
        self.size = size
        self.data = data
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = self.data[idx]
        if tmp.shape[-2] != self.size:
            tmp = vF.resize(np.transpose(tmp, [2, 0, 1]), 256)
            tmp = np.transpose(tmp, [1, 2, 0])
        tmp = (tmp / 127.5) - 1
        return vF.resize(tmp, self.size), 0


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        img = img.permute(1,2,0).numpy()
        return img, 0

    def __len__(self):
        return len(self.data)


class NewDataset:
    # Image pixel values are in [-1, 1]
    def __init__(self, args):
        self.dtype              = 'uint8'
        self.label_size         = 0
        self.label_dtype        = 'float32'
        self.dynamic_range      = [0, 255]
        self.pix2pix            = False
        self.batch_size = args.batch_size
        self.args = args
        transform = [transforms.ToTensor()]
        if args.dataset == "cifar10":
            transform.append(transforms.Resize(256))
            (trX, _), (vaX, _), _ = cifar10(args.data_root, one_hot=False)
        elif args.dataset == "ffhq256":
            trX, vaX, _ = ffhq256(args.data_root)
        else:
            assert False
        transform.append(transforms.Normalize(0.5, 0.5))
        transform = transforms.Compose(transform)
        # self.train_set = torch.utils.data.TensorDataset(
        #     torch.as_tensor(trX))
        # self.train_set = ResizedDataset(torch.as_tensor(trX), 256)
        # self.valid_set = ResizedDataset(torch.as_tensor(vaX), 256)
        self.train_set = TransformedDataset(trX, transform)
        self.valid_set = TransformedDataset(vaX, transform)
        mask_dataset = MaskDataset(self.args, self.train_set[0][0].shape)

        if self.args.num_workers is None or self.args.num_workers == 0:
            dataloader_kwargs = {}
        else:
            dataloader_kwargs = dict(num_workers=self.args.num_workers,
                                     prefetch_factor=4)
        self.train_iterator = self._iterator(
            torch.utils.data.DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                **dataloader_kwargs))
        self.valid_iterator = self._iterator(
            torch.utils.data.DataLoader(
                self.valid_set,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True))
        self.mask_loader = infinite_loader(
            torch.utils.data.DataLoader(mask_dataset,
                                        batch_size=4,
                                        shuffle=False,
                                        drop_last=True,
                                        **dataloader_kwargs))
        self.example_batch = next(self.train_iterator)

    def _iterator(self, dataloader):
        while True:
            for x in dataloader:
                yield x[0].contiguous().numpy()

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images
        return tf.convert_to_tensor(self.get_minibatch_np())
        
    def get_minibatch_val_tf(self): # => images
        return tf.convert_to_tensor(self.get_minibatch_val_np())

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self): # => images
        imgs = next(self.train_iterator)
        return imgs
            
    def get_minibatch_val_np(self): # => images
        imgs = next(self.valid_iterator)
        return imgs

    # Get next minibatch as TensorFlow expressions.
    def get_random_masks_tf(self, minibatch_size=-1): # => images
        return tf.convert_to_tensor(self.get_random_masks_np(minibatch_size))

    # # Get next minibatch as NumPy arrays.
    def get_random_masks_np(self, minibatch_size=-1):
        if minibatch_size == -1:
            minibatch_size = self.batch_size
        # This is a hacky way of utilizing pytorch dataloaders to prefetch the masks and get a better run-time.
        b = 0
        mask_parts = []
        while b < minibatch_size:
            m = next(self.mask_loader)
            mask_parts.append(m)
            b += len(m)
        return torch.cat(mask_parts, dim=0)[:minibatch_size].numpy()