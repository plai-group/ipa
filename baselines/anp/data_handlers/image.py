import glob
import numpy as np
import torch
from math import pi
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import batch_context_target_mask
import os
from utils import img_mask_to_np_input


DATA_ROOT = "datasets" if "NP_DATA_ROOT" not in os.environ else os.environ["NP_DATA_ROOT"]
RNG = np.random


def collate_fn(batch, num_context_range, num_extra_target_range, img_size, rng_state=None):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    #labels = torch.tensor(labels)
    batch_size = len(images)
    rng = RNG if rng_state is None else np.random.RandomState(rng_state)

    # Sample number of context and target points
    num_context = rng.randint(*num_context_range)
    num_extra_target = rng.randint(*num_extra_target_range)
    while num_context == 0 and num_extra_target == 0:
        num_extra_target = rng.randint(*num_extra_target_range)
    
    context_mask, target_mask = \
        batch_context_target_mask(img_size,
                                  num_context, num_extra_target,
                                  batch_size,
                                  rng=rng)
    x_context, y_context = img_mask_to_np_input(images, context_mask)
    x_target, y_target = img_mask_to_np_input(images, target_mask)
    # Returns two respresentations of the data: x-y pairs and image-masks.
    return (x_context, y_context, x_target, y_target), (images, context_mask, target_mask)


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(size, path_to_data=os.path.join(DATA_ROOT, "mnist"), split="train"):
    """MNIST dataset.

    Parameters
    ----------
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    assert split in ["train", "test", "valid"]
    trans = [transforms.Resize(size)]
    trans.append(transforms.ToTensor())
    all_transforms = transforms.Compose(trans)

    dataset = datasets.MNIST(path_to_data,
                             train=True if split == "train" else False,
                             download=True,
                             transform=all_transforms)

    return dataset


def celeba(size, path_to_data=os.path.join(DATA_ROOT, "celeba"), split="train", crop=128, target_type="identity"):
    """CelebA dataset.

    Parameters
    ----------
    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    assert split in ["train", "test", "valid"]
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    dataset = datasets.CelebA(path_to_data,
                              transform=transform,
                              split=split,
                              target_type=target_type,
                              download=True)
    return dataset


def fashion_mnist(size, path_to_data=os.path.join(DATA_ROOT, "fashion_mnist"), split="train"):
    """FashionMNIST dataset.

    Parameters
    ----------
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to FashionMNIST data files.
    """
    assert split in ["train", "test", "valid"]
    trans = [transforms.Resize(size)]
    trans.append(transforms.ToTensor())
    all_transforms = transforms.Compose(trans)

    dataset = datasets.FashionMNIST(path_to_data,
                                    train=True if split == "train" else False,
                                    download=True,
                                    transform=all_transforms)

    return dataset


def cifar10(size, path_to_data=os.path.join(DATA_ROOT, "cifar10"), split="train"):
    """FashionMNIST dataset.

    Parameters
    ----------
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to FashionMNIST data files.
    """
    assert split in ["train", "test", "valid"]
    trans = [transforms.Resize(size)]
    trans.append(transforms.ToTensor())
    all_transforms = transforms.Compose(trans)

    dataset = datasets.CIFAR10(path_to_data,
                               train=True if split == "train" else False,
                               download=True,
                               transform=all_transforms)

    return dataset


class ToyDataset(Dataset):
    def __init__(self, n, img_size=32, categories=10):
        super().__init__()
        assert categories <= img_size
        self.categories = categories
        self.img_size = img_size
        self.n = n
        self.noisy=True

    def __getitem__(self, index):
        n = index % self.categories
        color_channel = (index // self.categories) % 3
        ret = torch.zeros(3, self.img_size, self.img_size)
        colored_size = (self.img_size // self.categories)*n
        ret[color_channel, :colored_size, :colored_size] = 1
        if self.noisy:
            ret += torch.randn(ret.shape) * 0.05
            ret = torch.clamp(ret, 0, 1)
        return ret, 0
    
    def __len__(self):
        return self.n


def toy(size=32, path_to_data=None, split="train"):
    categories = 10
    if split == "train":
        n = 100000
    elif split == "valid":
        n = categories * 3 * 10
    elif split == "test":
        n = 10000
    dataset = ToyDataset(n=n, img_size=size, categories=categories)
    return dataset


def toy_tiny(size=32, path_to_data=None, split="train"):
    categories = size
    if split == "train":
        n = 100000
    elif split == "valid":
        n = categories * 3 * 10
    elif split == "test":
        n = 10000
    dataset = ToyDataset(n=n, img_size=size, categories=categories)
    return dataset


class ToyMNIST(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def _segment_on(self, segment, tensor):
        if segment == "A":
            tensor[0, 1:4] = 1
        elif segment == "B":
            tensor[0:3, 3] = 1
        elif segment == "C":
            tensor[3:6, 3] = 1
        elif segment == "D":
            tensor[4, 1:4] = 1
        elif segment == "E":
            tensor[3:6, 1] = 1
        elif segment == "F":
            tensor[0:3, 1] = 1
        elif segment == "G":
            tensor[2, 1:4] = 1
        return tensor
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        idx = idx % 10
        item = torch.zeros(5, 5)
        segments = {0: ["A", "B", "C", "D", "E", "F"],
                    1: ["B", "C"],
                    2: ["A", "B", "G", "E", "D"],
                    3: ["A", "B", "G", "C", "D"],
                    4: ["F", "G", "B", "C"],
                    5: ["A", "F", "G", "C", "D"],
                    6: ["A", "F", "G", "C", "D", "E"],
                    7: ["A", "B", "C"],
                    8: ["A", "B", "C", "D", "E", "F", "G"],
                    9: ["A", "B", "C", "D", "F", "G"]}
        for segment in segments[idx]:
            item = self._segment_on(segment, item)
        return item.unsqueeze(0), 0
    
    def __len__(self):
        return self.n


def toy_mnist(size=5, path_to_data=None, split="train"):
    n = 10000 if split == "train" else 10
    dataset = ToyMNIST(n)
    return dataset


DATASET_DICT = {"mnist": mnist,
                "celeba": celeba,
                "fashion_mnist": fashion_mnist,
                "cifar10": cifar10,
                "toy": toy,
                "toy_tiny": toy_tiny,
                "toy_mnist": toy_mnist}


def get_dataloaders(dataset, batch_size, size=32, num_workers=4, collate_fn=None, **kwargs):
        assert dataset in DATASET_DICT
        dataset_func = DATASET_DICT[dataset]
        train_set, test_set, valid_set = [dataset_func(size=size, split=split, **kwargs) for split in ["train", "test", "valid"]]

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=0,
                                 collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  shuffle=False, num_workers=0,
                                  collate_fn=collate_fn)
        
        return train_loader, test_loader, valid_loader
