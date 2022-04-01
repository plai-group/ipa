import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class Xrays(ImageFolder):
    attrs = ['No Finding', 'Emphysema', 'Infiltration', 'Nodule', 'Mass',
             'Atelectasis', 'Pleural_Thickening', 'Hernia', 'Pneumonia', 'Fibrosis',
             'Consolidation', 'Pneumothorax', 'Edema', 'Effusion', 'Cardiomegaly']
    def __init__(self, csv_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = {}
        for line in open(csv_path, 'r').readlines():
            image_fname , findings, *_= line.split(',')
            if image_fname == 'Image Index':
                continue  # line is header
            findings = findings.split('|')
            # sanity checks ----------------
            assert len(findings) != 0
            if len(findings) > 1:
                assert 'No Finding' not in findings
            # ------------------------------
            target = torch.tensor([1 if attr in findings else 0 for attr in self.attrs])
            self.targets[image_fname] = target

    def __getitem__(self, index):
        """
        copy of default, but gets target from somewhere else
        """
        path, _ = self.samples[index]
        fname = path.split('/')[-1]
        target = self.targets[fname]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def set_up_data(H, labelled=False, custom_transform=None):
    if H.data_root[0] == '$':
        H.data_root = os.environ[H.data_root[1:]]
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
        resize_to = None
    elif H.dataset == 'imagenet64':
        trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
        resize_to = None
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
        resize_to = None
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
        resize_to = None
    elif H.dataset == 'xray':
        trX, vaX, teX = xray(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -0.5  # mean of first 987 images is 0.494
        scale = 1.0 / 0.25  # standard deviation of first 987 images is 0.2489 for each channel
        shift_loss = -0.5
        scale_loss = 2.0
        resize_to = None
    elif H.dataset in ['shoes', 'shoes64']:
        trX, vaX, teX = shoes(H.data_root)
        H.image_size = 64 if H.dataset == 'shoes64' else 256
        H.image_channels = 3
        if H.dataset == 'shoes64':
            shift = -115.92961967 / 256  # params from imagenet64 so we can use pretrained models
            scale = 1. / 69.37404 * 256
        else:
            shift = -0.7507      # -ve mean of first 1000 training B (non-edge) images
            scale = 1. / 0.3417  # inverse of std of first 1000 training B images
        shift_loss = -0.5
        scale_loss = 2.0
        resize_to = None if H.dataset == 'shoes' else (64, 128)
    elif H.dataset in ['bags', 'bags64']:
        trX, vaX, teX = bags(H.data_root)
        H.image_size = 64 if H.dataset == 'bags64' else 256
        H.image_channels = 3
        if H.dataset == 'bags64':
            shift = -115.92961967 / 256  # params from imagenet64 so we can use pretrained models
            scale = 1. / 69.37404 * 256
        else:
            shift = -0.6995      # -ve mean of first 1000 training B (non-edge) images
            scale = 1. / 0.3544  # inverse of std of first 1000 training B images
        shift_loss = -0.5
        scale_loss = 2.0
        resize_to = None if H.dataset == 'bags' else (64, 128)
    elif H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        if H.norm_like is None:
            shift = -120.63838
            scale = 1. / 64.16736
        elif H.norm_like == 'imagenet32_from_imagefolder':
            shift = -116.2373
            scale = 1. / 69.37404
        else:
            raise Exception
        resize_to = None
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

    if labelled:
        assert H.dataset == 'xray'
        csv_path = os.path.join(H.data_root, 'chest-xrays', 'Data_Entry_2017_v2020.csv')
        ts = transforms.ToTensor() if custom_transform is None else custom_transform
        train_data = Xrays(csv_path, trX, ts)
        valid_data = Xrays(csv_path, eval_dataset, transforms.ToTensor())
        untranspose = True
    elif H.dataset in ['ffhq_1024', 'xray', 'shoes', 'bags', 'shoes64', 'bags64']:
        transform = transforms.ToTensor()
        if resize_to is not None:
            transform = transforms.Compose([transforms.Resize(resize_to), transform])
        train_data = ImageFolder(trX, transform)
        valid_data = ImageFolder(eval_dataset, transform)
        untranspose = True
    else:
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))
        untranspose = False

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        if do_low_bit:
            # 5 bits of precision
            out.mul_(1. / 8.).floor_().mul_(8.)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def imagenet32(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet32-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet32-valid.npy'), mmap_mode='r')
    return train, valid, test


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')

def xray(data_root):
    # we did not significantly tune hyperparameters, and so simply evaluate on the test set
    return os.path.join(data_root, 'chest-xrays/train'), os.path.join(data_root, 'chest-xrays/test'), os.path.join(data_root, 'chest-xrays/test')

def bags(data_root):
    # we did not significantly tune hyperparameters, and so simply evaluate on the test set
    return os.path.join(data_root, 'edges2handbags/train'), os.path.join(data_root, 'edges2handbags/val'), os.path.join(data_root, 'edges2handbags/val')

def shoes(data_root):
    # we did not significantly tune hyperparameters, and so simply evaluate on the test set
    return os.path.join(data_root, 'edges2shoes/train'), os.path.join(data_root, 'edges2shoes/val'), os.path.join(data_root, 'edges2shoes/val')

def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid


def cifar10(data_root, one_hot=True):
    tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)
