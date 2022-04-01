"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import math
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import zlib

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from .inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, nargs=2, default='',
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))


class FakeData(IterableDataset):
    """
    Make sure that the number of samples (N) is a multiple of
    the batch size, otherwise some samples are ignored. This
    behavior is retained to match the original FID score
    implementation.
    """

    def __init__(self, generator, N=1000, batch_size=32):

        self.generator = generator
        self.N = N
        self.number_sampled = 0
        self.batch_size = batch_size
        self.sample_shape = torch.Size([batch_size])

    def __iter__(self):

        self.number_sampled = 0
        return self

    def __next__(self):

        if self.number_sampled < self.N:
            batch_size = min(self.batch_size, self.N-self.number_sampled)
            self.number_sampled += batch_size
            return self.generator.sample(torch.Size([batch_size]))
        else:
            raise StopIteration

    def __len__(self):
        return math.ceil(self.N / self.batch_size)


def collate_fn(data):
    data = data[0]
    # Ensure data has shape (3xHxW)
    return data.expand(-1, 3, *data.shape[2:])


def collate_fn_real_data(data):
    data, target = zip(*data)
    data = torch.stack(data)
    # Ensure data has shape (3xHxW)
    return data.expand(-1, 3, *data.shape[2:])


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=50, dims=2048, cuda=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))

    for i in tqdm(range(0, len(files), batch_size)):
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr


def get_activations_dataloader(dataloader, model, N, dims=2048, cuda=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataloader  : Dataloader from which to get activations
    -- model       : Instance of inception model
    -- N           : Number of samples to use for the estimate
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((N, dims))
    end = 0
    for batch in tqdm(dataloader,
                      total=len(dataloader),
                      desc="Fid Score"):
        batch_size = batch.size(0)

        if type(batch) == tuple:
            batch = batch[0]

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        start = end
        end = start+batch_size
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    cuda=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_dataloader(dataloader, model, N, dims=2048,
                                               cuda=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader  : Dataloader from which to calculate statistics
    -- model       : Instance of inception model
    -- N           : Number of samples to use for the estimate
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_dataloader(dataloader, model, N, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims, inception_model=None):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if inception_model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    else:
        model = inception_model

    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_no_paths(generator, dataset, batch_size, cuda, dims, N,
                           normalize_input=True, inception_model=None,
                           num_workers=0):


    if inception_model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx], normalize_input=normalize_input)
    else:
        model = inception_model
    if cuda:
        model.cuda()

    # Set up dataloader for generator
    # (note we manually handle batch size in the dataset!!)
    dataloader_gen = DataLoader(FakeData(generator, N,
                                         min(batch_size, N)),
                                collate_fn=collate_fn, batch_size=1)

    # Set up dataloader for dataset
    # (note we do not manually handle batch size here!)
    dataloader_gt = DataLoader(dataset, collate_fn=collate_fn_real_data,
                               batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)

    m1, s1 = calculate_activation_statistics_dataloader(dataloader_gen, model,
                                                        N, dims, cuda)

    # check if we have saved m2 and s2
    dataset_type = type(dataset).__name__
    # dataset_hash = zlib.adler32(repr(dataset).encode('utf-8'))
    fname = f"./{dataset_type}_moments.npz"
    if not os.path.exists(fname):
        m2, s2 = calculate_activation_statistics_dataloader(dataloader_gt,
                                                            model,
                                                            len(dataset),
                                                            dims, cuda)
        np.savez(fname, m2=m2, s2=s2)
    npz = np.load(fname)
    m2 = npz['m2']
    s2 = npz['s2']

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    paths = args.path
    batch_size = args.batch_size
    cuda = args.gpu != ''
    dims = args.dims

    if '' not in paths:
        fid_value = calculate_fid_given_paths(paths, batch_size, cuda, dims)
    else:
        import torch.nn as nn
        from torch.nn import functional as F
        from torch.distributions import Normal
        from torchvision import datasets as torch_dataset, transforms

        class Test_Generator(nn.Module):

            LATENT_DIM = 4

            def __init__(self):
                super().__init__()
                self.ff = nn.Sequential(
                    nn.Linear(self.LATENT_DIM, 4*4),
                    nn.ReLU(True),
                )
                self.mean = nn.Parameter(torch.zeros(self.LATENT_DIM),
                                         requires_grad=False)
                self.std = nn.Parameter(torch.ones(self.LATENT_DIM),
                                        requires_grad=False)
                self._style_sampler = Normal(self.mean, self.std)

            def sample(self, batch_shape=torch.Size([1])):
                styles = self._style_sampler.sample(batch_shape)
                return self(styles)

            def forward(self, x):
                return F.sigmoid(self.ff(x).view(x.size(0), 1, 4, 4))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = torch_dataset.MNIST('.', download=True, train=False,
                                           transform=transform)
        fid_value = calculate_fid_no_paths(Test_Generator(), test_dataset,
                                           batch_size, cuda, dims,
                                           len(test_dataset))

    print('FID: ', fid_value)

if __name__ == '__main__':
    main()
