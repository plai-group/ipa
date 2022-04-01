#!/usr/bin/env python3
# code adapted from https://github.com/sbarratt/inception-score-pytorch
# to reflect pytorch 1.6
#
# Whether or not to use inception score look at
# https://arxiv.org/pdf/1801.01973.pdf
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import math
from torch.utils.data import IterableDataset
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


class FakeData(IterableDataset):
    def __init__(self, p, N=1000, batch_size=32):
        self.p = p
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
            return self.p.sample(torch.Size([batch_size]))
        else:
            raise StopIteration
    def __len__(self):
        return math.ceil(self.N / self.batch_size)

def collate_fn(data):
    # Ensure data has shape (3xHxW)
    data = data[0]
    return data.expand(-1, 3, *data.shape[2:])


def inception_score(generator, N=50000, cuda=True, batch_size=32, resize=True,
                    splits=10, normalize_input=True, inception_model=None,
                    entropy_only=False, num_classes=1000):
    """Computes the inception score of the generated images imgs
    p -- the generator model
    N -- total number of datapoints to use for the metric
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    assert batch_size > 0
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Set up dataset
    dataset = FakeData(generator, N, batch_size)
    # Set up dataloader (note we manually handle batch size in the dataset!!)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn,
                                             batch_size=1)
    # Load inception model
    if inception_model is None:
        inception_model = inception_v3(pretrained=True,
                                       transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear',
                     align_corners=False).type(dtype)
    def get_pred(x):
        if normalize_input:
            x = x * 2 - 1
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    # Get predictions
    preds = np.empty((N, num_classes))
    end = 0
    for batch in tqdm(dataloader,
                      total=len(dataloader),
                      desc="Inception Score"):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]
        start = end
        end = start+batch_size_i
        preds[start:end] = get_pred(batch)
    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            if entropy_only:
                scores.append(entropy(pyx))
            else:
                scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


def inception_score_mnist(generator, mnist_classifier, N=50000, cuda=True, batch_size=32,
                          splits=10, entropy_only=False, grayscale=True, num_classes=10):
    """Computes the inception score of the generated images imgs
    p -- the generator model
    N -- total number of datapoints to use for the metric
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    assert batch_size > 0
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Set up dataset
    dataset = FakeData(generator, N, batch_size)
    # Set up dataloader (note we manually handle batch size in the dataset!!)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn,
                                             batch_size=1)
    # Prepare MNIST lcassifier
    mnist_classifier.eval()
    def get_pred(x):
        x = mnist_classifier(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    # Get predictions
    preds = np.empty((N, num_classes))
    end = 0
    for batch in tqdm(dataloader,
                      total=len(dataloader),
                      desc="Inception Score"):
        # Convert images to Grayscale by removing 2 of the 3 color channels
        if grayscale:
            batch = batch[:, :1]
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]
        start = end
        end = start+batch_size_i
        preds[start:end] = get_pred(batch)
    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            if entropy_only:
                scores.append(entropy(pyx))
            else:
                scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    from torch.distributions import Normal
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
    print("Calculating Inception Score...")
    print(inception_score(Test_Generator(), N=int(5e4), cuda=True,
                          batch_size=32, resize=True, splits=10))