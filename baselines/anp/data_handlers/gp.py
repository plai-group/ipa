import math
import torch
import gpytorch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)


def get_random_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state()
    }

def set_random_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])



class GP(gpytorch.models.ExactGP):
    def __init__(self, kernel_args, noise=0.001, mean=0):
        """
        noise: \sigma^2 parameter
        y(x) = f(x) + \epsilon  where  \epsilon ~ N(0, \sigma^2)
        K(x_1, x_2) = lengthscale \exp{-1/(2*lengthscale^2) (x_1-x_2)^2}
        """
        self._noise = noise
        self._kernel_args = kernel_args
        self._mean = mean
        # Likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self._noise
        super().__init__(train_inputs=None,
                         train_targets=None,
                         likelihood=likelihood)
        # Mean
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = torch.nn.Parameter(torch.tensor(mean, dtype=torch.float))
        # Kernel
        base_kernel = self.initialize_kernel(kernel_args)
        self.kernel_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.kernel_module.outputscale = kernel_args.outputscale
        # Put the model in eval mode and fix the parameters
        self.eval()
        self.requires_grad_(False)
        
    def initialize_kernel(self, kernel_args):
        if kernel_args.type == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
            kernel.lengthscale = kernel_args.lengthscale
            # Higher lengthscale: Smoother functions
        elif kernel_args.type == "cosine":
            kernel = gpytorch.kernels.CosineKernel()
            kernel.period_length = kernel_args.period_length
        elif kernel_args.type == "matern":
            assert kernel_args.nu in [0.5, 1.5, 2.5], "MaternKernel's nu is expecte to be in [0.5, 1.5, 2.5]"
            kernel = gpytorch.kernels.MaternKernel(nu=kernel_args.nu)
            kernel.lengthscale = kernel_args.lengthscale
            # Higher lengthscale: Smoother functions
            # Higher nu: Smoother functions
        elif kernel_args.type == "linear":
            kernel = gpytorch.kernels.LinearKernel()
            kernel.variance = kernel_args.variance
            # Variance: controls variance of slope of functions
        elif kernel_args.type == "poly":
            kernel = gpytorch.kernels.PolynomialKernel(power=kernel_args.power)
            kernel.offset = kernel_args.offset
            # Lower offset -> Smoother functions
        else:
            raise NotImplementedError
        return kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @torch.no_grad()
    def prior_manual(self, x):
        cov = self.kernel_module(x).add_jitter(self.likelihood.noise)
        cov = cov.evaluate()
        prior = gpytorch.distributions.MultivariateNormal(torch.ones(len(x)) * self.mean_module.constant, cov)
        return prior
    
    @torch.no_grad()
    def prior(self, x):
        with gpytorch.settings.prior_mode(state=True):
            prior = self.likelihood(self(x))
        return prior
    
    @torch.no_grad()
    def posterior(self, x_target, x_context, y_context):
        self.set_train_data(x_context, y_context, strict=False)
        posterior = self.likelihood(self(x_target))
        return posterior


class GPDataloader(Dataset):
    def __init__(self, model, batch_size, num_context_range, num_extra_target_range, xrange=[-1, 1], dataset_size=1024):
        self.model = model#kernel_args, noise=0.001, mean=0
        self.dataset_size = dataset_size
        self.xrange = xrange
        self.batch_size = batch_size
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self._cnt = 0

    def __iter__(self):
        self._cnt = 0
        return self

    def __next__(self):
        if self._cnt + self.batch_size <= self.dataset_size:
            self._cnt += self.batch_size
            return self.get_batch()
        else:
            raise StopIteration

    def get_batch(self):
        batch_size = self.batch_size
        num_context = np.random.randint(*self.num_context_range)
        num_extra_target = np.random.randint(*self.num_extra_target_range)
        x_batch = self.sample_x(batch_size=batch_size, n=num_context + num_extra_target)
        # TODO: batched sampling
        y_batch = torch.stack([self.model.prior(x).sample() for x in x_batch], dim=0)
        x_batch = x_batch.unsqueeze(-1)
        y_batch = y_batch.unsqueeze(-1)
        x_context = x_batch[:, :num_context].clone()
        y_context = y_batch[:, :num_context].clone()
        x_target = x_batch
        y_target = y_batch
        return (x_context, y_context, x_target, y_target), None

    def sample_x(self, n, batch_size=1):
        return torch.rand(batch_size, n) * (self.xrange[1] - self.xrange[0]) + self.xrange[0]

    def __len__(self):
        return self.dataset_size // self.batch_size

def test():
    from argparse import Namespace
    kernel_args = Namespace(type="rbf", lengthscale=0.2, outputscale=1)
    model = GP(kernel_args=kernel_args)
    loader = GPDataloader(model, batch_size=10, num_context_range=[1, 10], num_extra_target_range=[0, 10])
    return loader