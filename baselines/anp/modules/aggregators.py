"""This file implements different (permutation-invaraint) aggregation functions
to be used for implementing the permutation-invariant function.

Parameters
----------
All the functions take one argument: items.
It is supposed to have a shape of BxKx... where B is the batch size and
K is the number of items in the set. "..." denotes arbitrary number of shape
of the rest of dimensions. Pooling operates on the second dimension (with size K)
"""
import torch
from torch import nn
import numpy as np
from .attention import DotAttender


def mean_pool(items):
    return torch.mean(items, dim=1)


def max_pool(items):
    return torch.max(items, dim=1)[0]


def sum_pool(items):
    return torch.sum(items, dim=1)


def logsumexp_pool(items):
    return torch.logsumexp(items, dim=1)


def get_agg_fn(name):
    AGG_FN_DICT = {"mean": mean_pool,
                   "max": max_pool,
                   "sum": sum_pool,
                   "logsumexp": logsumexp_pool}
    return AGG_FN_DICT[name]