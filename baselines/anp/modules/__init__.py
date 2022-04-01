import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal

from .utils import _mlp
from .attention import get_attender
from .aggregators import get_agg_fn


POSTERIOR_STD_BIAS = 0.1 # In order to have more numerical stability, variational distribution's will have a bias in their standard deviation
LIKELIHOOD_STD_BIAS = 0.1 # The bias in the learned lielihood's standard deviation


class StochasticEncoder(nn.Module):
    """
    Implements s as a function of (x, y)
    Maps an a set of (x_i, y_i) pairs to a an aggregated
    representation s.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    hidden_dim : int
        Dimension of hidden layer.

    output_dim : int
        Dimension of output representation s.
    """
    def __init__(self, x_dim, y_dim, hidden_dim, output_dim,
                 self_attentions, pooling, attention_type="transformer"):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.agg_fn = get_agg_fn(pooling)

        # Input embedding layers
        self.input_projection = _mlp([x_dim + y_dim] + [hidden_dim] * 2 + [output_dim])
        # Self-attention layers
        self.self_attentions = nn.ModuleList([
            get_attender(attention_type, kq_size=output_dim,
                         value_size=output_dim, out_size=output_dim)
            for _ in range(self_attentions)
            ])

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        if num_points == 0:
            s = x.new_zeros(batch_size, self.output_dim)
        else:
            input_pairs = torch.cat((x, y), dim=-1)
            s_i = self.input_projection(input_pairs)
            # Apply self-attention
            for attention in self.self_attentions:
                s_i = attention(s_i, s_i, s_i)
            # Aggregate representations s_i into a single representation s
            s = self.agg_fn(s_i)
        return s
        

class DeterministicEncoder(nn.Module):
    """
    Implements r as a function of (x, y)
    Maps an a set of (x_i, y_i) pairs to a an aggregated
    representation s.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    hidden_dim : int
        Dimension of hidden layer.

    output_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, hidden_dim, output_dim,
                 self_attentions, cross_attentions, pooling,
                 attention_type="transformer"):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.agg_fn = get_agg_fn(pooling)

        # Input embedding layers
        self.input_projection = _mlp([x_dim + y_dim] + [hidden_dim] * 2 + [output_dim])
        # Self-attention layers
        self.self_attentions = nn.ModuleList([
            get_attender(attention_type, kq_size=output_dim,
                         value_size=output_dim, out_size=output_dim)
            for _ in range(self_attentions)
            ])
        # Cross-attention layers
        if cross_attentions > 0:
            self.context_projection = _mlp([x_dim] + [hidden_dim] + [output_dim])
            self.target_projection = _mlp([x_dim] + [hidden_dim] + [output_dim])
            self.cross_attentions = nn.ModuleList([
                get_attender(attention_type, kq_size=output_dim,
                            value_size=output_dim, out_size=output_dim)
                for _ in range(cross_attentions)
                ])

    def forward(self, x, y, x_target):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        if num_points == 0:
            query = x.new_zeros(batch_size, self.output_dim)
        else:
            input_pairs = torch.cat((x, y), dim=-1)
            r_i = self.input_projection(input_pairs)
            # Apply self-attention
            for attention in self.self_attentions:
                r_i = attention(r_i, r_i, r_i)
            # Apply cross-attention
            if hasattr(self, "cross_attentions") and len(self.cross_attentions) > 0:
                # query: x_target, key: x, value: r_i
                query = self.target_projection(x_target)
                keys = self.context_projection(x)
                for attention in self.cross_attentions:
                    query = attention(keys, query, r_i)
            else:
                query = self.agg_fn(r_i)
        return query


class LatentNormalPosteriorEncoder(nn.Module):
    """
    Implements q_\phi(z|s) i.e. q_\phi(z|x,y) where x,y are encoded in the vector s
    Maps a representation s to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    s_dim : int
        Dimension of output representation s.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, s_dim, z_dim):
        super().__init__()

        self.s_dim = s_dim
        self.z_dim = z_dim
        hidden_dim = s_dim

        self.s_to_hidden = _mlp([s_dim, hidden_dim])
        self.hidden_to_mu = _mlp([hidden_dim, z_dim])
        self.hidden_to_sigma = _mlp([hidden_dim, z_dim])

    def forward(self, s):
        """
        s : torch.Tensor
            Shape (batch_size, s_dim)
        """
        hidden = torch.relu(self.s_to_hidden(s))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = POSTERIOR_STD_BIAS + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)


class Decoder(nn.Module):
    """
    Implements the likelihood p_\theta(y_T|s, r, x_T). r is optional, depending
    on if the model has a deterministic path
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, r_dim, h_dim, y_dim, likelihood_std, deterministic_path):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.likelihood_std = likelihood_std
        self.deterministic_path = deterministic_path

        input_dim = x_dim + z_dim + r_dim if deterministic_path else x_dim + z_dim

        self.xz_to_hidden = _mlp([input_dim] + [h_dim] * 4)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        if self.likelihood_std is None:
            self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z, r):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Expand z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        if self.deterministic_path:
            # r is either 2-dimensional (without cross-attention) or 3-dimensional (with cross-attention)
            assert r.ndim in [2, 3]
            if r.ndim == 2:
                r = r.unsqueeze(1).repeat(1, num_points, 1)
            r_flat = r.view(batch_size * num_points, self.r_dim)
        # Input is concatenation of z with every row of x and r (if deterministc path exists)
        if self.deterministic_path:
            input_tuples = torch.cat((x_flat, z_flat, r_flat), dim=1)
        else:
            input_tuples = torch.cat((x_flat, z_flat), dim=1)

        hidden = torch.relu(self.xz_to_hidden(input_tuples))
        mu = self.hidden_to_mu(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        if self.likelihood_std is None:
            pre_sigma = self.hidden_to_sigma(hidden)
            # Reshape output into expected shape
            pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
            # Define sigma following convention in "Empirical Evaluation of Neural
            # Process Objectives" and "Attentive Neural Processes"
            sigma = LIKELIHOOD_STD_BIAS + 0.9 * F.softplus(pre_sigma)
        else:
            sigma = torch.ones_like(mu) * self.likelihood_std
        return Normal(mu, sigma)