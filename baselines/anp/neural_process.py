import torch
import numpy as np
import modules
from modules import DeterministicEncoder, StochasticEncoder, LatentNormalPosteriorEncoder # Encoders
from modules import Decoder # Decoders
from torch import nn
from torch.distributions import Normal
from utils import img_mask_to_np_input, xy_to_img
from argparse import Namespace


class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    s_dim : int
        Dimension of output representation s.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, s_dim, z_dim, h_dim, likelihood_std, pooling,
                 self_attentions, cross_attentions, deterministic_path,
                 posterior_std_bias=0.1, likelihood_std_bias=0.1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.r_dim = s_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.pooling = pooling

        modules.POSTERIOR_STD_BIAS = posterior_std_bias
        modules.LIKELIHOOD_STD_BIAS = likelihood_std_bias

        # Initialize networks
        ## Encoders
        self.stochastic_encoder = StochasticEncoder(self.x_dim, self.y_dim, self.h_dim, self.s_dim,
                                                    self_attentions=self_attentions, pooling=self.pooling)
        if deterministic_path == True:
            self.deterministic_encoder = DeterministicEncoder(self.x_dim, self.y_dim, self.h_dim, self.s_dim,
                                                              self_attentions=self_attentions,
                                                              cross_attentions=cross_attentions,
                                                              pooling=self.pooling)
        else:
            self.deterministic_encoder = lambda *args, **kwargs: None
        ## Latent-specific networks
        self.s_to_q = LatentNormalPosteriorEncoder(s_dim, z_dim)
        ## Decoder
        self.decoder = Decoder(self.x_dim, self.z_dim, self.r_dim, self.h_dim, self.y_dim,
                               likelihood_std, deterministic_path=deterministic_path)
    
    def latent_posterior(self, x, y):
        """Given a set of I/O pairs x, y returns the variational posterior
        of the latent variable z.
        q(z|x,y)
        """
        embedding = self.stochastic_encoder(x, y)
        q = self.s_to_q(embedding)
        return q

    def forward(self, x_context, y_context, x_target, y_target):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        computes and return the loss for neural processes.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim).
        """
        assert self.training == True
        # Encode target and context (context needs to be encoded to
        # calculate kl term)
        # Latent path
        q_target = self.latent_posterior(x_target, y_target)
        q_context = self.latent_posterior(x_context, y_context)
        # Deterministic path
        r = self.deterministic_encoder(x_context, y_context, x_target)
        # Sample from encoded distribution using reparameterization trick
        z_sample = q_target.rsample()
        # Get parameters of output distribution
        p_y_pred = self.decoder(x_target, z_sample, r)
        return q_context, q_target, p_y_pred

    def forward_test(self, x_context, y_context, x_target):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        """
        assert self.training == False
        # Infer the variational posterior distribution
        q_context = self.latent_posterior(x_context, y_context)
        # Sample from distribution based on context
        z_sample = q_context.rsample()
        # Deterministc path
        r = self.deterministic_encoder(x_context, y_context, x_target)
        # Predict target points based on context and the latent variable
        p_y_pred = self.decoder(x_target, z_sample, r)
        return p_y_pred

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralProcessImg(NeuralProcess):
    """
    Wraps regular Neural Process for image processing.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 32, 32)

    s_dim : int
        Dimension of output representation s.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, args):
        self.img_size = args.img_size
        self.num_channels, self.height, self.width = self.img_size
        super().__init__(x_dim=2, y_dim=self.num_channels,
                         s_dim=args.s_dim, z_dim=args.z_dim,
                         h_dim=args.h_dim,
                         likelihood_std=args.likelihood_std,
                         pooling=args.pooling,
                         self_attentions=args.self_attentions,
                         cross_attentions=args.cross_attentions,
                         deterministic_path=args.deterministic_path,
                         posterior_std_bias=args.posterior_std_bias,
                         likelihood_std_bias=args.likelihood_std_bias)

    @torch.no_grad()
    def inpaint_img(self, img_batch, mask_batch, enforce_obs=False):
        ## For inpainting, use Neural Process in prediction mode
        was_training = self.training
        self.eval()
        ## Prepare the context mask
        context_batch = mask_batch.bool() # Convert the mask type to boolean
        ## Prepare the target mask
        if enforce_obs:
            target_batch = ~context_batch # All pixels which are not in context
            target_batch[:, 0, 0] = 1 # Makes sure that the target mask is not all zeros, to avoid errors
        else:
            target_batch = torch.ones_like(context_batch)
        ## Forward-pass
        x_context, y_context = img_mask_to_np_input(img_batch, context_batch)
        x_target, y_target = img_mask_to_np_input(img_batch, target_batch)
        p_y_pred = self.forward_test(x_context, y_context, x_target)
        ## Use the mean (i.e. loc) parameter of normal distribution as predictions
        ## for y_target
        img_rec = xy_to_img(x_target, p_y_pred.loc.detach(), img_batch.shape[1:])
        ## Add context points back to image
        if enforce_obs:
            context_batch_expanded = context_batch.unsqueeze(1).expand(img_rec.shape)
            inpainted = img_rec + context_batch_expanded * img_batch
        else:
            inpainted = img_rec
        inpainted = torch.clamp(inpainted, 0, 1)
        ## Reset model to mode it was in before inpainting
        if was_training:
            self.train()
        return inpainted

    @torch.no_grad()
    def encode_img(self, img_batch, mask_batch):
        assert self.neural_process.decode.deterministic_path == False
        ## Prepare the context mask
        context_batch = mask_batch.bool() # Convert the mask type to boolean
        x_context, y_context = img_mask_to_np_input(img_batch, context_batch)
        s_context = self.stochastic_encoder(x, y)
        return s_context

    @torch.no_grad()
    def posterior_img(self, img_batch, mask_batch):
        assert self.decoder.deterministic_path == False
        ## Prepare the context mask
        context_batch = mask_batch.bool() # Convert the mask type to boolean
        x_context, y_context = img_mask_to_np_input(img_batch, context_batch)
        q_context = self.latent_posterior(x_context, y_context)
        return q_context

    @torch.no_grad()
    def decode_img(self, z_batch, mask_batch):
        assert self.neural_process.decode.deterministic_path == False
        ## Prepare the target mask
        target_batch = mask_batch.bool() # Convert the mask type to boolean
        p_y_pred = self.decoder(target_batch, z_batch, None)
        x_target, _ = img_mask_to_np_input(z_batch.new_zeros(len(mask_batch), *self.img_size), target_batch)
        pred = p_y_pred.loc.detach().cpu()
        img_rec = xy_to_img(x_target.cpu(), pred, torch.Size(self.img_size))
        return img_rec


def model_dispatcher_args(args):
    if args.dataset.startswith("gp-"):
        return NeuralProcess(x_dim=1, y_dim=1,
                             s_dim=args.s_dim, z_dim=args.z_dim,
                             h_dim=args.h_dim,
                             likelihood_std=args.likelihood_std,
                             pooling=args.pooling,
                             self_attentions=args.self_attentions,
                             cross_attentions=args.cross_attentions,
                             deterministic_path=args.deterministic_path,
                             posterior_std_bias=args.posterior_std_bias,
                             likelihood_std_bias=args.likelihood_std_bias)
    else:
        return NeuralProcessImg(args)


def model_dispatcher(path):
    """Given path to a saved model, will instantiate a model and
    load its weights from the file.
    Args:
        path (str): Path to the saved model
    """
    data = torch.load(path, map_location=lambda storage, loc: storage)
    config = data["config"]
    state_dict = data["state_dict"]
    model = model_dispatcher_args(Namespace(**config))
    model.load_state_dict(state_dict)
    return model


def save_model(model, path, config, epoch):
    torch.save({"state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch},
                path)