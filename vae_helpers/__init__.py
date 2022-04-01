import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from .mask_generator import np_mask_generator, RandomMask, BatchRandomMask
import multiprocessing as mp
import random

# def approx_gaussian_cross_entropy(mu1, logsigma1, mu2, logsigma2):
#     """ returns E_{p1} [ -log p2 ]
#         approximated with a zeroth order (first order terms are zero)
#         Taylor expansion
#         - accurate if sigma1 is small compared to (mu1-mu2)
#         - logsigmas are elementwise stds assuming independence between all variables
#     """
#     k = mu1.shape[-1]
#     return logsigma2 + 0.5 * k*np.log(2*np.pi) + 0.5 * (mu1-mu2)**2 * (-2*logsigma2).exp()

# def approx_gaussian_cross_entropy_scaled_grads(mu1, logsigma1, mu2, logsigma2, max_grad):
#     grad_logsigma2 = torch.ones_like(logsigma2) - (mu1-mu2)**2 * (-2*logsigma2).exp()
#     grad_logsigma1 = 0. * grad_logsigma2
#     grad_mu1 = (mu1-mu2) * (-2*logsigma2).exp()
#     grad_mu2 = -grad_mu1
#     grad_all = torch.stack([grad_mu1, grad_mu2, grad_logsigma1, grad_logsigma2], dim=-1)
#     grad_norm = torch.norm(grad_all, dim=-1)
#     scaling = (max_grad/grad_norm).clamp(max=1)
#     ce = approx_gaussian_cross_entropy(mu1, logsigma1, mu2, logsigma2)
#     return ce.detach() + \
#         torch.sum(
#             (scaling*grad_mu1).detach()* (mu1 - mu1.detach()) +
#             (scaling*grad_mu2).detach()* (mu2 - mu2.detach()) +
#             (scaling*grad_logsigma1).detach()* (logsigma1 - logsigma1.detach()) +
#             (scaling*grad_logsigma2).detach()* (logsigma2 - logsigma2.detach()),
#             dim=-1)

# def test_approx_gaussian_cross_entropy_scaled_grads():
#     for func in [approx_gaussian_cross_entropy,
#                  functools.partial(approx_gaussian_cross_entropy_scaled_grads, max_grad=100)]:
#         mu1 = nn.Parameter(torch.tensor([[0.5, 0.9], [1., 1.]]), requires_grad=True)
#         mu2 = nn.Parameter(torch.tensor([[8.5, 8.9], [1., 0.9]]), requires_grad=True)
#         logsigma1 = nn.Parameter(torch.tensor([[0.]]), requires_grad=True)
#         logsigma2 = nn.Parameter(torch.tensor([[-0.3, -0.9], [-0.3, -0.9]]), requires_grad=True)
#         ce = func(mu1, logsigma1, mu2, logsigma2)
#         print(ce.detach().numpy())
#         ce.sum().backward()
#         print(mu1.grad.numpy(), 0, mu2.grad.numpy(), logsigma2.grad.numpy(), sep='\n')
#         print()


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min):
        return input.clamp(min=min)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min)


@torch.jit.script
def gaussian_log_prob(x, mu, logsigma):
    return -logsigma - 0.5*torch.tensor(2*np.pi).log() - 0.5 * ( (x - mu) / logsigma.exp() ) ** 2


@torch.jit.script
def gaussian_analytical_kl(mu1, logsigma1, mu2, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma, t=torch.tensor(1.)):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps * t + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False, mask=None):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    if mask is not None:
        mixture_probs = mixture_probs * mask
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)


class HModule(nn.Module):
    def __init__(self, H, *args, **kwargs):
        super().__init__()
        self.H = H
        self.build(*args, **kwargs)


class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x, **kwargs):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'], **kwargs)

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat


# define helper
def sample_patches_mask(H, b, h, w, device, n_patches=None):
    if n_patches is None:
        n_patches = torch.randint(0, H.max_patches+1, (b,))
    patch_dim = round(w * H.patch_size_frac)
    masks = torch.zeros(b, h, w, 1, device=device)
    for mask, n_p in zip(masks, n_patches):
        for p in range(n_p):
            r = torch.randint(-patch_dim+1, h, ())
            c = torch.randint(-patch_dim+1, h, ())
            r1 = max(0, r)
            c1 = max(0, c)
            r2 = min(h, r+patch_dim)
            c2 = min(w, c+patch_dim)
            mask[r1:r2, c1:c2] = 1.
    return masks

def channel_last_interpolate(t, *args, **kwargs):
    t = t.permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(t, *args, **kwargs)
    return t.permute(0, 2, 3, 1)

def sample_foveal(H, images):
    assert 'q_r2' not in H.kls
    b, h, w, _ = images.shape
    n_patches = torch.randint(0, H.max_patches+1, (b,))
    centres = [[(torch.randint(h, (1,)), torch.randint(w, (1,))) for _ in range(int(n_p))] for n_p in n_patches]
    emb = []
    for down_factor, size_frac in zip(H.foveal_down_factors, H.foveal_size_fracs):
        blurred_image = channel_last_interpolate(
            channel_last_interpolate(images, scale_factor=1/down_factor, mode='bilinear'),
            size=images.shape[1:-1], mode='bilinear')
        masks = torch.zeros(b, h, w, 1, device=images.device)
        width = round(w * size_frac)
        for i, centres_i in enumerate(centres):
            for r, c in centres_i:
                patch_dim = round(h*size_frac)
                minus = width // 2
                plus = width - minus
                r1 = max(0, r-minus)
                r2 = min(h, r+plus)
                c1 = max(0, c-minus)
                c2 = min(w, c+plus)
                masks[i, r1:r2, c1:c2] = 1.
        emb.append(torch.cat([blurred_image*masks, masks], dim=-1))
    return torch.cat(emb, dim=-1)

def sample_part_images(H, images, categories=None):
    b, h, w, _ = images.shape

    if H.conditioning == 'image':
        assert H.dataset in ['shoes', 'bags', 'shoes64', 'bags64']
        return images[..., :H.image_size, :]

    if H.conditioning == 'foveal':
        return sample_foveal(H, images)

    # sample mask
    if H.conditioning == 'patches':
        masks = sample_patches_mask(H, b, h, w, device=images.device, n_patches=categories)
    elif H.conditioning == 'patches-missing':
        inv = sample_patches_mask(H, b, h, w, device=images.device, n_patches=categories)
        masks = 1 - inv
    elif H.conditioning == 'blank':
        masks = torch.zeros_like(images[..., :1])
    elif H.conditioning == 'freeform':
        hole_range_dict = {0: [0, 0.2],
                           1: [0.2, 0.4],
                           2: [0.4, 0.6],
                           3: [0.6, 0.8],
                           4: [0.8, 1.0],}
        if categories is None:
            hole_ranges = [(0, 1),] * b
        else:
            hole_ranges = [hole_range_dict[int(c.item())] for c in categories]

        if hasattr(H, "mask_sample_workers") and H.mask_sample_workers is not None:
            p = mp.Pool(H.mask_sample_workers)
            masks = np.stack(p.starmap(RandomMask, [(h, hole_range, np.random.RandomState(np.random.randint(2**32-1))) for hole_range in hole_ranges]), axis=0)
        else:
            masks = BatchRandomMask(b, h, hole_ranges=hole_ranges)

        # - old stuff
        # masks = [torch.tensor(next(gen)).permute(1, 2, 0) for _ in range(b)]
        # masks = torch.stack(masks, dim=0).to(images.device)
        masks = torch.tensor(masks).permute(0, 2, 3, 1).to(images.device)

    emb = torch.cat([images*masks, masks], dim=-1)
    return emb


# RNG --------------------------------------------------

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


class RNG():

    def __init__(self, seed=None, state=None):

        self.state = get_random_state()
        with self:
            if seed is not None:
                set_random_seed(seed)
            elif state is not None:
                set_random_state(state)

    def __enter__(self):
        self.external_state = get_random_state()
        set_random_state(self.state)

    def __exit__(self, *args):
        self.state = get_random_state()
        set_random_state(self.external_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

class rng_decorator():

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            with RNG(self.seed):
                return f(*args, **kwargs)

        return wrapped_f