import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
from vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl, gaussian_log_prob, dclamp
from collections import defaultdict
import numpy as np
import itertools
import wandb


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x[..., -self.H.image_size:, :]   # does nothing unless it is edges2shoes or similar
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)  # encoder that takes full image
        self.part_enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        # still need prior if H is conditional, as it includes part of decoder architecture
        self.prior = Block(width, cond_width, H.zdim * 2 + width,
                           residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def z_fn(self, x):
        return self.z_proj(x)

    def new_sample(self, x, sample_from=None, part_acts=None, full_acts=None, t=None, lvs=None, get_kl=None, get_ents=False):
        feats = self.prior(x)
        priorm, priorv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        if part_acts is not None:
            part_params = self.part_enc(torch.cat([x, part_acts], dim=1))
        if full_acts is not None:
            fullm, fullv = self.enc(torch.cat([x, full_acts], dim=1)).chunk(2, dim=1)

        # sampling
        t = torch.tensor(t if t is not None else 1)
        if sample_from == 'fixed':
            z = lvs
        elif sample_from == 'prior':
            z = draw_gaussian_diag_samples(priorm, priorv, t)
        elif sample_from == 'part':
            z = draw_gaussian_diag_samples(*part_params.chunk(2, dim=1), t)
        elif sample_from == 'full':
            z = draw_gaussian_diag_samples(fullm, fullv, t)
        else:
            raise Exception(f"Invalid sample_from, {sample_from}.")

        if get_kl == 'full-prior':
            kl = gaussian_analytical_kl(fullm, fullv, priorm, priorv)
        elif get_kl == 'full-part':
            kl = gaussian_analytical_kl(fullm, fullv, *part_params.chunk(2, dim=1))
        elif get_kl == 'noisy-full-part':
            assert sample_from == 'full'
            kl = gaussian_log_prob(z, fullm, fullv) - gaussian_log_prob(z, *part_params.chunk(2, dim=1))
        elif get_kl == 'prior-part':
            kl = gaussian_analytical_kl(priorm, priorv, *part_params.chunk(2, dim=1))
        elif get_kl == 'part-prior':
            kl = gaussian_analytical_kl(*part_params.chunk(2, dim=1), priorm, priorv)
        elif get_kl == 'nll-part':
            # return -ve log likelihood of the part encoder's distribution
            m, v = part_params.chunk(2, dim=1)
            if self.H.clamp_std is not None:
                v = dclamp(v, min=self.H.clamp_std)
            ll = gaussian_log_prob(z, m, v)
            kl = -ll
        elif get_kl == 'nll-prior':
            ll = gaussian_log_prob(z, priorm, priorv)
            kl = -ll
        elif get_kl == 'nll-full':
            ll = gaussian_log_prob(z, fullm, fullv)
            kl = -ll
        else:
            assert get_kl is None, f'Unrecognised kl, {get_kl}.'

        # Skip connection
        x = x + xpp

        returns_dict = {}
        if get_kl is not None:
            returns_dict['kl'] = kl
        if get_ents:
            def logs2ent(logs):
                return logs+0.5*np.log(2*np.pi*np.e)
            if part_acts is None:
                part_ent = 0.
            else:
                part_ent = logs2ent(list(part_params.chunk(2, dim=1))[1]).sum(dim=1)
            if part_ent is not None:
                prior_ent = logs2ent(priorv).sum(dim=1)
                full_ent = logs2ent(fullv).sum(dim=1) if full_acts is not None else 0.
                returns_dict['ents'] = (part_ent, full_ent, prior_ent) #  tuple(logs+0.2*np.log(2*np.pi*np.e) for logs in [partvm, fullvm, priorvm])
        return z, x, returns_dict

    def get_inputs(self, xs, *activationses):
        if all(acts is None for acts in activationses):
            try:
                x = xs[self.base]
            except KeyError:
                x = torch.zeros(1, self.widths[self.base], self.base, self.base,
                                device=xs[1].device)
            return (x,) + activationses
        actses = [activations[self.base] if activations is not None else None
                  for activations in activationses]
        acts = next(acts for acts in actses if acts is not None)
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return (x,) + tuple(actses)

    def forward(self, xs, part_activations=None, full_activations=None, get_latents=False, **kwargs):
        x, part_acts, full_acts = self.get_inputs(xs, part_activations, full_activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, stats = self.new_sample(x=x, part_acts=part_acts, full_acts=full_acts, **kwargs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            stats['z'] = z.detach()
        return xs, stats


class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])

        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))

    def final_fn(self, x):
        return x * self.gain + self.bias


    def run(self, sample_from, get_kl=None, part_activations=None,
            full_activations=None, n=None, get_latents=False,
            manual_latents=(), t=None, get_ents=False):

        xs = {}
        stats = []
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(1 if n is None else n, 1, 1, 1)
        for idx, (lvs, block) in enumerate(itertools.zip_longest(manual_latents, self.dec_blocks)):
            try:
                temp = t[idx]
            except TypeError:
                temp = t

            xs, block_stats = block(xs, get_kl=get_kl,
                                    part_activations=part_activations,
                                    full_activations=full_activations,
                                    sample_from=sample_from if lvs is None else
                                    'fixed', get_latents=get_latents, t=temp,
                                    lvs=lvs, get_ents=get_ents)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats


    def forward(self, part_activations, full_activations,
                n=None, get_latents=False, get_ents=False):
        return self.run(sample_from='full',
                        get_kl='noisy-full-part' if self.H.noisy_kl else 'full-part',
                        part_activations=part_activations,
                        full_activations=full_activations, n=None,
                        get_latents=get_latents, get_ents=get_ents)

    def forward_uncond(self, part_activations, n, t=None, y=None, get_latents=False):
        return self.run(sample_from='part', get_kl=None,
                        part_activations=part_activations, n=n, t=t,
                        get_latents=get_latents)

    def forward_manual_latents(self, part_activations, n, latents, t=None):
        output, stats = self.run(sample_from='part', part_activations=part_activations,
                                 n=n, manual_latents=latents, t=t, get_kl=None)
        return output


class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, full_x, x_target):
        full_x = full_x[..., -self.H.image_size:, :]   # does nothing unless it is edges2shoes or similar
        x_target = x_target[..., -self.H.image_size:, :]   # does nothing unless it is edges2shoes or similar
        full_activations = self.encode_full_image(full_x)
        # px_z, stats = self.decoder.forward(NullTensorDict(full_x.shape), full_activations)
        px_z, stats = self.decoder.run(sample_from='full',
                                       get_kl='noisy-full-prior' if self.H.noisy_kl else 'full-prior',
                                       full_activations=full_activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(full_x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = distortion_per_pixel + rate_per_pixel
        return dict(elbo=elbo.mean(), loss=elbo.mean(), distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean(), batch_size=len(elbo))

    def encode_full_image(self, images):
        return self.encoder(images)

    def forward_get_latents(self, full_x):
        full_activations = self.encoder.forward(full_x)
        _, stats = self.decoder.forward(NullTensorDict(full_x.shape), full_activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z, _ = self.decoder.forward_uncond(NullTensorDict((n_batch,)), n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(NullTensorDict((n_batch,)), n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)


class NullTensorDict():
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, i):
        return torch.Tensor(self.shape)


class ConditionalVAE(HModule):
    def build(self):
        self.decoder = Decoder(self.H)
        if not self.H.share_encoders:
            self.encoder = Encoder(self.H)
        H_ = self.H.copy()
        if self.H.conditioning == 'image':
            cond_channels = self.H.image_channels
        else:
            cond_channels = self.H.image_channels + 1
        H_.image_channels = cond_channels
        self.part_encoder = Encoder(H_)

    def encode_part_image(self, part_x):
        return self.part_encoder(part_x)

    def encode_full_image(self, images):
        if self.H.share_encoders:
            images = torch.cat([images, torch.ones_like(images[..., :1])], dim=-1)
            return self.part_encoder(images)
        else:
            return self.encoder(images)

    def kl_q_r2(self, part_x, full_x, x_target):

        full_activations = self.encode_full_image(full_x)
        part_activations = self.encode_part_image(part_x)

        px_z, stats = self.decoder.run(sample_from='part', get_kl='part-prior',
                                       part_activations=part_activations,
                                       full_activations=full_activations)
        mask = part_x[..., -1]
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target, mask=mask)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(full_x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = distortion_per_pixel + rate_per_pixel
        schedule_iters, start_inv_temp, final_inv_temp = self.H.likelihood_temp_schedule
        if self.iterate < schedule_iters:
            prop = self.iterate / schedule_iters
            likelihood_inv_temp = start_inv_temp ** (1-prop) * final_inv_temp ** prop
        else:
            likelihood_inv_temp = final_inv_temp
        loss = distortion_per_pixel * likelihood_inv_temp + rate_per_pixel
        return dict(loss=loss.mean(), elbo=elbo.mean(),
                    distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())

    def kl_r1_q(self, part_x, full_x, x_target):

        full_activations = self.encode_full_image(full_x)
        part_activations = self.encode_part_image(part_x)
        px_z, stats = self.decoder.forward(part_activations, full_activations)
        rate_per_pixel = 0
        ndims = np.prod(full_x.shape[1:])
        kls = {}
        for i_layer, statdict in enumerate(stats):
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
            kls[i_layer+1] = statdict['kl'].sum(dim=(1, 2, 3,)).mean()
        rate_per_pixel /= ndims
        if self.H.train_encoder_decoder:
            if self.H.mask_distortion:
                distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target, mask=1-part_x[..., -1])
            else:
                distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
            elbo = distortion_per_pixel + rate_per_pixel
            logged_distortion = distortion_per_pixel.mean()
        else:
            elbo = rate_per_pixel
            logged_distortion = rate_per_pixel.mean()*0
        return dict(loss=elbo.mean(), elbo=elbo.mean(), distortion=logged_distortion,
                    rate=rate_per_pixel.mean(), kl1=kls[1], kl2=kls[2], kl3=kls[3])

    def forward(self, part_x=None, full_x=None, x_target=None, obj='r1_q', iterate=None):
        full_x = full_x[..., -self.H.image_size:, :]   # does nothing unless it is edges2shoes or similar
        x_target = x_target[..., -self.H.image_size:, :]   # does nothing unless it is edges2shoes or similar
        self.iterate = iterate
        if obj == 'r1_q':
            return self.kl_r1_q(part_x, full_x, x_target)
        elif obj == 'q_r2':
            return self.kl_q_r2(part_x, full_x, x_target)
        else:
            raise Exception('KL not recognised.')

    def forward_get_latents(self, part_x, full_x, get_ents=False):
        full_activations = self.encode_full_image(full_x)
        part_activations = self.encode_part_image(part_x)
        _, stats = self.decoder.forward(part_activations, full_activations,
                                        get_latents=True, get_ents=get_ents)
        return stats

    def forward_uncond_samples(self, n_batch, part_x, t=None):
        assert n_batch == part_x.shape[0]
        part_activations = self.encode_part_image(part_x)
        px_z, _ = self.decoder.forward_uncond(part_activations, n=None, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, part_x, latents, t=None):
        assert n_batch == part_x.shape[0]
        part_activations = self.encode_part_image(part_x)
        px_z = self.decoder.forward_manual_latents(part_activations, n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)
