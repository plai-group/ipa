import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        size = opt.img_size

        ## Encoder ##
        # input is (nc) x s x s  (s:= opt.img_size)
        layers = [nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
        size //= 2
        c_scale = 1 # scaling in the number of channels
        # state size: (opt.nef*c_scale) x size x size
        while size > 4:
            c_scale_next = c_scale * 2 if size <= 32 else c_scale
            layers.extend([
                nn.Conv2d(opt.nef*c_scale, opt.nef*c_scale_next , 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.nef*c_scale_next),
                nn.LeakyReLU(0.2, inplace=True)
                ])
            size //= 2
            c_scale = c_scale_next
            # state size: (opt.nef*c_scale) x size x size
        assert size == 4
        # state size: (opt.nef*c_scale) x 4 x 4
        layers.extend([
            nn.Conv2d(opt.nef*c_scale, opt.nBottleneck, 4, bias=False),
            nn.BatchNorm2d(opt.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        # state size: (opt.nBottleneck) x 1 x 1

        ## Decoder ##
        layers.extend([
            nn.ConvTranspose2d(opt.nBottleneck, opt.ngf*c_scale, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf*c_scale),
            nn.ReLU(True)
        ])
        size = 4
        # state size: (channels*c_scale) x 4 x 4
        while(size < opt.img_size // 2):
            c_scale_next = c_scale // 2 if c_scale > 1 else c_scale
            layers.extend([
                nn.ConvTranspose2d(opt.ngf*c_scale, opt.ngf*c_scale_next, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.ngf*c_scale_next),
                nn.ReLU(True)
            ])
            size *= 2
            c_scale = c_scale_next
            # state size: (channels*c_scale) x size x size
        assert size == opt.img_size // 2
        assert c_scale == 1
        layers.extend([
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        # state size: (opt.nc) x opt.img_size x opt.img_size

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        size = opt.img_size

        ## Encoder ##
        # input is (nc) x s x s  (s:= opt.img_size)
        layers = [nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
        size //= 2
        c_scale = 1 # scaling in the number of channels
        # state size: (opt.ndf*c_scale) x size x size
        while size > 4:
            c_scale_next = c_scale * 2 if size <= 32 else c_scale
            layers.extend([
                nn.Conv2d(opt.ndf*c_scale, opt.ndf*c_scale_next , 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.ndf*c_scale_next),
                nn.LeakyReLU(0.2, inplace=True)
                ])
            size //= 2
            c_scale = c_scale_next
            # state size: (opt.ndf*c_scale) x size x size
        assert size == 4
        # state size: (opt.ndf*c_scale) x 4 x 4
        layers.extend([
            nn.Conv2d(opt.ndf*c_scale, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

