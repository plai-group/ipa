from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
import wandb
from vae_helpers import rng_decorator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data import cifar10, ffhq256
from vae_helpers import sample_part_images
from vae_helpers.baseline_utils import update_args
import time

from model import _netlocalD,_netG
import utils


PROJECT_NAME = 'context-encoders'
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"

device = "cuda" if torch.cuda.is_available else "cpu"


@rng_decorator(0)
@torch.no_grad()
def log_images(opt, netG, viz_batch_processed):
    old_mode = netG.training
    netG.training = False
    mask = sample_mask(opt, viz_batch_processed)
    masked = apply_mask(viz_batch_processed.clone(), mask)
    recon = netG(masked)
    mask_expanded = mask.unsqueeze(1).expand_as(recon).bool()
    inpainted = recon * (~mask_expanded) + viz_batch_processed * mask_expanded
    # Normalize the pixel values to [0-1]
    viz_batch_processed = (viz_batch_processed.cpu() + 1) / 2
    masked = (masked.cpu() + 1) / 2
    inpainted = (inpainted.cpu() + 1) / 2
    # Make numpy grids out of the image tensors
    real_img = vutils.make_grid(viz_batch_processed, nrow=4, range=(0, 1)).permute(1,2,0).numpy()
    masked_img = vutils.make_grid(masked, nrow=4, range=(0, 1)).permute(1,2,0).numpy()
    inpainted_img = vutils.make_grid(inpainted, nrow=4, range=(0, 1)).permute(1,2,0).numpy()
    log_dict = {"real": wandb.Image(real_img, caption="real"),
                "cropped": wandb.Image(masked_img, caption="cropped"),
                "inpainted": wandb.Image(inpainted_img, caption="inpainted")}
    netG.training = old_mode
    return log_dict


def sample_mask(opt, batch):
    # opt shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(opt, batch.permute(0, 2, 3, 1))[..., -1]
    return x.contiguous()

def apply_mask(img, mask):
    mask = mask.bool()
    img[:, 0][~mask] = (2*117.0/255.0 - 1.0)
    img[:, 1][~mask] = (2*104.0/255.0 - 1.0)
    img[:, 2][~mask] = (2*123.0/255.0 - 1.0)
    return img


class MaskDataset(Dataset):
    def __init__(self, args, img_shape):
        super().__init__()
        self.example_batch = torch.zeros(1, *img_shape)
        self.args = args

    def __len__(self):
        return 1000 * 1000 * 1000

    def __getitem__(self, idx):
        return sample_mask(self.args, self.example_batch).numpy()[0]


def _infinite_loader(loader):
        while True:
            for x in loader:
                yield x


class MaskLoader:
    def __init__(self, mask_dataset, num_workers):
        self.loader = _infinite_loader(DataLoader(mask_dataset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  num_workers=num_workers,
                                                  prefetch_factor=8))
    
    def get_batch(self, batch_size):
        b = 0
        mask_parts = []
        while b < batch_size:
            m = next(self.loader)
            mask_parts.append(m)
            b += len(m)
        return torch.cat(mask_parts, dim=0)[:batch_size]


class MyDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), 0 # Returns the image and a dummy label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'ffhq256'], required=True)
    parser.add_argument('--dataroot',  default='dataset/cifar10', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
    parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
    parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
    parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
    parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
    parser.add_argument('--img_size', type=int)
    parser = update_args(parser)
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for mask_generator sampler. If not given, the main process will sample the masks. Recommended: 8")

    opt = parser.parse_args()
    if opt.img_size is None:
        opt.img_size = 32 if opt.dataset == "cifar10" else 256
    print(opt)
    wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
               config=opt, tags=opt.tags)

    try:
        os.makedirs("result/train/cropped")
        os.makedirs("result/train/real")
        os.makedirs("result/train/recon")
        os.makedirs("model")
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    transform = [transforms.ToTensor(),
                 #transforms.Resize(opt.img_size),
                 transforms.Normalize(0.5, 0.5)]
    if opt.dataset == "cifar10":
        (trX, _), (vaX, _), (teX, _) = cifar10(opt.data_root, one_hot=False)
    elif opt.dataset == "ffhq256":
        trX, vaX, teX = ffhq256(opt.data_root)
    transform = transforms.Compose(transform)
    train_set = MyDataset(trX, transform)
    valid_set = MyDataset(vaX, transform)
    dataloader = DataLoader(train_set, batch_size=opt.batchSize,
                            shuffle=True, num_workers=int(opt.workers),
                            drop_last=True)
    viz_batch_processed = next(iter(DataLoader(valid_set, batch_size=16)))[0]
    viz_batch_processed = viz_batch_processed.to(device)


    mask_dataset = MaskDataset(opt, train_set[0][0].shape)
    mask_loader = MaskLoader(mask_dataset, opt.num_workers)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
    nef = int(opt.nef)
    nBottleneck = int(opt.nBottleneck)
    wtl2 = float(opt.wtl2)
    overlapL2Weight = 10

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    resume_epoch=0

    netG = _netG(opt)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netG)['epoch']
    print(netG)


    netD = _netlocalD(opt)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netD)['epoch']
    print(netD)
    wandb.log({"nparams.netG": sum([p.numel() for p in netG.parameters() if p.requires_grad]),
               "nparams.netD": sum([p.numel() for p in netD.parameters() if p.requires_grad])})

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    #input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

    netD.to(device)
    netG.to(device)
    criterion.to(device)
    criterionMSE.to(device)
    #input_real, input_cropped,label = input_real.to(device),input_cropped.to(device), label.to(device)
    input_cropped,label = input_cropped.to(device), label.to(device)
    real_center = real_center.to(device)


    #input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    label = Variable(label)


    real_center = Variable(real_center)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    total_iteration = 0
    for epoch in range(resume_epoch, opt.niter):
        t_0 = time.time()
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input_real = real_cpu.to(device)
            real_center = real_cpu.to(device)
            input_cropped = real_cpu.to(device)
            mask = mask_loader.get_batch(len(input_cropped))
            input_cropped = apply_mask(input_cropped, mask)

            # train with real
            netD.zero_grad()
            label = input_real.new_ones(batch_size)

            output = netD(real_center)
            errD_real = criterion(output, label.unsqueeze(-1))
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            # noise.data.resize_(batch_size, nz, 1, 1)
            # noise.data.normal_(0, 1)
            fake = netG(input_cropped)
            label = input_real.new_zeros(batch_size)
            output = netD(fake.detach())
            errD_fake = criterion(output, label.unsqueeze(-1))
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label = input_real.new_ones(batch_size)
            output = netD(fake)
            errG_D = criterion(output, label.unsqueeze(-1))
            # errG_D.backward(retain_variables=True)

            # errG_l2 = criterionMSE(fake,real_center)
            wtl2Matrix = torch.ones_like(real_center) * wtl2 * overlapL2Weight
            wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred)] = wtl2
            
            errG_l2 = (fake-real_center).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()

            errG = (1-wtl2) * errG_D + wtl2 * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

            if total_iteration % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
                    % (epoch, opt.niter, i, len(dataloader),
                        errD.item(), errG_D.item(),errG_l2.item(), D_x.item(), D_G_z1.item(), ))
                log_dict = {"epoch": epoch, "Loss_D": errD.item(), "Loss_G_D": errG_D.item(),
                            "Loss_G_l2": errG_l2.item(), "l_D(x)": D_x.item(),
                            "l_D(G(z))": D_G_z1.item()}
                wandb.log(log_dict)

            if total_iteration == 0:
                log_dict = log_images(opt, netG, viz_batch_processed)
                log_dict.update({"epoch": epoch})
                wandb.log(log_dict)

            total_iteration += 1


        log_dict = log_images(opt, netG, viz_batch_processed)
        log_dict.update({"epoch_time": time.time() - t_0,
                         "epoch": epoch})
        wandb.log(log_dict)
        # do checkpointing
        model_path = os.path.join("checkpoints", wandb.run.id, 'model_{}.pt'.format(epoch))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'epoch':epoch+1,
                    'netG_state_dict':netG.state_dict(),
                    'netD_state_dict':netD.state_dict(),
                    'config': opt.__dict__},
                    model_path)
