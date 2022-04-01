import argparse
import os, sys
from model import RFRNetModel
#from dataset import Dataset
from torch.utils.data import DataLoader

from data import cifar10, ffhq256
import torch
import torchvision.transforms as transforms
from vae_helpers import sample_part_images, rng_decorator
from vae_helpers.baseline_utils import update_args
import numpy as np
import wandb


PROJECT_NAME = 'RFR-Inpainting'
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * bytes

    return image_numpy.astype(imtype)


def sample_mask(args, batch):
    # args shoudld have the following attributes:
    # conditioning, max_patches, patch_size_frac, and kls (only for foveal conditioning)
    x = sample_part_images(args, batch.permute(0, 2, 3, 1))[..., -1]
    x = x.unsqueeze(1)
    x = x.repeat(1, 3, 1, 1)
    return x.contiguous()


@rng_decorator(0)
@torch.no_grad()
def log_images(args, model, viz_batch_processed):
    def inpaint(gt_images, masks):
        masked_images = gt_images * masks
        recon, mask = model(masked_images, masks, rounds=-1)
        inpainted = recon * (1 - masks) + gt_images * masks
        return inpainted.cpu().numpy()
    model.eval()
    masks = sample_mask(args, viz_batch_processed)
    log_dict = {}
    for idx in range(len(viz_batch_processed)):
        to_plot = [(viz_batch_processed[idx] * masks[idx]).cpu().numpy(),
                   viz_batch_processed[idx].cpu().numpy()]
        for _ in range(args.num_samples_visualize):
            to_plot.append(np.clip(inpaint(viz_batch_processed[idx].unsqueeze(0), masks[idx].unsqueeze(0)).squeeze(0), 0, 1))
        to_plot = np.concatenate(to_plot, axis=-1)
        to_plot = to_plot.transpose(1,2,0)
        caption = f"Sample {idx}"
        log_dict.update({caption: wandb.Image(to_plot, caption=caption)})
    wandb.log(log_dict)
    model.train()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform, args):
        super().__init__()
        self.args = args
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        return img, sample_mask(self.args, img.unsqueeze(0)).squeeze(0) # Returns the image and a mask

    def __len__(self):
        return len(self.data)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'ffhq256'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="checkpoint/100000.pth")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--rounds', type=int, default=-1)
    parser = update_args(parser)
    args = parser.parse_args()

    wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
               config=args, tags=args.tags)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()

    transform = [transforms.ToTensor()]
    if args.dataset == "cifar10":
        transform.append(transforms.Resize(256))
        (trX, _), (vaX, _), (teX, _) = cifar10(args.data_root, one_hot=False)
    elif args.dataset == "ffhq256":
        trX, vaX, teX = ffhq256(args.data_root)
    transform = transforms.Compose(transform)
    train_set = MyDataset(trX, transform, args)
    valid_set = MyDataset(vaX, transform, args)
    viz_batch_processed = next(iter(DataLoader(valid_set, batch_size=args.num_images_visualize)))[0]
    viz_batch_processed = viz_batch_processed.to("cuda")

    if args.test:
        assert False
        model.initialize_model(args.model_path, False)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True, lr=args.lr)
        model.cuda()
        #dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(args, dataloader, args.model_save_path, lambda x: log_images(args, x, viz_batch_processed), args.finetune)

if __name__ == '__main__':
    run()