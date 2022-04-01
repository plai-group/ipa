import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
import wandb


class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.loss_G_val = 0.0
        self.rounds_sum = 0
    
    def initialize_model(self, path=None, train=True, lr=1e-4):
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr=lr)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                #self.optm_G = optim.Adam(self.G.parameters(), lr=1e-4)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0
        
    def cuda(self):
        return self.to("cuda" if torch.cuda.is_available() else "cpu")
        
    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.G.to(self.device)
        if self.lossNet is not None:
            self.lossNet.to(self.device)
        return self
        
    def train(self, args, train_loader, save_path, log_images, finetune=False):
        log_every = 50
        self.G.train(finetune = finetune)
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 1e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        log_images(self.G)
        for epoch in range(args.epochs):
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                self.forward(masked_images, masks, gt_images, rounds=args.rounds)
                self.update_parameters()
                self.iter += 1
                
                if self.iter % log_every == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/log_every, int_time))
                    wandb.log({"iteration": self.iter,
                               "l1_loss": self.l1_loss_val/log_every,
                               "loss_G": self.loss_G_val,
                               "time_taken": int_time,
                               "epoch": epoch,
                               "rounds": self.rounds_sum/log_every})
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    self.loss_G_val = 0.0
                    self.rounds_sum = 0
            log_images(self.G)
            model_path = os.path.join("checkpoints", wandb.run.id, 'model_{}.pt'.format(epoch))
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({'generator': self.G.state_dict(),
                        'optimizer_G': self.optm_G.state_dict(),
                        'config': args.__dict__,
                        'epoch': epoch + 1},
                        model_path)
    def test(self, test_loader, result_save_path, rounds=-1):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        for items in test_loader:
            gt_images, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks]*3, dim = 1)
            fake_B, mask = self.G(masked_images, masks, rounds=rounds)
            comp_B = fake_B * (1 - masks) + gt_images * masks
            if not os.path.exists('{:s}/results'.format(result_save_path)):
                os.makedirs('{:s}/results'.format(result_save_path))
            for k in range(comp_B.size(0)):
                count += 1
                grid = make_grid(comp_B[k:k+1])
                file_path = '{:s}/results/img_{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)
                
                grid = make_grid(masked_images[k:k+1] +1 - masks[k:k+1] )
                file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)
    
    def forward(self, masked_image, mask, gt_image, rounds):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, mask_group = self.G(masked_image, mask, rounds=rounds)
        self.rounds_sum += len(mask_group)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = 0#self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        
        loss_G = (  tv_loss * 0.1
                  + style_loss * 180
                  + preceptual_loss * 0.1
                  + valid_loss * 1
                  + hole_loss * 6)
        
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        self.loss_G_val += loss_G.item()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            