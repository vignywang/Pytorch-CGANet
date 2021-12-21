import torch
import torch.nn as nn
from torch.autograd import Variable as V
from scipy import ndimage
import cv2
import numpy as np
import Constants as config
from torch.optim import lr_scheduler
def get_scheduler(optimizer, epoch_count,maintain_epoch,decay_epoch):
    #define lr chage rule
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_count - maintain_epoch) / float(decay_epoch + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler
class Frame():
    def __init__(self, net, loss=None, lr=2e-4, evalmode=False):
        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.L1 = nn.L1Loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        self.scheduler = get_scheduler(self.optimizer,config.epoch_count,config.maintain_epoch,config.decay_epoch)
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (config.Image_size, config.Image_size))
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
       # print(img1.shape, img2.shape, img3.shape, img4.shape, img5.shape)
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)


    def optimize_edge(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        SR = pred
        GT = self.mask
        SR_flat = SR.view(SR.size(0), -1)
        GT_flat = GT.view(GT.size(0), -1)
        loss_all = self.loss(GT_flat, SR_flat)
        GT_edge_enhanced = ndimage.gaussian_laplace(np.squeeze(GT.cpu().detach().numpy()), sigma=0.2)
        GT_edge = torch.tensor(np.int64(GT_edge_enhanced > 0.001))
        edge_pre = torch.squeeze(SR)[GT_edge == 1]
        edge = torch.squeeze(GT)[GT_edge == 1]
        loss_edge = self.L1(edge_pre, edge)
        loss = loss_all + loss_edge*config.edge_loss_weight
        loss.backward()
        self.optimizer.step()
        return loss.data, pred

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
    def update_learning_rate(self):
        #Update learning rates for all the networks; called at the end of every epoch
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        return lr