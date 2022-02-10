from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
from networks import get_model
from frameworks.net import Frame
from loss import dice_bce_loss
from data import ImageFolder
import Constants
from tqdm import tqdm
from image_utils import getpatch4
os.environ['CUDA_VISIBLE_DEVICES'] = Constants.GPU_id

def Vessel_Seg_Train(net):
    NAME = Constants.Name
    solver = Frame(net, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    dataset = ImageFolder(root_path=Constants.ROOT, datasets=Constants.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)
    total_epoch = Constants.TOTAL_EPOCH
    for epoch in range(1, total_epoch + 1):
        if (epoch > Constants.Argument_start and epoch % Constants.Argument_epoch == 0) or epoch == Constants.Argument_start:
            print('Argumenting!')
            val_dataset = ImageFolder(root_path=Constants.ROOT, datasets=Constants.dataset, mode='Argument')
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=4)
            val_bar = tqdm(val_loader)
            id_s = 0
            for img_tensor, mask_tensor, img_path, mask_path in val_bar:
                id_s += 1
                ground_truth = mask_tensor.squeeze().cpu().data.numpy()
                img = np.array(Image.open(img_path[0]))
                img = cv2.resize(img, Constants.Image_size)
                if Constants.dataset=='Drive':
                    mask = np.array(Image.open(mask_path[0]))
                else:
                    mask = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, Constants.Image_size)
                mask[mask >= 128] = 255
                mask[mask < 128] = 0
                pre = solver.test_one_img(img_tensor)
                data_img, data_mask = getpatch4(pre, ground_truth, mask, img)
                if Constants.dataset=='Drive':
                    data_img.save(Constants.Hard_ROOT + "/images/" + str(id_s) + "_training.tif")
                    data_mask.save(Constants.Hard_ROOT + "/1st_manual/" + str(id_s) + "_manual1.tif")
                else:
                    data_img.save(Constants.Hard_ROOT + "/images/" + str(id_s) + ".jpg")
                    data_mask.save(Constants.Hard_ROOT + "/1st_manual/" + str(id_s) + "_1stHO.png")
                train_dataset2 = ImageFolder(root_path=Constants.ROOT, datasets=Constants.dataset, mode='Hard')
                train_loader2 = torch.utils.data.DataLoader(
                    train_dataset2,
                    batch_size=batchsize,
                    shuffle=True,
                    num_workers=4)
        if epoch<=Constants.Argument_start:
            train_bar = tqdm(data_loader)
            train_epoch_loss = 0
            for img, mask in train_bar:
                solver.set_input(img, mask)
                if Constants.edge==True:
                    train_loss, pred = solver.optimize_edge()
                else:
                    train_loss, pred = solver.optimize()
                train_epoch_loss += train_loss
                train_bar.set_description(desc='[%d/%d]  loss: %.4f ' % (
                    epoch, total_epoch, train_loss))
            train_epoch_loss = train_epoch_loss / len(data_loader)
            current_lr = solver.update_learning_rate()
            print('mean_loss:', train_epoch_loss.cpu().numpy(),'lr:',current_lr)
        elif epoch>Constants.Argument_start:
            train_bar2 = tqdm(train_loader2)
            for img, mask in train_bar2:
                solver.set_input(img, mask)
                if Constants.edge == True:
                    train_loss, pred = solver.optimize_edge()
                else:
                    train_loss, pred = solver.optimize()
                train_epoch_loss += train_loss
                train_bar.set_description(desc='[%d/%d]  loss: %.4f ' % (
                    epoch, total_epoch, train_loss))
            train_epoch_loss = train_epoch_loss / len(data_loader)
            current_lr = solver.update_learning_rate()
            print('mean_loss:', train_epoch_loss.cpu().numpy(), 'lr:', current_lr)
        if  epoch%20==0:
            solver.save(Constants.weight_root + NAME +str(epoch)+ '.th')
    solver.save(Constants.weight_root + NAME + '.th')
if __name__ == '__main__':
    net=get_model(Constants.net)
    Vessel_Seg_Train(net)

