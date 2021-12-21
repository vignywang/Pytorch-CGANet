"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc
import Constants
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def argument_Drive_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, Constants.Image_size)
    mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, Constants.Image_size)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask
def argument_CHASEDB_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, Constants.Image_size)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, Constants.Image_size)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask
def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, Constants.Image_size)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, Constants.Image_size)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask
def default_CHASEDB_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, Constants.Image_size)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
   # mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, Constants.Image_size)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask
def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode=='Hard':
        image_root = os.path.join(root_path, 'argtraining/images')
        gt_root = os.path.join(root_path, 'argtraining/1st_manual')
    else:
        image_root = os.path.join(root_path, 'training/images')
        gt_root = os.path.join(root_path, 'training/1st_manual')
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        if int(image_name.split('_')[0])>20:
            label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')
        else:
            label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.tif')
        images.append(image_path)
        masks.append(label_path)

  #  print(images, masks)

    return images, masks

def read_CHASEDB_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'Hard':
        image_root = os.path.join(root_path, 'argtraining/images')
        gt_root = os.path.join(root_path, 'argtraining/1st_manual')
    else:
        image_root = os.path.join(root_path, 'training/images')
        gt_root = os.path.join(root_path, 'training/1st_manual')
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '_1stHO.png')

        images.append(image_path)
        masks.append(label_path)

    #  print(images, masks)

    return images, masks



class ImageFolder(data.Dataset):

    def __init__(self,root_path, datasets='Messidor',  mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['CHASEDB','DRIVE'], \
            "the dataset should be in 'CHASEDB', 'DRIVE'."
        if self.dataset == 'DRIVE':
            self.images, self.labels = read_DRIVE_datasets(self.root, self.mode)
            if self.mode == 'Argument':
                self.loader = argument_Drive_loader
            else:
                self.loader = default_DRIVE_loader
        else:
            self.images, self.labels = read_CHASEDB_datasets(self.root, self.mode)
            if self.mode=='Argument':
                self.loader=argument_CHASEDB_loader
            else:
                self.loader = default_CHASEDB_loader



    def __getitem__(self, index):

        img, mask = self.loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        if self.mode=='Argument':
            return img, mask,self.images[index], self.labels[index]
        else:
            return img, mask


    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)