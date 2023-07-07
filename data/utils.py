import os
import numpy as np
import random
from tqdm import tqdm
import json

from PIL import Image, ImageEnhance
import cv2
import torch


# read image
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# several data augumentation strategies
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class Normalize(object):
    def __init__(self):
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
    
    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        return image, mask 
    

# -----------------------------------

def split_ref_data(data_root, record_file='./data/refsplits.json'):
    '''
    给定收集好的Ref图片，
    划分训练和测试集，记录划分情况
    '''
    assert os.path.exists(data_root)
    os.makedirs('/'.join(record_file.split('/')[:-1]), exist_ok=True)

    refsplits = {
        'train': {},
        "test": {}
    }

    ref_image_root = os.path.join(data_root, 'Ref', 'Images')
    assert os.path.exists(ref_image_root)
    cates = os.listdir(ref_image_root)

    for cate in cates:
        ref_cate_image_dir = os.path.join(ref_image_root, cate)
        ref_cate_image_names = os.listdir(ref_cate_image_dir)
        assert len(ref_cate_image_names) == 25
        random.shuffle(ref_cate_image_names)
        ref_cate_train_samples = [name[:-4] for name in ref_cate_image_names[:20]]
        ref_cate_test_samples = [name[:-4] for name in ref_cate_image_names[20:]]

        refsplits['train'][cate] = ref_cate_train_samples
        refsplits['test'][cate] = ref_cate_test_samples

    with open(record_file, 'w') as f:
        json.dump(refsplits, f, indent=4)


def collect_r2c_data(data_root, mode='train', record_file='./data/refsplits.json'):

    if not os.path.exists(record_file):
        split_ref_data(data_root, record_file) 

    assert os.path.exists(data_root)
    camo_Imgs_dir = os.path.join(data_root, 'Camo', mode if mode != 'val' else 'test', 'Imgs')
    camo_gts_dir = os.path.join(data_root, 'Camo', mode if mode != 'val' else 'test', 'GT')
    assert os.path.exists(camo_Imgs_dir) and os.path.exists(camo_gts_dir)

    ref_feats_dir = os.path.join(data_root, 'Ref', 'RefFeat_ICON-R')
    assert os.path.exists(ref_feats_dir)
    
    camo_classes = os.listdir(camo_Imgs_dir)
    ref_classes = os.listdir(ref_feats_dir)

    assert len(camo_classes) == len(ref_classes) == 64

    with open(record_file, 'r') as f:
        splits = json.load(f)

    image_label_list = []
    class_file_list = {}
    for c_idx in tqdm(range(len(camo_classes))):
        cate = camo_classes[c_idx]

        camo_cate_Imgs_dir = os.path.join(camo_Imgs_dir, cate)
        camo_cate_gts_dir = os.path.join(camo_gts_dir, cate)
        camo_img_names = sorted(os.listdir(camo_cate_Imgs_dir))
        camo_gt_names = sorted(os.listdir(camo_cate_gts_dir))
        assert len(camo_img_names) == len(camo_gt_names)

        image_label_list += [(os.path.join(camo_cate_Imgs_dir, camo_img_names[f_idx]), os.path.join(camo_cate_gts_dir, camo_gt_names[f_idx])) for f_idx in range(len(camo_img_names))]

        ref_cate_feats_dir = os.path.join(ref_feats_dir, cate)

        ref_cate_split_names = splits[mode if mode != 'val' else 'test'][cate]
        class_file_list[cate] = [os.path.join(ref_cate_feats_dir, ref_cate_split_names[f_idx]+'.npy') for f_idx in range(len(ref_cate_split_names))]

    print('>>> {}ing with {} r2c samples'.format(mode, len(image_label_list)))
    
    return image_label_list, class_file_list
