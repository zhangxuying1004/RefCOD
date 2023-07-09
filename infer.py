'''
the codes for evaluating the model.
created by Xuying Zhang (zhangxuying1004@gmail.com) on 2021-06-23
'''

import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from models.r2cnet import Network
from data import get_dataloader


def gen_maps(model, test_loader, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, name) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)                     # 标准化处理,把数值范围控制到(0,1)

                sal_f = sal_f.cuda()
                res, _ = model(x=image, ref_x=sal_f)
         
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                cv2.imwrite(os.path.join(target_dir, name[0]+'.png'), res*255)
              
                pbar.update()

def load_model_params(model, params_path):
    assert os.path.exists(params_path)
    checkpoints = torch.load(params_path)
    # print(checkpoints['epoch'])

    model.load_state_dict(checkpoints['state_dict'])
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='r2cnet')
    parser.add_argument('--dim', type=int, default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--shot', type=int, default=5)
  
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')

    parser.add_argument('--data_root', type=str, default='/home/zhangxuying/Datasets/R2C7K/', help='the path to put dataset')  
    parser.add_argument('--save_root', type=str, default='./snapshot/', help='the path to save model params and log')

    opt = parser.parse_args()
    print(opt)

    # load model
    ref_model = Network(channel=opt.dim, imagenet_pretrained=False).cuda()
    params_path = os.path.join(opt.save_root, 'saved_models', '{}.pth'.format(opt.model_name))  # './snapshot/saved_models/r2cnet.pth'
    ref_model = load_model_params(ref_model, params_path)

    # load data
    test_loader = get_dataloader(opt.data_root, opt.shot, opt.trainsize, opt.num_workers, mode='test')

    # where to save maps
    target_dir = os.path.join(opt.save_root, 'preds', opt.model_name)

    # processing
    gen_maps(ref_model, test_loader, target_dir)
