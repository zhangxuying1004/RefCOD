'''
the codes for training the model.
created by Xuying Zhang (zhangxuying1004@gmail.com) on 2023-06-23
'''

import os
import numpy as np
from time import time
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

import utils.metrics as Measure
from utils.utils import set_gpu, structure_loss, clip_gradient

from models.r2cnet import Network
from data import get_dataloader


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, supp_feats, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            supp_feats = supp_feats.cuda()

            preds, inner_preds = model(images, supp_feats)
            main_loss = structure_loss(preds, gts)
            aux_loss = structure_loss(inner_preds[0], gts)
            inner_num = len(inner_preds)
            for inner_idx in range(1, inner_num):
                aux_loss = aux_loss + structure_loss(inner_preds[inner_idx], gts)
            aux_loss /= inner_num
            loss = main_loss + aux_loss

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 10 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch
        }, save_path + 'Net_Interrupt_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch, best_score, best_other_epoch
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sf, _) in test_loader:
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)                     # 标准化处理,把数值范围控制到(0,1)
                image = image.cuda()
                sf = sf.cuda()
                res, _ = model(image, sf)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                WFM.step(pred=res*255, gt=gt*255)
                SM.step(pred=res*255, gt=gt*255)
                EM.step(pred=res*255, gt=gt*255)
                MAE.step(pred=res*255, gt=gt*255)
              
                pbar.update()

        sm1 = SM.get_results()['sm'].round(3)
        adpem1 = EM.get_results()['em']['adp'].round(3)
        wfm1 = WFM.get_results()['wfm'].round(3)
        mae1 = MAE.get_results()['mae'].round(3)

        writer.add_scalar('Sm', torch.tensor(sm1), global_step=epoch)
        writer.add_scalar('adaEm', torch.tensor(adpem1), global_step=epoch)
        writer.add_scalar('wF', torch.tensor(wfm1), global_step=epoch)
        writer.add_scalar('MAE', torch.tensor(mae1), global_step=epoch)

        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))

        if epoch == 1:
            best_mae = mae
            best_score = score
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch
                }, save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
            
            score = sm1 + adpem1 + wfm1
            if score > best_score:
                best_score = score
                best_other_epoch = epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch
                }, save_path + 'Net_epoch_other_best.pth')
                print('Save state_dict successfully! Best other epoch:{}.'.format(epoch))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='r2cnet')
    parser.add_argument('--epoch', type=int, default=55, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--trainsize', type=int, default=352, help='training image size')
    parser.add_argument('--shot', type=int, default=5, help='number of referring images')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--data_root', type=str, default='/home/zhangxuying/Datasets/R2C7K', help='the path to put dataset')  
    parser.add_argument('--save_root', type=str, default='./snapshot/', help='the path to save model params and log')

    opt = parser.parse_args()
    print(opt)

    # set the device for training
    set_gpu(opt.gpu_id)
    cudnn.benchmark = True

    start_time = time()

    model = Network(channel=opt.dim).cuda()
    base, body = [], []
    for name, param in model.named_parameters():
        if 'resnet' in name:
            base.append(param)   
        else:
            body.append(param)

    params_dict = [{'params': base, 'lr': opt.lr * 0.1},              
                {'params': body, 'lr': opt.lr}]
    optimizer = torch.optim.Adam(params_dict)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    print('load data...')
    train_loader = get_dataloader(opt.data_root, opt.shot, opt.trainsize, opt.batchsize, opt.num_workers, mode='train')
    val_loader = get_dataloader(opt.data_root, opt.shot, opt.trainsize, opt.num_workers, mode='val')
    total_step = len(train_loader)

    save_path = opt.save_root + 'saved_models/' + opt.model_name + '/'
    save_logs_path = opt.save_root + 'logs/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_logs_path, exist_ok=True)

    writer = SummaryWriter(save_logs_path + opt.model_name)
    
    step = 0

    best_mae = 1
    best_epoch = 0

    best_score = 0.
    best_other_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('lr_base', cosine_schedule.get_lr()[0], global_step=epoch)
        writer.add_scalar('lr_body', cosine_schedule.get_lr()[1], global_step=epoch)

        # train
        train(train_loader, model, optimizer, epoch, save_path, writer)
        # val
        val(val_loader, model, epoch, save_path, writer)

    end_time = time()

    print('it costs {} h to train'.format((end_time - start_time)/3600))
