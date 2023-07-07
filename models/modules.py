import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import ConvLSTMCell, BasicConv2d


class FeatFusion(nn.Module):
    '''
    Dual-source Information Fusion (DSF) + Multi-scale Feature Fusion (MSF)
    '''
    def __init__(self, channel=64):
        super(FeatFusion, self).__init__()

        # DSF: y = gamma * x + beta
        self.beta_proj2 = nn.Linear(channel, channel+8)
        self.gamma_proj2 = nn.Linear(channel, channel+8)
        self.norm2 = nn.InstanceNorm2d(channel+8)

        self.beta_proj3 = nn.Linear(channel, channel+8)
        self.gamma_proj3 = nn.Linear(channel, channel+8)
        self.norm3 = nn.InstanceNorm2d(channel+8)

        self.beta_proj4 = nn.Linear(channel, channel+8)
        self.gamma_proj4 = nn.Linear(channel, channel+8)
        self.norm4 = nn.InstanceNorm2d(channel+8)

        self.fusion_process2 = BasicConv2d(channel+8, channel, 3, padding=1)
        self.fusion_process3 = BasicConv2d(channel+8, channel, 3, padding=1)
        self.fusion_process4 = BasicConv2d(channel+8, channel, 3, padding=1)

        # MSF: convlstm
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lstmcell43 = ConvLSTMCell(input_dim=channel, hidden_dim=channel, kernel_size=(3, 3), bias=True)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lstmcell32 = ConvLSTMCell(input_dim=channel, hidden_dim=channel, kernel_size=(3, 3), bias=True)


    def forward(self, ref_x, x):
        x2, x3, x4 = x
        bs = ref_x.shape[0]

        # DSF
        coord_feat2 = self._make_coord(bs, x2.shape[2], x2.shape[3])
        coord_feat3 = self._make_coord(bs, x3.shape[2], x3.shape[3])
        coord_feat4 = self._make_coord(bs, x4.shape[2], x4.shape[3])
        if ref_x.is_cuda:
            coord_feat2 = coord_feat2.cuda()
            coord_feat3 = coord_feat3.cuda()
            coord_feat4 = coord_feat4.cuda()

        x2 = self.norm2(torch.cat([x2, coord_feat2], 1))
        x3 = self.norm3(torch.cat([x3, coord_feat3], 1))
        x4 = self.norm4(torch.cat([x4, coord_feat4], 1))

        # y = gamma * x + beta
        beta2 = torch.tanh(self.beta_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        gamma2 = torch.tanh(self.gamma_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        beta3 = torch.tanh(self.beta_proj3(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x3)
        gamma3 = torch.tanh(self.gamma_proj3(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x3)
        beta4 = torch.tanh(self.beta_proj4(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x4)
        gamma4 = torch.tanh(self.gamma_proj4(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x4)

        x2 = self.fusion_process2(F.relu(gamma2 * x2 + beta2))
        x3 = self.fusion_process3(F.relu(gamma3 * x3 + beta3))
        x4 = self.fusion_process4(F.relu(gamma4 * x4 + beta4))   

        # MSF
        x4_h, x4_c = self.upsample4(x4), self.upsample4(x4)
        x3_h, x3_c = self.lstmcell43(input_tensor=x3, cur_state=[x4_h, x4_c])
        # print('x3: ', x3_h.shape)

        x3_h, x3_c = self.upsample3(x3_h), self.upsample3(x3_c)
        x2_h, x2_c = self.lstmcell32(input_tensor=x2, cur_state=[x3_h, x3_c])
        # print('x2: ', x2_h.shape)

        return x2_h

    def _make_coord(self, batch, height, width):
        xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
        xv_min = (xv.float()*2 - width)/width
        yv_min = (yv.float()*2 - height)/height
        xv_max = ((xv+1).float()*2 - width)/width
        yv_max = ((yv+1).float()*2 - height)/height
        xv_ctr = (xv_min+xv_max)/2
        yv_ctr = (yv_min+yv_max)/2
        hmap = torch.ones(height, width)*(1./height)
        wmap = torch.ones(height, width)*(1./width)
        coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
            xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
            xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
            hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
        coord = coord.unsqueeze(0).repeat(batch,1,1,1)
        return coord


class RFE(nn.Module):
    ''' 
    Referring Feature Enrichment (RFE) Module
    Follow implementation of https://github.com/dvlab-research/PFENet/blob/master/model/PFENet.py
    '''
    def __init__(self, d_model=64):
        super(RFE, self).__init__()

        self.d_model = d_model
        self.pyramid_bins = [44, 22, 11]        # 352 // 8, 352 // 16, 352 // 32

        self.avgpool_list = [nn.AdaptiveAvgPool2d(bin_) for bin_ in self.pyramid_bins if bin_ > 1]

        self.init_merge = []
        self.alpha_conv = []
        self.beta_conv = []
        self.inner_cls = []

        for idx in range(len(self.pyramid_bins)):
            if idx > 0:
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(self.d_model*2, self.d_model, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.d_model),
                    nn.ReLU()
                )) 
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(self.d_model + 1, self.d_model, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(self.d_model, 1, kernel_size=1)
            )) 
       
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.pyramid_cat_conv = nn.Sequential(
            nn.Conv2d(self.d_model*len(self.pyramid_bins), self.d_model, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),                          
        )              
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),   
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),                             
        )  

    def forward(self, feats, mask):
        '''
        feats: [bs, 64, 44, 44]
        sf: [bs, 1, 44, 44]
        '''

        inner_out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin_ = int(feats.shape[2] * tmp_bin)
                feats_bin = nn.AdaptiveAvgPool2d(bin)(feats)
            else:
                bin_ = tmp_bin
                feats_bin = self.avgpool_list[idx](feats)
            
            mask_bin = F.interpolate(mask, size=(bin_, bin_), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([feats_bin, mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin_, bin_), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin

            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(feats.size(2), feats.size(3)), mode='bilinear', align_corners=True)
            
            inner_out_list.append(inner_out_bin)
            pyramid_feat_list.append(merge_feat_bin)
            
        feats_refine = self.pyramid_cat_conv(torch.cat(pyramid_feat_list, 1))
        feats_refine = self.conv_block(feats_refine) + feats_refine  

        return feats_refine, inner_out_list
