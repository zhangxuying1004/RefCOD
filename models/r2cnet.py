import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.utils import BasicConv2d
from models.featfusion import FeatFusion
from models.featenrich import RFE


class Network(nn.Module):
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = torchvision.models.resnet50(pretrained=imagenet_pretrained)
        
        self.x2_down_channel = BasicConv2d(512, channel, 1)
        self.x3_down_channel = BasicConv2d(1024, channel, 1)
        self.x4_down_channel = BasicConv2d(2048, channel, 1)

        self.ref_proj = BasicConv2d(2048, channel, 1)

        # dsf + msf
        self.feat_fusion = FeatFusion(channel=channel)

        # target matching
        self.relevance_norm = nn.BatchNorm2d(1)
        self.relevance_acti = nn.LeakyReLU(0.1, inplace=True)

        # rfe
        self.rfe = RFE(d_model=channel)

        self.cls = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x, ref_x):
        bs, _, H, W = x.shape

        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88

        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Down Channel
        x2 = self.x2_down_channel(x2)    # bs, 64, 44, 44
        x3 = self.x3_down_channel(x3)    # bs, 64, 22, 22   
        x4 = self.x4_down_channel(x4)    # bs, 64, 11, 11
        ref_x = self.ref_proj(ref_x)

        # Dual-source Information Fusion + Multi-scale Feature Fusion
        x2_h = self.feat_fusion(ref_x=ref_x, x=[x2, x3, x4])

        # Target Matching
        mask = torch.cat([F.conv2d(x2_h[i].unsqueeze(0), ref_x[i].unsqueeze(0)) for i in range(bs)], 0)
        mask = self.relevance_acti(self.relevance_norm(mask))

        # Referring Feature Enrichment
        x2_h, inner_out_list = self.rfe(x2_h, mask)

        # Conv Head
        S_g = self.cls(x2_h)
        S_g_pred = F.interpolate(S_g, size=(H, W), mode='bilinear', align_corners=True)      # (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        S_inner_preds = [F.interpolate(inner_out, size=(H, W), mode='bilinear', align_corners=True) for inner_out in inner_out_list]

        return S_g_pred, S_inner_preds

    
