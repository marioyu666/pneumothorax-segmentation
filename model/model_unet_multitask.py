import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

from loss.losses import dice_loss, FocalLoss, weighted_bce, soft_dice_loss
from utils.metrics import dice


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        #x = self.dropout(x)
        x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.spa_cha_gate(x)
        return x

class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)#16
        self.channel_gate = ChannelGate2d(in_ch)
    
    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2 #x = g1*x + g2*x
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.
    def load_pretrain(self, pretrain_file):
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, pretrained=True, debug=False):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.debug = debug

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool,
        )# 64
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSE(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSE(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSE(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSE(512))

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(512+256, 512, 64)
        self.decoder4 = Decoder(256+64, 256, 64)
        self.decoder3 = Decoder(128+64, 128,  64)
        self.decoder2 = Decoder( 64+ 64, 64, 64)
        self.decoder1 = Decoder(64    , 32,  64)

        self.logit_mask = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
            #nn.Sigmoid()#for dice loss
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit_clf = nn.Sequential(
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),
            nn.Linear(256, 1), #block.expansion=1, num_classes=28
            #nn.Sigmoid()
        )

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)        
        if self.debug:
            print('input: ', x.size())

        x = self.conv1(x)
        if self.debug:
            print('e1',x.size())
        e2 = self.encoder2(x)
        if self.debug:
            print('e2',e2.size())
        e3 = self.encoder3(e2)
        if self.debug:
            print('e3',e3.size())
        e4 = self.encoder4(e3)
        if self.debug:
            print('e4',e4.size())
        e5 = self.encoder5(e4)
        if self.debug:
            print('e5',e5.size())

        c = self.center(e5)
        if self.debug:
            print('center',c.size())

        d5 = self.decoder5(c,e5)
        if self.debug:
            print('d5',d5.size())
        d4 = self.decoder4(d5, e4)
        if self.debug:
            print('d4',d4.size())
        d3 = self.decoder3(d4,e3)
        if self.debug:
            print('d3',d3.size())
        d2 = self.decoder2(d3,e2)
        if self.debug:
            print('d2',d2.size())
        d1 = self.decoder1(d2)
        if self.debug:
            print('d1',d1.size())

        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)
        if self.debug:
            print('hypercolum', f.size())
        
        f = F.dropout(f, p=0.40)#training=self.training
        
        logit_mask = self.logit_mask(f)
        if self.debug:
            print('logit_mask', logit_mask.size())
        
        f = self.avgpool(c)
        if self.debug:
            print('avgpool',f.size())
        f = f.view(f.size(0), -1)
        if self.debug:
            print('reshape',f.size())
        logit_clf = self.logit_clf(f)
        if self.debug:
            print('logit_clf', logit_clf.size())
        return logit_mask, logit_clf
    
    ##-----------------------------------------------------------------

    def criterion(self, logit_mask, truth_mask, logit_clf, truth_clf):
        """Define the (customized) loss function here."""        
        Loss_FUNC = nn.BCEWithLogitsLoss(reduction='mean')

        loss_mask = Loss_FUNC(logit_mask, truth_mask)

        loss_clf = Loss_FUNC(logit_clf, truth_clf)
        
        loss = 0.95*loss_mask + 0.05*loss_clf

        return loss, loss_mask, loss_clf

    def metric(self, logit_mask, truth_mask, logit_clf, truth_clf):
        """Define metrics for evaluation especially for early stoppping."""
        #return iou_pytorch(logit, truth)
        #return dice_multitask(logit_mask, truth_mask, logit_clf, truth_clf, iou=False, eps=1e-8)
        return dice(logit, truth)

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


def predict_proba(net, test_dl, device, multi_gpu=False, mode='test', tta=True):
    pred_mask, pred_clf = None, None
    if multi_gpu:
        net.module.set_mode('test')
    else:
        net.set_mode('test')
    with torch.no_grad():
        if mode=='valid':
            for i, (image, masks) in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit_mask, logit_clf = net(input_data)
                logit_mask, logit_clf = logit_mask.cpu().numpy(), logit_clf.cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_mask_flip, logit_clf_flip = net(input_data_flip)
                    logit_mask_flip, logit_clf_flip = logit_mask_flip.cpu().numpy(), logit_clf_flip.cpu().numpy()
                    logit_mask = (logit_mask + logit_mask_flip[:,:,:,::-1]) / 2
                    logit_clf = (logit_clf + logit_clf_flip) / 2
                if pred_mask is None:
                    pred_mask = logit_mask
                    pred_clf = logit_clf
                else:
                    pred_mask = np.concatenate([pred_mask, logit_mask], axis=0)
                    pred_clf = np.concatenate([pred_clf, logit_clf], axis=0)
        elif mode=='test':
            for i, image in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit_mask, logit_clf = net(input_data)
                logit_mask, logit_clf = logit_mask.cpu().numpy(), logit_clf.cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_mask_flip, logit_clf_flip = net(input_data_flip)
                    logit_mask_flip, logit_clf_flip = logit_mask_flip.cpu().numpy(), logit_clf_flip.cpu().numpy()
                    logit_mask = (logit_mask + logit_mask_flip[:,:,:,::-1]) / 2
                    logit_clf = (logit_clf + logit_clf_flip) / 2
                if pred_mask is None:
                    pred_mask = logit_mask
                    pred_clf = logit_clf
                else:
                    pred_mask = np.concatenate([pred_mask, logit_mask], axis=0)
                    pred_clf = np.concatenate([pred_clf, logit_clf], axis=0)
    IMG_SIZE = pred_mask.shape[-1]
    return pred_mask.reshape(-1, IMG_SIZE, IMG_SIZE), pred_clf.reshape(-1, 1)



