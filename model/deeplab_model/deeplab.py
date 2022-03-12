import sys
sys.path.insert(0,'../..')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone

import loss.lovasz_losses as L
from loss.losses import dice_loss, FocalLoss, weighted_bce
from metrics import iou_pytorch, dice


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, debug=False):
        super(DeepLab, self).__init__()
        self.debug = debug

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        input = torch.cat([
            (input-mean[0])/std[0],
            (input-mean[1])/std[1],
            (input-mean[2])/std[2],
        ],1)       
        if self.debug:
            print('input:', input.size())
        x, low_level_feat = self.backbone(input)
        if self.debug:
            print('backbone--x:', x.size())
            print('backbone--low_level_feat:', low_level_feat.size())
        x = self.aspp(x)
        if self.debug:
            print('aspp:', x.size())
        x = self.decoder(x, low_level_feat)
        if self.debug:
            print('decoder:', x.size())
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.debug:
            print('interpolate:', x.size())

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""        
        Loss_FUNC = nn.BCEWithLogitsLoss()
        #Loss_FUNC = FocalLoss(alpha=1, gamma=2, logits=True, reduce=True)
        #bce_loss = Loss_FUNC(logit, truth)
        loss = Loss_FUNC(logit, truth)
        
        #loss = L.lovasz_hinge(logit, truth, ignore=None)#255
        #loss = L.symmetric_lovasz(logit, truth)
        #loss = 0.66 * dice_loss(logit, truth) + 0.33 * bce_loss
        #loss = dice_loss(logit, truth)
        #loss = weighted_bce(logit, truth)
        return loss

    def metric(self, logit, truth):
        """Define metrics for evaluation especially for early stoppping."""
        #return iou_pytorch(logit, truth)
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
    if tta:
        print("use TTA")
    else:
        print("not use TTA")
    y_pred = None
    if multi_gpu:
        net.module.set_mode('test')
    else:
        net.set_mode('test')
    with torch.no_grad():
        if mode=='valid':
            for i, (image, masks) in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
        elif mode=='test':
            for i, image in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
    IMG_SIZE = y_pred.shape[-1]
    return y_pred.reshape(-1, IMG_SIZE, IMG_SIZE)#256


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


