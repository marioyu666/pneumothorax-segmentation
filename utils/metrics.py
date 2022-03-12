import torch
import numpy as np


SMOOTH = 1e-6

def dice(logit:torch.Tensor, truth:torch.Tensor, iou:bool=False, eps:float=1e-8):#->Rank0Tensor
    """
    A slight modification of the default dice metric to make it comparable with the competition metric: 
    dice is computed for each image independently, and dice of empty image with zero prediction is 1. 
    Also I use noise removal and similar threshold as in my prediction pipline.
    """
    IMG_SIZE = truth.shape[-1]#256
    EMPTY_THRESHOLD = 100.0*(IMG_SIZE/128.0)**2 #count of predicted mask pixles<threshold, predict as empty mask image
    MASK_THRESHOLD = 0.22 #softmax>threshold, predict a mask=1
    
    n = truth.shape[0]
    if len(logit.size())==4:
        logit = logit.squeeze(1)
    if len(truth.size())==4:
        truth = truth.squeeze(1)
    logit = torch.sigmoid(logit).view(n, -1)
    pred = (logit>MASK_THRESHOLD).long()
    truth = truth.view(n, -1).long()
    pred[pred.sum(dim=1) < EMPTY_THRESHOLD, ] = 0
    
    intersect = (pred * truth).sum(dim=1).float()
    union = (pred + truth).sum(dim=1).float()
    if not iou:
        return ((2.0*intersect + eps) / (union+eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()

