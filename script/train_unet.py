import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import pickle
import os
import random
import logging
import time
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import save_checkpoint, load_checkpoint, set_logger
from utils.gpu_utils import set_n_get_device

from dataset.dataset_unet import prepare_trainset
from model.model_unet import UNetResNet34, predict_proba

import argparse


######### Define the training process #########
def run_training(train_dl, val_dl, multi_gpu=[0, 1]):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    #---
    if MODEL == 'UNetResNet34':
        net = UNetResNet34(debug=False).cuda(device=device)

#     for param in net.named_parameters():
#         if param[0][:8] in ['decoder5']:#'decoder5', 'decoder4', 'decoder3', 'decoder2'
#             param[1].requires_grad = False

    train_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=0.0001, lr=LearningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=4,
                                                       verbose=False, threshold=0.0001, 
                                                       threshold_mode='rel', cooldown=0, 
                                                       min_lr=0, eps=1e-08)
    
    if warm_start:
        logging.info('warm_start: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)
    
    # using multi GPU
    if multi_gpu is not None:
        net = nn.DataParallel(net, device_ids=multi_gpu)

    diff = 0
    best_val_metric = -0.1
    optimizer.zero_grad()
    
    for i_epoch in range(NUM_EPOCHS):
        ## adjust learning rate
        #scheduler.step(epoch=i_epoch)
        #print('lr: %f'%scheduler.get_lr()[0])
        
        t0 = time.time()
        # iterate through trainset
        if multi_gpu is not None:
            net.module.set_mode('train')
        else:
            net.set_mode('train')
        train_loss_list, train_metric_list = [], []
        for i, (image, masks) in enumerate(train_dl):            
            input_data = image.to(device=device, dtype=torch.float)
            truth = masks.to(device=device, dtype=torch.float)
            #set_trace()
            logit = net(input_data)
            
            if multi_gpu is not None:
                _train_loss  = net.module.criterion(logit, truth)
                _train_metric  = net.module.metric(logit, truth)
            else:
                _train_loss  = net.criterion(logit, truth)
                _train_metric  = net.metric(logit, truth)
            train_loss_list.append(_train_loss.item())
            train_metric_list.append(_train_metric.item())

            #grandient accumulation step=2
            acc_step = GradientAccStep
            _train_loss = _train_loss / acc_step
            _train_loss.backward()
            if (i+1)%acc_step==0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = np.mean(train_loss_list)
        train_metric = np.mean(train_metric_list)

        # compute valid loss & metrics (concatenate valid set in cpu, then compute loss, metrics on full valid set)
        net.module.set_mode('valid')
        with torch.no_grad():
            val_loss_list, val_metric_list = [], []
            for i, (image, masks) in enumerate(val_dl):
                input_data = image.to(device=device, dtype=torch.float)
                truth = masks.to(device=device, dtype=torch.float)
                logit = net(input_data)
                
                if multi_gpu is not None:
                    _val_loss  = net.module.criterion(logit, truth)
                    _val_metric  = net.module.metric(logit, truth)
                else:
                    _val_loss  = net.criterion(logit, truth)
                    _val_metric  = net.metric(logit, truth)
                val_loss_list.append(_val_loss.item())
                val_metric_list.append(_val_metric.item())

            val_loss = np.mean(val_loss_list)
            val_metric = np.mean(val_metric_list)

        # Adjust learning_rate
        scheduler.step(val_metric)
        
        #force to at least train N epochs
        if i_epoch>=-1:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                is_best = True
                diff = 0
            else:
                is_best = False
                diff += 1
                if diff > early_stopping_round:
                    logging.info('Early Stopping: val_metric does not increase %d rounds'%early_stopping_round)
                    break
        else:
            is_best = False

        #save checkpoint
        checkpoint_dict = \
        {
            'epoch': i,
            'state_dict': net.module.state_dict() if multi_gpu is not None else net.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'metrics': {'train_loss': train_loss, 'val_loss': val_loss, 
                        'train_metric': train_metric, 'val_metric': val_metric}
        }
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=checkpoint_path)

        #if i_epoch%20==0:
        if i_epoch>-1:
            logging.info('[EPOCH %05d]train_loss, train_metric: %0.5f, %0.5f; val_loss, val_metric: %0.5f, %0.5f; time elapsed: %0.1f min'%(i_epoch, train_loss.item(), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))


#
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)


if __name__ == "__main__":
    ######### 1. Define SEED #########
    parser = argparse.ArgumentParser(description='====Model Parameters====')
    parser.add_argument('--SEED', type=int, default=1234)
    params = parser.parse_args()
    SEED = params.SEED
    print('SEED=%d'%SEED)

    ######### 2. Config the Training Arguments #########
    MODEL = 'UNetResNet34'
    print('====MODEL ACHITECTURE: %s===='%MODEL)

    device = set_n_get_device("0", data_device_id="cuda:0")#use the first GPU
    multi_gpu = None #[0,1] use 2 gpus; None single gpu

    debug = True # if True, load 100 samples, False
    IMG_SIZE = 256 #1024#768#512#256
    BATCH_SIZE = 2
    GradientAccStep = 1
    NUM_WORKERS = 4
    
    warm_start, last_checkpoint_path = False, '../checkpoint/%s_%s_v1_seed%s/best.pth.tar'%(MODEL, IMG_SIZE, SEED)
    checkpoint_path = '../checkpoint/%s_%s_v1_seed%s'%(MODEL, IMG_SIZE, SEED)
    LOG_PATH = '../logging/%s_%s_v1_seed%s.log'%(MODEL, IMG_SIZE, SEED)#

    NUM_EPOCHS = 30
    early_stopping_round = 5
    LearningRate = 0.2
    #MIN_LR = 0.002


    seed_everything(SEED)
    ######### 3. Load data #########
    train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE, debug, nonempty_only=False)#True: Only using nonempty-mask!

    ######### 4. Run the training process #########
    run_training(train_dl, val_dl, multi_gpu=multi_gpu)

    print('------------------------\nComplete SEED=%d\n------------------------'%SEED)
