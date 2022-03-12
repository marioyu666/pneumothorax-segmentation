import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import collections
from tqdm import tqdm_notebook, tqdm
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import random
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from utils.mask_functions import rle2mask
from .augmentation import do_augmentation


class SIIMDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train', augmentation=False):
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        self.augmentation = augmentation
        if self.mode=='train':
            #read and transform mask data
            self.mask_data = mask2data()
            self.data = [item for item in self.mask_data if item['img_id'] in img_id_list]
            self.path = '../data/processed/train/'
        elif self.mode=='test':
            self.path = '../data/processed/test/'
            self.data = self.img_id_list#for __len__
    
    def __getitem__(self, idx):
        if self.mode=='train':
            item = self.data[idx]
            img_path = self.path + '%s.png'%item['img_id']
            img = plt.imread(img_path)
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)

            cnt_masks = item['cnt_masks']
            masks_in_rle = item['masks']
            if cnt_masks==1:
                mask = rle2mask(masks_in_rle[0], width, height).T
            elif cnt_masks>1: #v1: just simply merge overlapping masks to get union of masks
                masks = []
                for mask_in_rle in masks_in_rle:
                    mask = rle2mask(mask_in_rle, width, height).T
                    masks.append(mask)
                mask = (np.array(masks).sum(axis=0)>=1).astype(np.int)#mask as 1 if at least one of the mask is 1
            else:
                mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE))
            mask = cv2.resize(mask.astype(np.float), (self.IMG_SIZE, self.IMG_SIZE))
            mask = np.expand_dims(mask, 0)
            ##augmentation
            if self.augmentation:
                img, mask = do_augmentation(img, mask)
            return img, mask
        elif self.mode=='test':
            img_id = self.img_id_list[idx]
            img_path = self.path + '%s.png'%img_id
            img = plt.imread(img_path)
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)
            return img
    
    def __len__(self):
        return len(self.data)

def mask2data():
    """
    return: [{}, {}, ...]
    each is 
    {'img_id': '1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027',
     'masks': ['891504 5 1018 8 1015 10 1013 12 1011 14 1009 16 1008 17', 
                '49820 3 1017 11 1012 13 1009 16 1007 18 1006 19 1005 20 1004 21', ...],
     'cnt_masks': 2}
    """
    #if preprocessed, load
    if os.path.exists('../data/processed/train_mask_in_rle.pkl'):
        with open('../data/processed/train_mask_in_rle.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    train_mask = pd.read_csv('../data/raw/train-rle.csv')
    #img_id_exist_list = [f.split('/')[-1][:-4] for f in glob.glob('data/processed/train/*')]
    grp = train_mask.groupby('ImageId')#so image id is unique, some have multiple-masks to combine
    data = []
    for img_id, subdf in grp:
        masks = []
        for j in subdf.index:
            mask_in_rle = subdf.loc[j, ' EncodedPixels'].strip()
            if mask_in_rle!='-1':
                #mask = rle2mask(mask_in_rle, 1024, 1024).T
                #masks.append(mask)
                masks.append(mask_in_rle)
        #if masks!=[]:
        #    merged_mask = (np.array(masks).sum(axis=0)>=1).astype(np.int)#mask as 1 if at least one of the mask is 1
        #else:
        #    merged_mask = []
        data.append({'img_id': img_id, 'masks': masks, 'cnt_masks': len(masks)})#'merged_mask': merged_mask
    # save
    with open('../data/processed/train_mask_in_rle.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE=512, debug=False, nonempty_only=False):
    #stratified split dataset by cnt_masks
    mask_data = mask2data()
    if nonempty_only:
        mask_data = [item for item in mask_data if item['cnt_masks']>0]
        print('Warning: Only using non-empty-mask images, count: ', len(mask_data))
    train_fname_list = [item['img_id'] for item in mask_data]
    cnt_masks = [item['cnt_masks'] if item['cnt_masks']<5 else 5 for item in mask_data]
    train_fnames, valid_fnames = train_test_split(train_fname_list, test_size=0.1, 
                                                  stratify=cnt_masks, random_state=SEED)

    #debug mode
    if debug:
        train_fnames = np.random.choice(train_fnames, 900, replace=True).tolist()
        valid_fnames = np.random.choice(valid_fnames, 200, replace=True).tolist()
    print('Count of trainset (for training): ', len(train_fnames))
    print('Count of validset (for training): ', len(valid_fnames))
    
    ## build pytorch dataset and dataloader
    train_ds = SIIMDataset(train_fnames, IMG_SIZE, mode='train', augmentation=True)
    val_ds = SIIMDataset(valid_fnames, IMG_SIZE, mode='train', augmentation=False)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    
    return train_dl, val_dl

def prepare_testset(BATCH_SIZE, NUM_WORKERS, IMG_SIZE=512):
    #sub = pd.read_csv('data/raw/sample_submission.csv')
    #test_fnames = sub.ImageId.tolist()
    test_fnames = [f.split('/')[-1][:-4] for f in glob.glob('../data/processed/test/*')]
    test_ds = SIIMDataset(test_fnames, IMG_SIZE, mode='test', augmentation=False)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    test_dl = DataLoader(
                        test_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        #sampler=sampler,
                        num_workers=NUM_WORKERS,
                        drop_last=False
                    )
    return test_dl

