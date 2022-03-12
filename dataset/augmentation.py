import numpy as np
import pandas as pd
import cv2
import random
import math
import time, os


def do_resize(image, H, W):
    resized_image = cv2.resize(image,dsize=(W,H))
    return resized_image

def do_augmentation(image, mask=None):
    """image: shape=(3, 256, 256)
    """
    #seed = get_seed()
    #np.random.seed(seed)
    #image = image.reshape(3, 256, 256)
    #print('do_augmentation: ', image.shape, mask.shape)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_3channels(image, do_horizontal_flip)
            if mask is not None:
                mask = do_3channels(mask, do_horizontal_flip)
        elif c==1:
            image = do_3channels(image, do_vertical_flip)
            if mask is not None:
                mask = do_3channels(mask, do_vertical_flip)
        elif c==2:
            k = np.random.randint(1, 4)
            image = do_3channels(image, do_rotation, k=k)
            if mask is not None:
                mask = do_3channels(mask, do_rotation, k=k)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_3channels(image, do_brightness_shift, alpha=np.random.uniform(-0.05, +0.05))
        elif c==1:
            image = do_3channels(image, do_brightness_multiply, alpha=np.random.uniform(1-0.05, 1+0.05))
        elif c==2:
            image = do_3channels(image, do_gamma, gamma=np.random.uniform(1-0.05, 1+0.05))
    
    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c==0:
            image = do_3channels(image, do_guassian_blur, kernal_size=(3, 3))
        elif c==1:
            image = do_3channels(image, do_perspective_transform)
            if mask is not None:
                mask = do_3channels(mask, do_perspective_transform)
    if mask is not None:
        return image, mask#.reshape(1, 256, 256)
    return image#.reshape(3, 256, 256)


def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed

def do_3channels(image, aug_method, **kwargs):
    output = []
    for _image in image:
        aug_img = aug_method(_image, **kwargs)
        output.append(aug_img)
    return np.array(output)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Define some augmentation methods below
#----------------------------------------------------------------------
#----------------------------------------------------------------------
def do_horizontal_flip(image):
    #flip left-right
    image = cv2.flip(image, 1)
    return image

def do_vertical_flip(image):
    #flip top-down
    image = cv2.flip(image, 0)
    return image

def do_rotation(image, k=1):
    # k: how many times rotate 90 degrees
    image = np.rot90(image, k)
    return image

#----
def do_invert_intensity(image):
    #flip left-right
    image = np.clip(1-image,0,1)
    return image

def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image

def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image

#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image

#----
def do_guassian_blur(image, kernal_size=(3, 3)):
    image = cv2.GaussianBlur(image, kernal_size, 0)
    return image

def do_perspective_transform(image):
    pts1 = np.float32([[0, 0],[100, 0],[0, 100],[100, 510]])
    pts2 = np.float32([[0, 0],[100, 0],[0, 100],[100, 520]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image, M, image.shape)
    return image


