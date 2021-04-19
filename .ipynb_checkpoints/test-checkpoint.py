''' dsfsdf '''

''' ============================================================================ import '''
import sys
# sys.path.append('/anaconda3/envs/wj/lib/python3.7/site-packages')

import os, numpy as np, sys, time, pandas as pd, datetime, shutil, glob, random
import argparse


import albumentations
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow import keras

sys.path.append('./utils')

from helper import *
from log_helper import setup_logger
from data_loader import valid_sequence, train_sequence





''' ============================================================================= setting '''
# keras, tensorflow, cuda
GPU_ID = 'AUTO' # 'AUTO' 'CPU'
MIXED_PRECISION = False
MIXED_PRECISION_DTYPE = 'mixed_float16' # 'float16' 'float32' 'float64' 'mixed_float16'

# Set Parameters
BATCH_SIZE = 32
EPOCHS = 30000
STEP_PER_EPOCH = 1024//BATCH_SIZE

IMAGE_SIZE = 128
PAD_SIZE = 0
# FOLD_VALID = 0
# FOLD_RANDOM = True

# MIX = False # True는 꼭 ce랑 같이 쓰기
# LOSS_TYPE = 'hinge'
# TH254 = True

LR_TARGET = 3e-4
LR_WARMUP = 5
LR_RESTART = 20
LR_MINIMUM = 1e-6
# LOOK_AHEAD = False




''' ============================================================================= init '''
### gpu 지정
GPU_ID = set_gpu(GPU_ID)
    

''' ============================================================================= data load '''

'''-------------- valid '''
meta = pd.read_csv('meta.csv',index_col=0)
cond = meta['data_type'].isin(['transistor']) & \
       meta['ttg_type'].isin(['test']) 
meta_vl = meta.loc[cond]

sq_vl = train_sequence( meta_vl, IMAGE_SIZE, PAD_SIZE, BATCH_SIZE)



''' ============================================================================= test random '''

model_all = keras.models.load_model('model_all')

for _ in range(10):
    sq_vl._shuffle()
    path, images, cword = sq_vl.__getitem__(0)
    gen_image, _,_, st_image = model_all(images)

    I = np.float32(images[0])
    P = np.float32(gen_image[0])
    S = np.float32(st_image[0])
    D = np.clip(np.abs((I-P)*2+0.0),0,1)
    D2 = np.clip(np.abs((S-P)*2+0.0),0,1)
    sss = np.hstack( (I,P,S,D,D2) )
#     imshow(cv2.resize(sss,(256*5,256))*255)
#     print( cword[0])
    
    P3 = cv2.ximgproc.guidedFilter(guide=S,src=P,radius=3,eps=0.00000001)
    P3 = np.expand_dims(P3,-1)
    D3 = np.clip(np.abs((S-P3)*2+0.0),0,1)
    sss = np.hstack( (I,S,P,P3,D,D2,D3) )
    print( cword[0])
    imshow(cv2.resize(sss,(256*7,256))*255)