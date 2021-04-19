'''

'''

''' ============================================================================ import '''
import os, numpy as np, sys, time, pandas as pd, datetime, shutil, glob, random


import albumentations
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow import keras

sys.path.append('./utils')

from helper import *
from log_helper import setup_logger
from data_loader import wj_dacon_valid_sequence, wj_dacon_train_sequence






''' ============================================================================= setting '''
# keras, tensorflow, cuda
GPU_ID = 'AUTO' # 'AUTO' 'CPU'

# Set Parameters
BATCH_SIZE = 128
IMAGE_SIZE = 256




''' ============================================================================ setting- data load '''
DB_ROOT = '../data'

''' ============================================================================= functions '''




''' ============================================================================= init '''
### gpu 지정
GPU_ID = set_gpu(GPU_ID)


    

''' ============================================================================= data load '''

db_root_tr = '../data/data/dirty_mnist_2nd'
meta = pd.read_csv('../data/data/dirty_mnist_2nd_answer.csv',index_col='index')
meta_tr = meta.sample(5000)
sq_tr = wj_dacon_train_sequence( db_root_tr, meta_tr, 
                                 IMAGE_SIZE, 0, BATCH_SIZE)
tr_loader = sq_tr
    
    
    
    
    
    
db_root_te = '../data/data/test_dirty_mnist_2nd'
meta_te = pd.read_csv('../data/data/sample_submission.csv',index_col='index')
te_loader = wj_dacon_valid_sequence( db_root_te, meta_te, IMAGE_SIZE, 0, BATCH_SIZE)
    

# te_queuer = keras.utils.OrderedEnqueuer(
#     sequence = te_loader,
#     use_multiprocessing = False,
#     shuffle = False
# )
# te_queuer.start( workers=1, max_queue_size=10 )
# te_loader = te_queuer.get()

    
''' ============================================================================= model '''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)





''' ===================================================================== optimizer, loss function, metrics '''
### Set loss function, optimizer
# loss_func_ce = keras.losses.CategoricalCrossentropy(from_logits=False) # from logits. softmax 적용에 따라서..
loss_func = keras.losses.BinaryCrossentropy()

valid_loss = keras.metrics.Mean()
valid_acc = keras.metrics.BinaryAccuracy()


list_train_valid_loss = [
    valid_loss,
    valid_acc,
]



''' ========================================================================= valid step '''
# @tf.function  
def valid_step(images, labels):
    
    ###
    predictions = model(images)
    loss = loss_func( labels, predictions )
    
    valid_loss(loss)
    valid_acc(labels, predictions)
    
    return loss, predictions

''' =========================================================================== test step '''
def test_step( images, labels_dummy):
    
    predictions = []
    
    for r in range(4):
        images_ = np.rot90(images,k=r, axes=(1,2))
        loss, pred = valid_step(images, labels_dummy)
        predictions.append( pred )
#         predictions.append( model(images_) )
    
    predictions = np.stack(predictions, axis=-1)
    predictions = np.mean( predictions, axis=-1, keepdims=False)
    
    return 0, predictions
            
            
    

''' ========================================================================= logging text function '''    
def logging_text_train(cpu_time, gpu_time):
    tr_loss = np.float32(train_loss.result())
    tr_acc = np.float32(train_acc.result())

    str__ = [
        f'TR {epoch}|{step}]   ',
        f'L {tr_loss:.05f}  ',
        f'A {tr_acc:.03f}   ',
        f'lr {sch.last_lr.numpy():.2E}   ',
        f'Ti {gpu_time:.03f}/{cpu_time:.03f}   ',
    ]
    str__ = ''.join(str__)
    
    return str__
    
def logging_text_valid():            
    vl_loss = np.float32(valid_loss.result())
    vl_acc = np.float32(valid_acc.result())

    str__ = [
        f'VL {epoch}|{step}]   ',
        f'L {vl_loss:.05f}  ',
        f'A {vl_acc:.03f}   ',
    ]
    str__ = ''.join(str__)
    
    return str__


''' ========================================================================= train '''
print('========start test=======')


paths_model = [
    './log/res50 -noise ensemble fold 2_PM:05:29:49/model_save',
    './log/res50 -noise ensemble fold 3_PM:05:30:52/model_save',
    './log/res50 -noise ensemble fold 0_PM:05:29:04/model_save',
    './log/res50 -noise ensemble fold 1_PM:05:29:27/model_save',
]

paths_model = [
    '../log_old/res50 ensemble fold 1_AM:10:47:51/model_save',
    '../log_old/res50 ensemble fold 3_AM:10:49:37/model_save',
    '../log_old/res50 ensemble fold 2_AM:10:48:40/model_save',
    '../log_old/res50 ensemble fold 0_AM:10:46:55/model_save',
]

paths_model = [
    '../log_old/hingeloss res50 bsize128 fold0 PM:03:48:44/model_save',
    '../log_old/hingeloss res50 bsize128 fold2 PM:03:49:11/model_save',
    '../log_old/hingeloss res50 bsize128 fold3 PM:03:49:23/model_save',
    '../log_old/hingeloss res50 bsize128 fold1 PM:03:48:54/model_save',
]

paths_model = [
    '../log/res50 th254 hinge bsize128 foldTrue0 AM:01:21:13/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 AM:06:39:57/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 AM:07:42:01/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 PM:01:27:58/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 PM:09:55:40/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 PM:10:56:11/model_save',
    '../log/res50 th254 hinge bsize128 foldTrue0 PM:12:19:37/model_save',
]


preds_sum = []
for i_model, path_model in enumerate(paths_model):
    model = keras.models.load_model(path_model)

    # === test loop
    preds = []
    for i_batch, (paths, images, labels_dummy, _) in enumerate(te_loader):
        
        _, preds_ = test_step( images, labels_dummy )
        preds.append(preds_)
        
        vl_acc = np.float32(valid_acc.result())
        print(f'{i_model}/{i_batch*BATCH_SIZE}/{vl_acc}')
        
    preds = np.concatenate(preds)
    
    preds_sum.append( preds )
    
    valid_acc.reset_states()

preds_sum = np.stack(preds_sum,-1)

nums = list(range(50000,55000))
meta_sub = meta_te.copy()
for num, pred in zip( nums, preds_sum ):
    pred = np.mean( pred, axis=-1 )
    meta_sub.loc[num] = np.int32(pred>0.5)
meta_sub.to_csv(os.path.join('ensenble_aaa.csv'))
    
