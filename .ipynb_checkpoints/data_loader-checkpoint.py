'''
개인적으로 사용할 데이터 로더 표준 
'''

#%%
import tensorflow as tf
import cv2, numpy as np, os
from tensorflow import keras
import pandas as pd
from utils.helper import *
import albumentations as aug

import numpy as np
import skimage.measure

'''
path -- index(meta) -- index(zero_to_end)
'''
class valid_sequence(keras.utils.Sequence):
    
    def __init__(self, meta, image_size, pad_size=0, batch_size=32):
        self.cache = {}
        self.meta = meta
        self.meta_index = np.int32(self.meta.index)
        self.image_size = image_size
        self.pad_size = pad_size
        self.batch_size = batch_size

        lword_to_label = []
        
    def _shuffle(self):
        np.random.shuffle(self.meta_index)
        return 
    
    def _index_to_meta_index(self,index):
        return self.meta_index[index]

    def _get_path_from_meta_index(self,mindex):
        return self.meta['path'].loc[mindex]
    
    def _get_lward_from_meta_index(self,mindex):
        return self.meta['error_type'].loc[mindex]
        
    def _augmentation(self,image):
        return image
    
    def _get_image(self,path):

        if path in self.cache.keys():
            image = self.cache[path]
        else:
            image = cv2.imread( path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(self.image_size,self.image_size))
            image = np.pad(image,[[self.pad_size,self.pad_size],
                                [self.pad_size,self.pad_size]])
            # image = cv2.bilateralFilter( image, 31, 10, 31/6)
            # image = cv2.bilateralFilter( image, 15, 10, 15/6)
            # image = cv2.bilateralFilter( image, 7, 10, 7/6)
            self.cache[path] = image
        
        # ==== Augmentation ============
        image = self._augmentation(image)
        # ==== Augmentation ============
            
        image = np.expand_dims(image,axis=-1)
        image = np.float32(image)/255
        
        return image
    
    def _get_one(self,index):
        
        mindex = self._index_to_meta_index(index)
        path = self._get_path_from_meta_index(mindex)
        lword = self._get_lward_from_meta_index(mindex)

        image = self._get_image(path)
        
        return path, image, lword
          
        
    def __len__(self):
        return int(np.ceil(len(self.meta_index)/self.batch_size))
    
    def on_epoch_end(self):
#         np.random.shuffle(self.meta_index)
        return 
    
    def __getitem__(self, idx):
        
        idx_beg = idx*self.batch_size
        idx_end = idx*self.batch_size+self.batch_size
        idx_end = min( idx_end, len(self.meta_index) )
        idxs = list(range(idx_beg,idx_end))
        
        paths, images, lwords = [], [], []
        for i in idxs:
            path, image, lword = self._get_one(i)
            paths.append(path)
            images.append(image)
            lwords.append(lword)
            
        ### augmentation
        
        ### numpy 배열로 변경
        images = np.stack( images, axis=0 )
        
        return paths, images, lwords
        
        
#         breakpoint()

class train_sequence(valid_sequence):
    
    
    def on_epoch_end(self):
        self._shuffle()
        return 
        
        
    def __len__(self):
        return int(np.floor(len(self.meta_index)/self.batch_size))
    
    def _augmentation(self,image):
        
        
        
        ### 필수 적용
        image = aug.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=[-1,1],
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )(image=image)['image']


        return image
        

    
        


    
if __name__ == '__main__':
    
    meta = pd.read_csv('./meta.csv',index_col=0)
    cond = meta['data_type'].isin(['screw']) & \
           meta['ttg_type'].isin(['test'])
    meta = meta.loc[cond]

    sq_vl = valid_sequence(meta, 256)
    print(sq_vl._get_one(0))

    for paths, images, lwords in sq_vl:
        imshow( images[0]*255) 
        print(lwords[0])
    
#     sq_tr.on_epoch_end()

# class wj_dacon_train_sequence(wj_dacon_valid_sequence):
    
    
#     def on_epoch_end(self):
#         np.random.shuffle(self.meta_index)
#         return 
        
        
#     def __len__(self):
#         return int(np.floor(len(self.meta)/self.batch_size))
    
#     def _augmentation(self,image):
            
#         def RandomNoise(image):   
# #             if np.random.rand()<0.5 :
# #                 return {'image':image}
            
#             noise = np.random.rand( image.shape[0], image.shape[1] )
#             image[noise< np.random.randint(0,32)/255] = 0
#             image[ np.random.randint(250,254)/255<noise] = 255
#             return {'image':image}
            
#         augs = [ 
#             aug.GridDistortion(
#                 num_steps=5,
#                 distort_limit=0.25,
#                 border_mode=0,
#             ),
#             aug.ElasticTransform(
#                 alpha=1,
#                 sigma=10,
#                 alpha_affine=10,
#                 border_mode=0,
#             ),
#             aug.CoarseDropout(
#                 max_holes=16,   min_holes=4,
#                 max_height=16,  min_height=4,
#                 max_width=16,   min_width=4,
#             ),
# #             aug.RandomBrightnessContrast(
# #             ),
# #             aug.RandomGamma(
# #             ),
#         ]
        
        
#         ### 필수 적용
#         image = aug.ShiftScaleRotate(
#             shift_limit=0.1,
#             scale_limit=0.3,
#             rotate_limit=[-180,180],
#             border_mode=0,
#             always_apply=True,
#         )(image=image)['image']
# #         image = RandomNoise(
# #             image=image,
# #         )['image']
        
        
#         ### 랜덤적용
#         augs_pick3 = np.random.choice(augs,min(len(augs),3))
#         for func in augs_pick3:
#             image = func(image = image)['image']
                
#         return image
        




    
# if __name__ == '__main__':
    
    
#     meta = pd.read_csv('../data/data/dirty_mnist_2nd_answer.csv',index_col='index')
#     meta_tr = meta.iloc[:40000]
#     meta_vl = meta.iloc[40000:]
#     sq_tr = wj_dacon_train_sequence('../data/data/dirty_mnist_2nd',meta_tr,256,32,128)
#     sq_vl = wj_dacon_valid_sequence('../data/data/dirty_mnist_2nd',meta_vl,256,0,128)
    
# #     sq_tr.on_epoch_end()
#     for step, (paths, images, labels, lwords) in enumerate(sq_tr):
#         print(step,paths[0])
#         imshow(images[0]*255)
#         if step == 10:
#             break;
#         pass
# #     from tensorflow.keras.utils import OrderedEnqueuer
# #     oq = OrderedEnqueuer(sq_tr)
# #     oq.start()
# #     gen_tr = oq.get()
# #     for i in range(10000000):
# #         _,images,_,_ = next(gen_tr)
# #         imshow(images[0]*255)
    
# #     for step, (paths, images, labels) in enumerate(sq):
# #         print(f'step {step} label {sq.label_2_string(labels[0])}')
# #         imshow(images[0])
# #         break;

# #     meta_te = pd.read_csv('../data/data/sample_submission.csv',index_col='index')
# #     sq_te = wj_dacon_valid_sequence(meta_te,1000)
    
# #     for i in range(10):
# #         image = sq_te._get(i)[1]
# #         imshow(image*255)
