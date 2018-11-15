#coding=utf-8
import keras
import math
import os
from skimage.io import imread
import numpy as np
from .data import rle_decode
import threading
from .image import ImageTransformer
from skimage.transform import resize
from keras import backend as K
from .data import fit_ndimage_param, decompose_ndimage, compose_ndcube
from skimage import measure
import random

class BaseGenerator(keras.utils.Sequence):
    
    def __init__(self, train_df, isship_list, nanship_list, 
                 train_img_dir='../images', batch_size=1, 
                 shuffle=True, pos_ratio=0.5, aug_parms=None,
                 target_shape=(768, 768),
                 preprocessing_function=None):
        self.batch_size = batch_size
        self.train_df = train_df
        self.aug_parms = aug_parms
        self.shuffle = shuffle
        self.isship_list = isship_list
        self.nanship_list = nanship_list
        self.pos_ratio = pos_ratio
        self.pos_indexes = np.arange(len(self.isship_list))
        self.neg_indexes = np.arange(len(self.nanship_list))
        self.train_img_dir = train_img_dir
        self.target_shape = target_shape
        self.transformer = ImageTransformer(**aug_parms)
        self.preprocessing_function = preprocessing_function

    def __len__(self):
        #计算每一个epoch的迭代次数
        batch_pos_size = int(self.pos_ratio * self.batch_size)

        return math.ceil(len(self.isship_list) / float(batch_pos_size))


    def __getitem__(self, index):
        # 生成batch_size个索引
        batch_pos_size = int(self.pos_ratio * self.batch_size)
        batch_neg_size = self.batch_size - batch_pos_size
        batch_pos_indexs = self.pos_indexes[index*batch_pos_size:(index+1)*batch_pos_size]
        batch_neg_indexs = self.neg_indexes[index*batch_neg_size:(index+1)*batch_neg_size]
        # 根据索引获取datas集合中的数据
        batch_names = [self.isship_list[k] for k in batch_pos_indexs]
        batch_names += [self.nanship_list[k] for k in batch_neg_indexs]
        batch_x, batch_y = self.data_generation(batch_names)

        return batch_x, batch_y


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.pos_indexes)
            np.random.shuffle(self.neg_indexes)


    def data_generation(self, batch_names):
        batch_image = []
        batch_label = []
        image_shape = list(self.target_shape) + [3]
        for i, name in enumerate(batch_names):
            image = imread(os.path.join(self.train_img_dir, name))
            image = image / 255
            mask_list = self.train_df['EncodedPixels'][self.train_df['ImageId'] == name].tolist()
            mask = np.zeros((768, 768, 1))
            for item in mask_list:
                instance_mask = rle_decode(item, (768, 768))
                mask[:, :, 0] += instance_mask
            
            if not self.target_shape == (768, 768):
                mask = np.squeeze(mask)
                mask = resize(mask, self.target_shape, order=0, preserve_range=True)
                image = resize(image, self.target_shape)
                mask = np.expand_dims(mask, axis=-1)

            params = self.transformer.get_random_transform(image_shape)
            image = self.transformer.apply_transform(image.astype(K.floatx()), params, order=1)
            mask = self.transformer.apply_transform(mask.astype(K.floatx()), params, order=0)

            if self.preprocessing_function:
                image = self.preprocessing_function(image, data_format=K.image_data_format())

            batch_image.append(image)
            batch_label.append(mask)
        
        return np.array(batch_image), np.array(batch_label)



class PredictGenerator(keras.utils.Sequence):
    
    def __init__(self, train_df, val_list, train_img_dir='../images', 
                 batch_size=1, target_shape=(768, 768), preprocessing_function=None):
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_list = val_list
        self.indexes = np.arange(len(self.val_list))
        self.train_img_dir = train_img_dir
        self.target_shape = target_shape
        self.preprocessing_function = preprocessing_function
        self.shuffle = True

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.val_list) / float(self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # print('processing batch--', index)
        # 生成batch_size个索引
        batch_size = self.batch_size
        batch_indexs = self.indexes[index*batch_size:(index+1)*batch_size]
        # 根据索引获取datas集合中的数据
        batch_names = [self.val_list[k] for k in batch_indexs]
        batch_x, batch_y = self.data_generation(batch_names)

        return batch_x, batch_y


    def data_generation(self, batch_names):
        batch_image = []
        batch_label = []
        for i, name in enumerate(batch_names):
            image = imread(os.path.join(self.train_img_dir, name))
            image = image / 255
            mask_list = self.train_df['EncodedPixels'][self.train_df['ImageId'] == name].tolist()
            mask = np.zeros((768, 768, 1))
            for item in mask_list:
                instance_mask = rle_decode(item, (768, 768))
                mask[:, :, 0] += instance_mask
            if not self.target_shape == (768, 768):
                mask = np.squeeze(mask)
                mask = resize(mask, self.target_shape, order=0, preserve_range=True)
                image = resize(image, self.target_shape)
                mask = np.expand_dims(mask, axis=-1)
            if self.preprocessing_function:
                image = self.preprocessing_function(image, data_format=K.image_data_format())
            batch_image.append(image)
            batch_label.append(mask)
            
        return np.array(batch_image), np.array(batch_label)


class BasePatchGenerator(keras.utils.Sequence):
    
    def __init__(self, train_df, isship_list, nanship_list, 
                 train_img_dir='../images', batch_size=1, 
                 shuffle=True, pos_ratio=0.5, aug_parms=None,
                 patch_shape=(256, 256),
                 target_shape=(768, 768),
                 preprocessing_function=None,
                 mean_instances_per_image=3,
                 sample_ratio=0.9):
                 
        self.batch_size = batch_size
        self.train_df = train_df
        self.aug_parms = aug_parms
        self.shuffle = shuffle
        self.isship_list = isship_list
        self.nanship_list = nanship_list
        self.pos_ratio = pos_ratio
        self.pos_indexes = np.arange(len(self.isship_list))
        self.neg_indexes = np.arange(len(self.nanship_list))
        self.train_img_dir = train_img_dir
        self.patch_shape = patch_shape
        self.target_shape = target_shape
        self.transformer = ImageTransformer(**aug_parms)
        self.preprocessing_function = preprocessing_function
        self.mean_instances_per_image = mean_instances_per_image
        self.sample_ratio = sample_ratio


    def __len__(self):
        #计算每一个epoch的迭代次数
        batch_pos_size = int(self.pos_ratio * self.batch_size) * self.mean_instances_per_image

        return math.ceil(len(self.isship_list) / float(batch_pos_size))


    def __getitem__(self, index):
        # 生成batch_size个索引
        batch_pos_size = int(self.pos_ratio * self.batch_size)
        batch_neg_size = self.batch_size - batch_pos_size
        batch_pos_indexs = self.pos_indexes[index*batch_pos_size:(index+1)*batch_pos_size]
        batch_neg_indexs = self.neg_indexes[index*batch_neg_size:(index+1)*batch_neg_size]
        # 根据索引获取datas集合中的数据
        batch_names = [self.isship_list[k] for k in batch_pos_indexs]
        batch_names += [self.nanship_list[k] for k in batch_neg_indexs]
        batch_x, batch_y = self.data_generation(batch_names)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.pos_indexes)
            np.random.shuffle(self.neg_indexes)


    def data_generation(self, batch_names):
        batch_image = []
        batch_label = []
        image_shape = list(self.target_shape) + [3]
        for i, name in enumerate(batch_names):
            image = imread(os.path.join(self.train_img_dir, name))
            image = image / 255
            mask_list = self.train_df['EncodedPixels'][self.train_df['ImageId'] == name].tolist()
            mask = np.zeros((768, 768, 1))
            for item in mask_list:
                instance_mask = rle_decode(item, (768, 768))
                mask[:, :, 0] += instance_mask
            
            if not self.target_shape == (768, 768):
                mask = np.squeeze(mask)
                mask = resize(mask, self.target_shape, order=0, preserve_range=True)
                image = resize(image, self.target_shape)
                mask = np.expand_dims(mask, axis=-1)

            params = self.transformer.get_random_transform(image_shape)
            image = self.transformer.apply_transform(image.astype(K.floatx()), params, order=1)
            mask = self.transformer.apply_transform(mask.astype(K.floatx()), params, order=0)
            patch_shape = self.patch_shape[:2]

            if random.random() < self.sample_ratio and np.sum(mask):
                patch_x, patch_y = self.get_random_pos_patch(image, mask, patch_shape, image_shape[:2])
            else:
                patch_x, patch_y = self.get_random_patch(image, mask, patch_shape, image_shape[:2])

            if self.preprocessing_function:
                patch_x = self.preprocessing_function(patch_x, data_format=K.image_data_format())

            batch_image.append(patch_x)
            batch_label.append(patch_y)

        return np.stack(batch_image, axis=0), np.stack(batch_label, axis=0)

    
    def get_random_pos_patch(self, image, mask, patch_shape, target_shape):
        labeled_mask, n_instances = measure.label(mask, return_num=True)
        props = measure.regionprops(labeled_mask)
        instance_prop = random.choice(props)
        y1, x1, y2, x2 = instance_prop.bbox
        height_range1 = max(y1 - patch_shape[0], 0)
        height_range2 = min(y2 + patch_shape[0], target_shape[0]) - patch_shape[0]
        height_range = range(height_range1, height_range2)

        width_range1 = max(x1 - patch_shape[1], 0)
        width_range2 = min(x2 + patch_shape[1], target_shape[1]) - patch_shape[1]
        width_range = range(width_range1, width_range2)

        y = random.choice(height_range)
        x = random.choice(width_range)

        patch_x = image[y:y+patch_shape[0], x:x+patch_shape[1]]
        patch_y = mask[y:y+patch_shape[0], x:x+patch_shape[1]]

        return patch_x, patch_y


    def get_random_patch(self, image, mask, patch_shape, target_shape):
        height_range = range(0, target_shape[0] - patch_shape[0])
        width_range = range(0, target_shape[1] - patch_shape[1])
        y = random.choice(height_range)
        x = random.choice(width_range)
        patch_x = image[y:y+patch_shape[0], x:x+patch_shape[1]]
        patch_y = mask[y:y+patch_shape[0], x:x+patch_shape[1]]

        return patch_x, patch_y