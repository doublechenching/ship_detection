from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from skimage.io import imread
from skimage import morphology as m
import os

def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1

def rle_to_mask(rle_list, SHAPE):
    """行程编码转mask"""
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T

    return mask


def calc_area_for_rle(rle_str):
    """计算面积"""
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area


def calc_class(area):
    """根据面积将图像划分为6个等级"""
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.005:
        return 1
    elif area < 0.015:
        return 2
    elif area < 0.025:
        return 3
    elif area < 0.035:
        return 4
    elif area < 0.045:
        return 5
    else:
        return 6


def clean_df(csv_path, remove10000=True):
    path = os.path.join(csv_path, 'train_ship_segmentations_v2.csv')
    train_df = pd.read_csv(path)
    print('length of dataframe is ', train_df.shape)
    # remove bug images
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
    train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
    counts = train_df['isnan'].value_counts()
    print('image with ship: ', counts[0])
    print('image without ship: ', counts[1])
    # remove 100000 non-ship images
    if remove10000:
        train_df = train_df.sort_values('isnan', ascending=False)
        train_df = train_df.iloc[100000:]
    train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
    train_gp = train_df.groupby('ImageId').sum()
    train_gp = train_gp.reset_index()
    train_gp['class'] = train_gp['area'].apply(calc_class)
    train_gp['class'].value_counts()

    return train_gp, train_df


def df_to_list(df, prefix='train'):
    isship_list = df['ImageId'][df['isnan']==0].tolist()
    isship_list = random.sample(isship_list, len(isship_list))
    nanship_list = df['ImageId'][df['isnan']==1].tolist()
    nanship_list = random.sample(nanship_list, len(nanship_list))
    print(prefix+'set images with ship: ', len(isship_list))
    print(prefix+'set images without ship: ', len(nanship_list))

    return isship_list, nanship_list

def dataset_split(df, test_size=0.01):
    train, val = train_test_split(df, test_size=test_size, stratify=df['class'].tolist())
    train_isship_list, train_nanship_list = df_to_list(train)
    val_isship_list, val_nanship_list = df_to_list(val, prefix='test')

    return [train_isship_list, train_nanship_list], val


def create_data(image_list, train_df, train_img_dir='./'):
    batch_img = []
    batch_mask = []
    for name in image_list:
        image_path = os.path.join(train_img_dir, name)
        tmp_img = imread(image_path)
        batch_img.append(tmp_img)
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
        one_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            one_mask[:,:,0] += tmp_mask
        batch_mask.append(one_mask)
    img = np.stack(batch_img, axis=0)
    mask = np.stack(batch_mask, axis=0)
    img = img / 255.0
    mask = mask / 255.0

    return img, mask


def base_generator(train_df, isship_list, nanship_list, batch_size, train_img_dir='../images'):

    while True:
        batch_img_names_nan = np.random.choice(isship_list, int(batch_size / 2))
        batch_img_names_is = np.random.choice(nanship_list, int(batch_size / 2))
        batch_img = []
        batch_mask = []
        for name in batch_img_names_nan:
            image_path = os.path.join(train_img_dir, name)
            tmp_img = imread(image_path)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)

        for name in batch_img_names_is:
            tmp_img = imread(os.path.join(train_img_dir, name))
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)
        img = img / 255.0
        mask = mask / 255.0
        
        yield img, mask


import numpy as np
import cv2

def masks_to_bounding_boxes(labeled_mask):
     if labeled_mask.max() == 0:
         return labeled_mask
    else:
         img_box = np.zeros_like(labeled_mask)
         for label_id in range(1, labeled_mask.max() + 1, 1):
            label = np.where(labeled_mask == label_id, 1, 0).astype(np.uint8)
            _, cnt, _ = cv2.findContours(label, 1, 2)
            rect = cv2.minAreaRect(cnt[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
           cv2.drawContours(img_box, [box], 0, label_id, -1)
     return img_box