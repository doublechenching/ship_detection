from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from skimage.io import imread
from skimage import morphology as morph
from skimage import measure
from skimage.transform import resize
import os
import copy

def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1

def rle_decode(rle_item, shape):
    """行程编码转mask"""
    rle_list = str(rle_item).split()
    tmp_flat = np.zeros(shape[0]*shape[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, shape).T
    else:
        start = rle_list[::2]
        length = rle_list[1::2]
        for i, v in zip(start, length):
            tmp_flat[(int(i) - 1): (int(i) - 1) + int(v)] = 1
        mask = np.reshape(tmp_flat, shape).T

    return mask


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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


def clean_df(csv_path):
    path = os.path.join(csv_path, 'train_ship_segmentations_v2.csv')
    train_df = pd.read_csv(path)
    print('length of dataframe is ', train_df.shape)
    # remove bug images
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
    train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
    counts = train_df['isnan'].value_counts()
    print('image with ship: ', counts[0])
    print('image without ship: ', counts[1])

    train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
    train_gp = train_df.groupby('ImageId').sum()
    train_gp = train_gp.reset_index()
    train_gp['class'] = train_gp['area'].apply(calc_class)
    train_gp['class'].value_counts()

    return train_gp, train_df

def df_to_list(df, prefix='train'):
    isship_list = df['ImageId'][df['isnan']==0].tolist()
    random.seed(42)
    isship_list = random.sample(isship_list, len(isship_list))
    nanship_list = df['ImageId'][df['isnan']==1].tolist()
    nanship_list = random.sample(nanship_list, len(nanship_list))
    print(prefix+'set images with ship: ', len(isship_list))
    print(prefix+'set images without ship: ', len(nanship_list))

    return isship_list, nanship_list

def dataset_split(df, test_size=0.01, seed=42):
    train, val = train_test_split(df, test_size=test_size, stratify=df['class'].tolist(), random_state=seed)
    train_isship_list, train_nanship_list = df_to_list(train)
    val_isship_list, val_nanship_list = df_to_list(val, prefix='test')
    val_list = val['ImageId'].tolist()
    return [train_isship_list, train_nanship_list], val_list

def create_data(image_list, train_df, train_img_dir='./', target_shape=(768, 768)):
    batch_img = []
    batch_mask = []
    for name in image_list:
        image = imread(os.path.join(train_img_dir, name))
        image = image / 255
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
        mask = np.zeros((768, 768, 1))
        for item in mask_list:
            instance_mask = rle_decode(item, (768, 768))
            mask[:, :, 0] += instance_mask
        if not target_shape == (768, 768):
            mask = np.squeeze(mask)
            mask = resize(mask, target_shape, order=0, preserve_range=True)
            image = resize(image, target_shape)
            mask = np.expand_dims(mask, axis=-1)

        batch_img.append(image)
        batch_mask.append(mask)
        
    return np.array(batch_img), np.array(batch_mask)


def create_mask(labels):
    labels = measure.label(labels, neighbors=8, background=0)
    tmp = morph.dilation(labels > 0, morph.square(9))    
    tmp2 = morph.watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = morph.dilation(tmp, morph.square(7))
    msk = (255 * tmp).astype('uint8')
    
    props = measure.regionprops(labels)
    msk0 = 255 * (labels > 0)
    msk0 = msk0.astype('uint8')
    msk1 = np.zeros_like(labels, dtype='bool')
    max_area = np.max([p.area for p in props])
    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                if max_area > 4000:
                    sz = 6
                else:
                    sz = 3
            else:
                sz = 3
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 1
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 2
            uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), 
                                    max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
                msk0[y0, x0] = 0
    
    msk1 = 255 * msk1
    msk1 = msk1.astype('uint8')
    msk2 = np.zeros_like(labels, dtype='uint8')
    msk = np.stack((msk0, msk1, msk2))
    msk = np.rollaxis(msk, 0, 3)

    return msk


def fit_ndimage_param(image_size, crop_size, min_overlap_rate=0.33):
    """
    获取分解参数

    Args:
    -----
        image_shape: a tuple of 3d volume or 2d image shape    
        crop_shape: a list or tuple of crop image shape
        min_overlap_rate: 分解的图像至少有多少比例的重合区域
    Returns:
    --------
        fold: cube在image进行了几折
        overlap: 相邻crop区域的重叠长度  
    """
    assert(min_overlap_rate <= 0.5, "overlap cant not be bigger than 0.5")
    image_size = np.asarray(image_size)
    crop_size = np.asarray(crop_size)
    min_overlap = min_overlap_rate * crop_size
    dim = image_size - min_overlap
    # ceil天花板，向右取整
    fold = np.ceil(dim / (crop_size - min_overlap))
    fold = fold.astype('int')
    overlap = np.true_divide((fold * crop_size - image_size), (fold - 1))
    return fold, overlap


def decompose_ndimage(ndimage, crop_size, min_overlap_rate=0.33):
    """
    decompose ndimage into list of cubes

    Args:
    -----
        ndimage: array, ndimage, when 3d, size is (depth, height, width)
        crop_size: cube size tuple    
        min_overlap_rate: 分解的图像至少有多少比例的重合区域       
    Returns:
    --------
        ndcubes:      
    """
    # get parameters for decompose
    fold, overlap = fit_ndimage_param(
        ndimage.shape[:-1], crop_size, min_overlap_rate)
    start_point_list = []

    for dim_fold, dim_overlap, dim_len in zip(fold, overlap, crop_size):
        start_point = np.asarray(range(dim_fold)) * (dim_len - dim_overlap)
        start_point = np.ceil(start_point)
        start_point = start_point.astype('int')
        start_point_list.append(start_point)
    ndcube_list = []
    if len(ndimage.shape[:-1]) == 2:
        for i in range(fold[0]):
            for j in range(fold[1]):
                if start_point_list[1][j] + crop_size[1] > ndimage.shape[1] or start_point_list[0][i] + crop_size[0] > ndimage.shape[0]:
                    crop_slice = (slice(ndimage.shape[0] - crop_size[0], ndimage.shape[0]),
                                  slice(ndimage.shape[1] - crop_size[1], ndimage.shape[1]))
                else:
                    crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                                  slice(start_point_list[1][j], start_point_list[1][j] + crop_size[1]))
                ndcube_list.append(ndimage[crop_slice])
            # print(start_point_list[0][i], ':',start_point_list[0][i] + crop_size[0],'\t', start_point_list[1][j], ':',start_point_list[1][j] + crop_size[1])

    else:
        for i in range(fold[0]):
            for j in range(fold[1]):
                for k in range(fold[2]):
                    crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                                  slice(
                                      start_point_list[1][j], start_point_list[1][j] + crop_size[1]),
                                  slice(start_point_list[2][j], start_point_list[2][j] + crop_size[2]))
                    ndcube_list.append(ndimage[crop_slice])

    return ndcube_list


def compose_ndcube(ndcube_list, ndimage_shape, min_overlap_rate=0.33):
    """对单通道图像进行合成"""
    mask = np.zeros(ndimage_shape)
    ndimage = np.zeros(ndimage_shape)
    crop_size = ndcube_list[0].shape[:-1]
    fold, overlap = fit_ndimage_param(
        ndimage_shape[:-1], crop_size, min_overlap_rate)
    start_point_list = []
    for dim_fold, dim_overlap, dim_len in zip(fold, overlap, crop_size):
        start_point = np.asarray(range(dim_fold)) * (dim_len - dim_overlap)
        start_point = np.ceil(start_point)
        start_point = start_point.astype('int')
        start_point_list.append(start_point)
    cnt = 0
    if len(ndimage_shape[:-1]) == 2:
        for i in range(fold[0]):
            for j in range(fold[1]):
                if start_point_list[1][j] + crop_size[1] > ndimage.shape[1] or start_point_list[0][i] + crop_size[0] > ndimage.shape[0]:
                    crop_slice = (slice(ndimage.shape[0] - crop_size[0], ndimage.shape[0]),
                                  slice(ndimage.shape[1] - crop_size[1], ndimage.shape[1]))
                else:
                    crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                                  slice(start_point_list[1][j], start_point_list[1][j] + crop_size[1]))
                mask[crop_slice] = mask[crop_slice] + 1.0
                ndimage[crop_slice] = ndimage[crop_slice] + ndcube_list[cnt]
                cnt = cnt + 1
    else:
        for i in range(fold[0]):
            for j in range(fold[1]):
                for k in range(fold[2]):
                    crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                                  slice(
                                      start_point_list[1][j], start_point_list[1][j] + crop_size[1]),
                                  slice(start_point_list[2][j], start_point_list[2][j] + crop_size[2]))
                    mask[crop_slice] = mask[crop_slice] + 1
                    ndimage[crop_slice] = ndcube_list[cnt]
                    cnt = cnt + 1
    ndimage[mask > 0] = ndimage[mask > 0] / mask[mask > 0]
    
    return ndimage


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from skimage.io import imread
    image = imread('V:/data_repos/airbus_ship_detection/train_v2/00c4be6fa.jpg') / 255
    print(image.shape)
    patch_shape = (128, 128, 3)
    plt.figure('org')
    plt.imshow(image)
    plt.figure('demo')
    flod, _ = fit_ndimage_param(image.shape[:-1], patch_shape[:2])
    ndcubes = decompose_ndimage(image, patch_shape[:2])
    for i in range(flod[0]):
        for j in range(flod[1]):
            index = i*flod[0] + j+1
            s_p = plt.subplot(flod[0], flod[1], i*flod[0] + j+1)
            s_p.imshow(ndcubes[index - 1])
            
    com_image = compose_ndcube(ndcubes, image.shape)
    plt.figure('com')
    plt.imshow(com_image)
    plt.show()