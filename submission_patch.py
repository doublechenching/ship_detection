#encoding: utf-8
from __future__ import print_function
from utils import init_env
init_env('1')
from config import config as cfg
from models.patch_dense_unet import dense121_unet
from dataset.data import compose_ndcube, decompose_ndimage
import os
import glob
import numpy as np
from dataset.data import rle_encode
from skimage import morphology as m
import pandas as pd
from skimage.io import imread
from tqdm import tqdm

cfg.patch_shape = (256, 256, 3)

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = m.label(img[0, :, :, :], connectivity=2)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]


def load_model(weigths_path):
    model = dense121_unet(cfg.patch_shape)
    model.load_weights(weigths_path)

    return model


def submission(model, test_img_dir, opt_threshold=0.5, tta=True):
    test_img_paths = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    pred_rows = []
    for path in tqdm(test_img_paths):
        test_img = imread(path) / 255.0
        test_patches = decompose_ndimage(test_img, cfg.patch_shape[:2])
        test_patches = np.stack(test_patches, axis=0)

        if tta:
            pred_prob1 = model.predict(test_patches)
            pred_prob2 = model.predict(np.flip(test_patches, axis=1))
            pred_prob2 = np.flip(pred_prob2, axis=1)
            pred_prob3 = model.predict(np.flip(test_patches, axis=2))
            pred_prob3 = np.flip(pred_prob3, axis=2)

            test_img4 = np.flip(test_patches, axis=1)
            test_img4 = np.flip(test_img4, axis=2)
            pred_prob4 = model.predict(test_img4)
            pred_prob4 = np.flip(pred_prob4, axis=2)
            pred_prob4 = np.flip(pred_prob4, axis=1)

            pred_prob = (pred_prob1 + pred_prob2 + pred_prob3 + pred_prob4) / 4
        else:
            pred_prob = model.predict(test_patches)

        pred_patches = [patches for patches in pred_prob]
        com_pred = compose_ndcube(pred_patches, list(test_img.shape[:2]) + [1])
        pred_mask = com_pred > opt_threshold
        pred_mask = np.expand_dims(pred_mask, axis=0)  # (b, h, w, c)
        rles = multi_rle_encode(pred_mask)
        name = os.path.split(path)[-1]

        if len(rles)>0:
            for rle in rles:
                pred_rows += [{'ImageId': name, 'EncodedPixels': rle}]
        else:
            pred_rows += [{'ImageId': name, 'EncodedPixels': None}]

    submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('submission_patch_28.csv', index=False)


if __name__ == "__main__":
    cfg.task_name = 'dense_unet_patch'
    epoch = 28
    log_dir = os.path.join(cfg.log_dir, cfg.task_name)
    weights_path = os.path.join(log_dir, cfg.weigts_file.format(epoch=epoch))
    model = load_model(weights_path)
    submission(model, cfg.test_dir)