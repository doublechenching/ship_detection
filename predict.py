#encoding: utf-8
from __future__ import print_function
from config import config as cfg
from utils import init_env
sess = init_env('3')
from dataset.data import clean_df, dataset_split, create_data
from models import dense121_unet
from metrics import dice_coef, dice_loss, calc_score_all_image
import os
import numpy as np
from tqdm import tqdm

def load_data():
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val = dataset_split(train_gp, test_size=0.05, seed=42)
    val_list = val['ImageId'].tolist()
    return val_list, train_df


def load_model(weigths_path):
    model = dense121_unet(cfg.input_shape)
    model.load_weights(weigths_path)

    return model


def predict(model, val_list, train_df, batch_size=10):
    thresholds = np.linspace(0.2, 0.8, 7)
    batchs = len(val_list) // batch_size
    scores = np.zeros( (batchs, len(thresholds)) )
    for i in tqdm(range(batchs)):
        image_name_list = val_list[i * batch_size : (i+1) * batch_size]
        val_img, val_mask = create_data(image_name_list, train_df, cfg.train_dir, cfg.input_shape[:2])
        pred_mask = model.predict(val_img)
        for j, threshold in enumerate(thresholds):
            F2 = calc_score_all_image(val_mask, pred_mask, threshold=threshold)*10
            scores[i, j] = F2
    val_F2 = np.sum(scores, axis=0) / (len(val_list) // 10 * 10)
    print(val_F2)
    opt_threshold = thresholds[np.argmax(val_F2)]
    print('best threshold is ', opt_threshold)

def predict_on_generator(model, val_list, train_df, batch_size=10):
    thresholds = np.linspace(0.2, 0.8, 7)
    batchs = len(val_list) // batch_size
    scores = np.zeros( (batchs, len(thresholds)) )
    for i in tqdm(range(batchs)):
        image_name_list = val_list[i * batch_size : (i+1) * batch_size]
        val_img, val_mask = create_data(image_name_list, train_df, cfg.train_dir, cfg.input_shape[:2])
        pred_mask = model.predict(val_img)
        for j, threshold in enumerate(thresholds):
            F2 = calc_score_all_image(val_mask, pred_mask, threshold=threshold)*10
            scores[i, j] = F2
    val_F2 = np.sum(scores, axis=0) / (len(val_list) // 10 * 10)
    print(val_F2)
    opt_threshold = thresholds[np.argmax(val_F2)]
    print('best threshold is ', opt_threshold)


if __name__ == "__main__":
    cfg.task_name = 'dense_unet'
    epoch = 8
    val_list = load_data()
    log_dir = os.path.join(cfg.log_dir, cfg.task_name)
    weights_path = os.path.join(log_dir, cfg.weigts_file.format(epoch=epoch))
    print('predicting weights from ---', weights_path)
    val_list, train_df = load_data()
    model = load_model(weights_path)
    predict(model, val_list, train_df)