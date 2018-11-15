#encoding: utf-8
from __future__ import print_function
from config import config as cfg
from utils import init_env
sess = init_env('0')
from dataset.data import clean_df, dataset_split
from dataset.generators import BaseGenerator
from models import XceptionUnet
from metrics import dice_coef, dice_loss, bce_with_tv_loss, binary_crossentropy
from utils import makedir, get_number_of_steps
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
task_name = 'xception_unet'

def pretrain():
    input_shape = (384, 384, 3)
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.01)
    datagen = BaseGenerator(train_df, train_list[0], train_list[1], 
                            batch_size=16, train_img_dir=cfg.train_dir,
                            pos_ratio=0.5, aug_parms=cfg.aug_parms, 
                            target_shape=input_shape[:2],
                            preprocessing_function=None)

    model = XceptionUnet(input_shape)
    model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[dice_coef])
    steps_per_epoch = get_number_of_steps(len(train_list[0]), int(16*cfg.pos_ratio))
    log_dir = os.path.join(cfg.log_dir, task_name, 'pretrain')
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    callbacks = [ModelCheckpoint(weights_path, monitor='val_loss')]
    his = model.fit_generator(datagen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=cfg.pretrain_epochs,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=cfg.n_works
                              )
    K.clear_session()


def train():
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.01)
    datagen = BaseGenerator(train_df,
                            train_list[0],
                            train_list[1],
                            batch_size=cfg.batch_size,
                            train_img_dir=cfg.train_dir,
                            pos_ratio=0.9,
                            aug_parms=cfg.light_aug_parms,
                            target_shape=cfg.input_shape[:2],
                            preprocessing_function=None)
    model = XceptionUnet(cfg.input_shape)
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    # pretrain_weights = os.path.join(log_dir, 'pretrain', cfg.weigts_file).format(epoch=cfg.pretrain_epochs)
    pretrain_weights = os.path.join(log_dir, cfg.weigts_file).format(epoch=18)
    model.load_weights(pretrain_weights)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
    steps_per_epoch = get_number_of_steps(len(train_list[0]), int(cfg.batch_size*cfg.pos_ratio))
    print('steps_per_epoch', steps_per_epoch)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    callbacks = [ModelCheckpoint(weights_path, monitor='val_loss')]
    his = model.fit_generator(datagen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=cfg.epochs,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=cfg.n_works)
    K.clear_session()


if __name__ == "__main__":
    # pretrain()
    train()