#encoding: utf-8
from __future__ import print_function
from config import config as cfg
from utils import init_env
sess = init_env('5, 6, 7')
gpus = 3
from dataset.data import clean_df, dataset_split
from dataset.generators import BaseGenerator, PredictGenerator
from models import dense121_unet
from metrics import dice_coef, dice_loss, bce_with_tv_loss, iou_loss, binary_crossentropy
from utils import makedir, get_number_of_steps
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from callbacks import MultiGPUCheckpoint, lr_schedule
from keras.optimizers import Adam
import os
from keras.utils import multi_gpu_model
import tensorflow as tf

task_name = 'dense_unet_refine'
def pretrain():
    input_shape = (384, 384, 3)
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.05)
    datagen = BaseGenerator(train_df, train_list[0], train_list[1], 
                            batch_size=16, train_img_dir=cfg.train_dir,
                            pos_ratio=0.8, aug_parms=cfg.aug_parms,
                            target_shape=input_shape[:2],
                            preprocessing_function=None)
                            
    val_gen = PredictGenerator(train_df, val_list, train_img_dir=cfg.train_dir,
                               batch_size=16, target_shape=input_shape[:2])
    model = dense121_unet(input_shape, trainable=False)
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=[dice_coef])
    steps_per_epoch = get_number_of_steps(len(train_list[0]), int(16*cfg.pos_ratio))
    log_dir = os.path.join(cfg.log_dir, task_name, 'pretrain')
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    callbacks = [ModelCheckpoint(weights_path, monitor='val_loss')]
    his = model.fit_generator(datagen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=cfg.pretrain_epochs,
                              validation_data=val_gen,
                              validation_steps=1000,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=cfg.n_works
                              )


def train():
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.05, seed=42)
    datagen = BaseGenerator(train_df, train_list[0], train_list[1], 
                            batch_size=cfg.batch_size, train_img_dir=cfg.train_dir,
                            pos_ratio=0.9, aug_parms=cfg.aug_parms,
                            target_shape=cfg.input_shape[:2],
                            preprocessing_function=None)
    val_gen = PredictGenerator(train_df, val_list,
                               train_img_dir=cfg.train_dir,
                               batch_size=cfg.batch_size,
                               target_shape=cfg.input_shape[:2])
    
    model = dense121_unet(cfg.input_shape)
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    pretrain_weights = os.path.join(cfg.log_dir, 'dense_unet3', cfg.weigts_file).format(epoch=12)
    model.load_weights(pretrain_weights)
    model.compile(optimizer=Adam(1e-4), loss=iou_loss, metrics=[dice_coef])
    steps_per_epoch = get_number_of_steps(len(train_list[0]), int(cfg.batch_size*cfg.pos_ratio))
    print('steps_per_epoch', steps_per_epoch)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    callbacks = [ModelCheckpoint(weights_path, monitor='val_loss')]
    his = model.fit_generator(datagen,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_gen,
                              validation_steps=1000,
                              epochs=cfg.epochs,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=cfg.n_works)

def train_on_parallel():
    cfg.batch_size  = cfg.batch_size * gpus
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.05, seed=42)
    datagen = BaseGenerator(train_df, train_list[0], train_list[1], 
                            batch_size=cfg.batch_size, train_img_dir=cfg.train_dir,
                            pos_ratio=0.9, aug_parms=cfg.aug_parms,
                            target_shape=cfg.input_shape[:2],
                            preprocessing_function=None)
    val_gen = PredictGenerator(train_df, val_list,
                               train_img_dir=cfg.train_dir,
                               batch_size=cfg.batch_size,
                               target_shape=cfg.input_shape[:2])
    with tf.device('/cpu:0'):
        cpu_model = dense121_unet(cfg.input_shape)
    pretrain_weights = os.path.join(cfg.log_dir, 'dense_unet3', cfg.weigts_file).format(epoch=12)
    cpu_model.load_weights(pretrain_weights)
    parallel_model = multi_gpu_model(cpu_model, gpus=gpus)
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    parallel_model.compile(optimizer=Adam(1e-4, amsgrad=True),
                           loss=bce_with_tv_loss,
                           metrics=[dice_coef, binary_crossentropy])
    steps_per_epoch = get_number_of_steps(len(train_list[0]), int(cfg.batch_size*cfg.pos_ratio))
    print('steps_per_epoch', steps_per_epoch)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    # lrs = LearningRateScheduler(lr_schedule)
    callbacks = [MultiGPUCheckpoint(weights_path, cpu_model, monitor='val_loss')]
    his = parallel_model.fit_generator(datagen,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=val_gen,
                                       validation_steps=200,
                                       epochs=cfg.epochs,
                                       callbacks=callbacks,
                                       use_multiprocessing=True,
                                       workers=cfg.n_works)

if __name__ == "__main__":
    # pretrain()
    # train()
    train_on_parallel()