#encoding: utf-8
from __future__ import print_function
from config import config as cfg
from utils import init_env
sess = init_env('2, 5, 6, 7')
gpus = 4
from dataset.data import clean_df, dataset_split
from dataset.generators import BasePatchGenerator
from models.patch_dense_unet import dense121_unet
from metrics import dice_coef, dice_loss, bce_with_tv_loss, iou_loss, binary_crossentropy
from utils import makedir, get_number_of_steps
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from callbacks import MultiGPUCheckpoint
from keras.optimizers import Adam
import os
from keras.utils import multi_gpu_model
import tensorflow as tf
from dataset.data import fit_ndimage_param

task_name = 'dense_unet_patch'
input_shape = (768, 768, 3)
patch_shape = (256, 256, 3)
batch_size = 16 * gpus


def lr_schedule(epoch):

    lr = 1e-3
    if epoch < 20:
        lr = 1e-3
    elif epoch < 30:
        lr = 1e-4
    elif epoch < 40:
        if epoch % 10 == 9:
            lr *= 0.5e-3
        elif 6 <= epoch % 10 < 9:
            lr *= 1e-3
        elif 3 <= epoch % 10 < 6:
            lr *= 1e-2
        elif epoch % 10 < 3:
            lr *= 1e-1
        print('Learning rate: ', lr)


    return lr


def load_gen():
    train_gp, train_df = clean_df(cfg.data_dir)
    train_list, val_list = dataset_split(train_gp, test_size=0.05, seed=42)
    datagen = BasePatchGenerator(train_df, train_list[0], train_list[1],
                                 batch_size=batch_size,
                                 train_img_dir=cfg.train_dir,
                                 pos_ratio=0.95, aug_parms=cfg.aug_parms,
                                 target_shape=input_shape[:2],
                                 sample_ratio=0.95,
                                 patch_shape=patch_shape,
                                 preprocessing_function=None)

    return datagen


def pretrain():
    datagen = load_gen()
    model = dense121_unet(patch_shape, trainable=False)
    model.compile(optimizer=Adam(), loss=iou_loss, metrics=[dice_coef])
    flod, _ = fit_ndimage_param(input_shape[:2], patch_shape[:2])
    steps_per_epoch = get_number_of_steps(flod[0] * flod[1] * len(datagen), batch_size)
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
def train():
    datagen = load_gen()
    model = dense121_unet(patch_shape)
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    pretrain_weights = os.path.join(cfg.log_dir, task_name, cfg.weigts_file).format(epoch=12)
    model.load_weights(pretrain_weights)
    model.compile(optimizer=Adam(1e-4), loss=iou_loss, metrics=[dice_coef])
    flod, _ = fit_ndimage_param(input_shape[:2], patch_shape[:2])
    steps_per_epoch = get_number_of_steps(flod[0] * flod[1] * len(datagen), batch_size)
    print('steps_per_epoch', steps_per_epoch)
    weights_path = os.path.join(log_dir, cfg.weigts_file)
    callbacks = [ModelCheckpoint(weights_path, monitor='val_loss')]
    his = model.fit_generator(datagen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=cfg.epochs,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=cfg.n_works)


def train_on_parallel():
    datagen = load_gen()
    with tf.device('/cpu:0'):
        cpu_model = dense121_unet(patch_shape)
    pretrain_weights = os.path.join(cfg.log_dir, task_name, cfg.weigts_file).format(epoch=5)
    cpu_model.load_weights(pretrain_weights)
    parallel_model = multi_gpu_model(cpu_model, gpus=gpus)
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    parallel_model.compile(optimizer=Adam(1e-3, amsgrad=True),
                           loss=dice_loss,
                           metrics=[dice_coef, binary_crossentropy])
    flod, _ = fit_ndimage_param(input_shape[:2], patch_shape[:2])

    weights_path = os.path.join(log_dir, cfg.weigts_file)
    lrs = LearningRateScheduler(lr_schedule)
    callbacks = [MultiGPUCheckpoint(weights_path, cpu_model, monitor='val_loss'), lrs]
    his = parallel_model.fit_generator(datagen,
                                       steps_per_epoch=flod[0]*flod[1]*len(datagen),
                                       epochs=cfg.epochs,
                                       callbacks=callbacks,
                                       use_multiprocessing=True,
                                       workers=cfg.n_works)


if __name__ == "__main__":
    # pretrain()
    # train()
    train_on_parallel()