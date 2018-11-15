#encoding: utf-8
from __future__ import print_function
import os
import platform

class Config(object):
    def __init__(self):
        pass

    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        return '\n'.join(str_list)

config = Config()
config.task_name = 'airbus'
config.batch_size = 8


if 'Win' in platform.system():
    print('using data from windows')
    config.data_dir = 'V:\data_repos/airbus_ship_detection'
else:
    print('using data from linux')
    config.data_dir = '/home/share/data_repos/airbus_ship_detection'


config.aug_parms = {'featurewise_normalization': False,
                    'samplewise_normalization':False,
                    'horizontal_flip': True,
                    'vertical_flip': True,
                    'height_shift_range': 0.1,
                    'width_shift_range': 0.1,
                    'rotation_range': 30,
                    'shear_range': 0.1,
                    'fill_mode': 'constant',
                    'cval': 0,
                    'zoom_range': [0.7, 1.5]
                    }

config.light_aug_parms = {'featurewise_normalization': False,
                          'samplewise_normalization':False,
                          'horizontal_flip': True,
                          'vertical_flip': True,
                          'height_shift_range': 0,
                          'width_shift_range': 0,
                          'rotation_range': 0,
                          'shear_range': 0,
                          'fill_mode': 'constant',
                          'cval': 0,
                          'zoom_range': 0
                          }


config.train_dir = os.path.join(config.data_dir, 'train_v2')
config.test_dir = os.path.join(config.data_dir, 'test_v2')
config.log_dir = './logs'
config.weigts_file = 'train_epoch_{epoch:02d}.hdf5'
config.epochs = 100
config.n_works = 20
config.seed = 30
config.pos_ratio = 0.5
config.pretrain_epochs = 5
config.input_shape = (768, 768, 3)
config.submission_dir = './submission'
config.patch_shape = (256, 256, 3)

if __name__ == "__main__":
    print(config)