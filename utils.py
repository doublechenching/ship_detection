#encoding: utf-8
from __future__ import print_function
from keras import backend as K
import tensorflow as tf
import os
import numpy as np

def init_env(cuda_vis_dev='0', phase=1):
    """init trainging environment

    # Args:
        cuda_vis_dev: str, visiable gpu devices
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis_dev
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.97, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    K.set_session(sess)
    return sess

def get_number_of_steps(n_samples, batch_size):
    """get keras training or validation steps
    
    # Args
        n_samples: int, number of samples in dataset
        batch_size：int, batch size

    # Returns
        kreas trian steps
    """
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist folder---', path)