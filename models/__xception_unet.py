#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
from keras import layers
from keras import backend as K
from keras import Model
from keras import utils as keras_utils

def conv_bn_relu(x, n_filter, kernel_size, name,
                 activation='relu',
                 strides=(1, 1),
                 padding='same',
                 use_bias=False):
    """naive convolutin block
    x -> Conv ->BN -> Act(optional)
    """
    x = layers.Conv2D(n_filter, kernel_size, 
               strides=strides, padding=padding,
               use_bias=use_bias, name=name)(x)
    x = layers.BatchNormalization(name=name+'_bn')(x)
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)

    return x

def sepconv_bn_relu(x, n_filter, kernel_size, name,
                 activation='relu',
                 strides=(1, 1),
                 padding='same',
                 use_bias=False):
    """depthwise separable convolution block
    x => sepconv -> BN -> Act(optional) 
    """
    x = layers.SeparableConv2D(n_filter, kernel_size, 
                        strides=strides, padding=padding,
                        use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name+'_bn')(x)
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)

    return x


def xception_block1(x, n_filter, kernel_size, name,
                   activation='relu'):
    """down_sampling Xception block
    x => sepconv block -> sepconv block -> Maxpooling -> add(Act(x))
    """
    residual = conv_bn_relu(x, n_filter[1], (1, 1), strides=(2, 2), padding='same', 
                            activation=None, name=name+'_res_conv')
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)
    x = sepconv_bn_relu(x, n_filter[0], kernel_size, padding='same', name=name+'_sepconv1')
    x = sepconv_bn_relu(x, n_filter[1], kernel_size, padding='same', activation=None, name=name+'_sepconv2')
    x = layers.MaxPooling2D(kernel_size, strides=(2, 2), padding='same', name=name+'_pool')(x)
    x = layers.add([x, residual])
    
    return x


def fine_seg_block(x, n_filters, kernel_size, name, activation='relu'):
    """创建定位模块
    整体结构为， CBA->CBA(point wise)
    Args:
        input_layer：输入结点
        n_filters： 特征数量

    Returns:
        整体输出通道数不变
    """
    residual = conv_bn_relu(x, n_filters[1], (1, 1), padding='same', 
                            activation=None, name=name+'_res_conv')
    if activation:
        x = layers.Activation(activation, name=name+'_act')(x)
    x = sepconv_bn_relu(x, n_filters[0], kernel_size, padding='same', name=name+'_sepconv1')
    x = sepconv_bn_relu(x, n_filters[1], kernel_size, padding='same', activation=None, name=name+'_sepconv2')
    x = layers.add([x, residual])

    return x


def up_sampling_block(x, n_filter, kernel_size, name,
                      activation='relu', up_size=(2, 2)):
    """Xception block
    x => sepconv block -> sepconv block -> sepconv block-> add(Act(x)) =>
    """
    x = layers.UpSampling2D(size=up_size, name=name+'up')(x)
    if activation:
        x = layers.Activation('relu', name=name+'_act')(x)   
    x = sepconv_bn_relu(x, n_filter, kernel_size, padding='same', activation=None, name=name+'_sepconv1')

    return x


def deep_supervise_block(x, n_class, name, activation='relu'):
    if activation:
        x = layers.Activation(activation, name=name+'_act')(x)
    if n_class > 2:
        x = layers.Conv2D(n_class, (1, 1), padding='same', activation='softmax', name=name)(x)
    else:
        x = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name)(x)

    return x


def load_imagenet_weights(model, include_top):
    TF_WEIGHTS_PATH = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.4/'
        'xception_weights_tf_dim_ordering_tf_kernels.h5')
    TF_WEIGHTS_PATH_NO_TOP = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.4/'
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if include_top:
        weights_path = keras_utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels.h5',
            TF_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')

    else:
        weights_path = keras_utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b0042744bf5b25fce3cb969f33bebb97')
    print("pretain from imagenet----", weights_path)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)


def XceptionUnet(input_shape=None,
                 n_class=1,
                 pretrain_weights='imagenet'):
    """Instantiates the Xception architecture.

    Note that the default input image size for this model is 299x299.

    Args:
        input_shape: list or tuple, (height, width, channel)
        pretrain_weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        n_class: int, num of class
    Returns:

    Raise:

    Examples:

    """
    inputs = layers.Input(shape=input_shape)
    conv1_1 = conv_bn_relu(inputs,  32, (5, 5), strides=(3, 3), padding='same', name='block1_conv1')
    conv1_2 = conv_bn_relu(conv1_1, 64, (3, 3), name='block1_conv2')
    conv2 = xception_block1(conv1_2, [128, 128], (3, 3), activation=None, name='block2')
    conv3 = xception_block1(conv2, [256, 256], (3, 3), name='block3')
    conv4 = xception_block1(conv3, [728, 728], (3, 3), name='block4')
    conv5 = xception_block1(conv4, [728, 728], (3, 3), name='block5')

    up5 = up_sampling_block(conv5, 728, (3, 3), name='up_sampling5')
    con5 = layers.concatenate([up5, conv4])
    fine5 = fine_seg_block(con5, [512, 512], (3, 3), name='fine5')
    ds5 = deep_supervise_block(fine5, n_class, name='dsb5')

    up4 = up_sampling_block(fine5, 256, (3, 3), name='up_sampling4')
    con4 = layers.concatenate([up4, conv3])
    fine4 = fine_seg_block(con4, [256, 256], (3, 3), name='fine4')
    ds4 = deep_supervise_block(fine4, n_class, name='dsb4')

    up3 = up_sampling_block(fine4, 128, (3, 3), name='up_sampling3')
    con3 = layers.concatenate([up3, conv2])
    fine3 = fine_seg_block(con3, [128, 128], (3, 3), name='fine3')
    ds3 = deep_supervise_block(fine3, n_class, name='dsb3')

    up2 = up_sampling_block(fine3, 64, (3, 3), name='up_sampling2')
    con2 = layers.concatenate([up2, conv1_2])
    fine2 = fine_seg_block(con2, [64, 64], (3, 3), name='fine2')
    ds2 = deep_supervise_block(fine2, n_class, name='dsb2')

    con1 = layers.concatenate([fine2, conv1_1])
    fine1 = fine_seg_block(con1, [32, 32], (3, 3), name='fine1')

    up1 = layers.UpSampling2D(size=(3, 3), name='up_sampling1')(fine1)
    ds1 = deep_supervise_block(up1, n_class, name='dsb1')
    
    ds5 = layers.UpSampling2D((24, 24))(ds5)
    ds4 = layers.UpSampling2D((12, 12))(ds4)
    ds3 = layers.UpSampling2D((6, 6))(ds3)
    ds2 = layers.UpSampling2D((3, 3))(ds2)
    dsn = layers.concatenate([ds5, ds4, ds3, ds2, ds1])
    
    dsn = layers.Conv2D(1, 1, activation='sigmoid', name='fusion_dsn')(dsn)
    # Create model
    model = Model(inputs, dsn, name='XceptionUnet')
    # Load weights
    if pretrain_weights == 'imagenet':
        load_imagenet_weights(model, include_top=False)
    elif os.path.exists(pretrain_weights):
        print("pretain from imagenet----", pretrain_weights)
        model.load_weights(pretrain_weights)
    else:
        print("weight path does not exist!")

    return model


if __name__ == "__main__":
    model = XceptionUnet([512, 512, 3], n_class=2, pretrain_weights='imagenet')
    model.summary()
    keras_utils.plot_model(model, to_file='model1.png', show_shapes=True)
