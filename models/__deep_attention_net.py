#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
from keras import backend
from keras import layers
from keras import Model
from keras import utils as keras_utils

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

if backend.image_data_format() == 'channels_last':
        bn_axis = 3
else:
    bn_axis = 1


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', 
                      padding='same')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c', 
                      padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', 
                      padding='same')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',
                      padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',
                             padding='same')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def deep_supervise_block(x, n_class=1, name='dsn'):
    if n_class > 2:
        x = layers.Conv2D(n_class, (1, 1), padding='same', activation='softmax', name=name)(x)
    else:
        x = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name)(x)

    return x

def conv_bn_relu(x, kernel_size, filters, name, use_bias=False):
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      name=name+'conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name+'bn')(x)
    x = layers.Activation('relu')(x)
    
    return x


def resnet50_unet(input_shape, weights='imagenet'):
    """Instantiates the ResNet50 architecture.
    """
    img_input = layers.Input(shape=input_shape)
    x = img_input
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    conv1 = x
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = conv_bn_relu(x, 3, 64, name='reduce')

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    conv2 = x
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    conv3 = x
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    conv4 = x
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')
    conv5 = x

    attentiona1 = conv_bn_relu(conv2, 3, 64, name='attentiona1')
    attentiona1 = layers.add([attentiona1, conv1])

    attentiona2 = conv_bn_relu(conv3, 3, 256, name='attentiona2')
    attentiona2 = layers.add([attentiona2, conv2])

    attentiona3 = conv_bn_relu(conv4, 3, 512, name='attentiona3')
    attentiona3 = layers.add([attentiona3, conv3])

    attentiona4 = conv_bn_relu(conv5, 3, 512, name='attentiona3')
    attentiona4 = layers.add([attentiona4, conv4])

    attentionb1 = conv_bn_relu(attentiona2, 3, 64, name='attentionb1')
    attentionb1 = layers.add([attentionb1, conv1])

    attentionb2 = conv_bn_relu(attentiona3, 3, 256, name='attentionb2')
    attentionb2 = layers.add([attentionb2, conv2])

    attentiona3 = conv_bn_relu(conv4, 3, 512, name='attentiona3')
    attentiona3 = layers.add([attentiona3, conv3])

    up5 = layers.UpSampling2D((2, 2))(conv5)
    up5 = conv_block(up5, 3, [256, 256, 1024], stage=6, block='up5a')
    up5 = layers.add([up5, conv4])
    up5 = identity_block(up5, 3, [128, 128, 512], stage=6, block='a')
    
    up4 = layers.UpSampling2D((2, 2))(up5)
    up4 = layers.concatenate([up4, conv3])
    up4 = identity_block(up4, 3, [64, 64, 256], stage=7, block='a')

    up3 = layers.UpSampling2D((2, 2))(up4)
    up3 = layers.concatenate([up3, conv2])
    up3 = conv_bn_relu(up3, 3, 32, 'stage8')

    up2 = layers.UpSampling2D((2, 2))(up3)
    up2 = layers.concatenate([up2, conv1])
    up2 = conv_bn_relu(up3, 3, 1, 'stage8')
    
    dsn5 = deep_supervise_block(conv5, name='dsn5')
    dsn4 = deep_supervise_block(up5, name='dsn4')
    dsn3 = deep_supervise_block(up4, name='dsn3')
    dsn2 = deep_supervise_block(up3, name='dsn2')
    dsn1 = deep_supervise_block(up2, name='dsn1')

    dsn5 = layers.UpSampling2D((48, 48))(dsn5)
    dsn4 = layers.UpSampling2D((24, 24))(dsn4)
    dsn3 = layers.UpSampling2D((12, 12))(dsn3)
    dsn2 = layers.UpSampling2D((6, 6))(dsn2)
    dsn1 = layers.UpSampling2D((3, 3))(dsn1)
    dsn = layers.concatenate([dsn5, dsn4, dsn3, dsn2, dsn1])
    dsn = layers.Conv2D(1, 1, activation='sigmoid', name='fusion_dsn')(dsn)
    model = Model(img_input, dsn, name='res_unet')

    # Load weights.
    if weights == 'imagenet':
        weights_path = keras_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        model.load_weights(weights_path, by_name=True)

    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model
