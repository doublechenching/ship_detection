# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras import utils as keras_utils
from keras import backend
from keras import layers
from keras import Model
from .__res_unet import conv_block as identity_block, conv_bn_relu, deep_supervise_block
from keras_applications import imagenet_utils

bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    # 下采样
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool', padding='same')(x)

    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    # 扩展通道
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    # 特征提取
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])

    return x


def dense_unet(blocks, input_shape, weights='imagenet', trainable=True):
    """Instantiates the DenseNet architecture.
    """
    inputs = layers.Input(shape=input_shape)

    x = inputs
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv', padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    conv1 = x

    # original is maxpooling
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False, name='new_conv1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    conv5 = x

    up5 = layers.UpSampling2D((2, 2))(conv5)
    up5 = layers.concatenate([up5, conv4])
    up5 = identity_block(up5, 3, [128, 128, 512], stage=6, block='a', strides=1)

    up4 = layers.UpSampling2D((2, 2))(up5)
    up4 = layers.concatenate([up4, conv3])
    up4 = identity_block(up4, 3, [64, 64, 256], stage=7, block='a', strides=1)

    up3 = layers.UpSampling2D((2, 2))(up4)
    up3 = layers.concatenate([up3, conv2])
    up3 = conv_bn_relu(up3, 3, 32, 'stage8')

    up2 = layers.UpSampling2D((2, 2))(up3)
    up2 = layers.concatenate([up2, conv1])
    up2 = conv_bn_relu(up2, 3, 1, 'stage9')

    dsn1 = deep_supervise_block(up2, name='dsn1')

    conv = conv_bn_relu(conv1, 3, 16, 'new_reduce')

    attention1 = conv_bn_relu(up5, 3, 1, 'attention1_1')
    attention1 = layers.UpSampling2D((8, 8))(attention1)
    attention1 = layers.concatenate([attention1, conv])
    attention1 = conv_bn_relu(attention1, 3, 1, 'attention1_2')

    attention2 = conv_bn_relu(up4, 3, 1, 'attention2_1')
    attention2 = layers.UpSampling2D((4, 4))(attention2)
    attention2 = layers.concatenate([attention1, attention2])
    attention2 = conv_bn_relu(attention2, 3, 1, 'attention2_2')
    attention2 = layers.concatenate([attention2, conv])
    attention2 = conv_bn_relu(attention2, 3, 1, 'attention2_3')

    attention3 = conv_bn_relu(up3, 3, 1, 'attention3_1')
    attention3 = layers.UpSampling2D((2, 2))(attention3)
    attention3 = layers.concatenate([attention3, attention2])
    attention3 = conv_bn_relu(attention3, 3, 1, 'attention3_2')
    attention3 = layers.concatenate([attention3, conv])
    attention3 = conv_bn_relu(attention3, 3, 1, 'attention3_3')

    dsn = layers.concatenate([dsn1, attention3])
    dsn = layers.UpSampling2D((2, 2))(dsn)
    dsn = conv_bn_relu(dsn, 3, 1, 'fusion_attention')
    dsn = layers.Conv2D(1, 1, activation='sigmoid', name='fusion_dsn')(dsn)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, dsn, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, dsn, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, dsn, name='densenet201')
    else:
        model = Model(inputs, dsn, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if blocks == [6, 12, 24, 16]:
            weights_path = keras_utils.get_file(
                'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                DENSENET121_WEIGHT_PATH,
                cache_subdir='models',
                file_hash='9d60b8095a5708f2dcce2bca79d332c7')
        elif blocks == [6, 12, 32, 32]:
            weights_path = keras_utils.get_file(
                'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                DENSENET169_WEIGHT_PATH,
                cache_subdir='models',
                file_hash='d699b8f76981ab1b30698df4c175e90b')
        elif blocks == [6, 12, 48, 32]:
            weights_path = keras_utils.get_file(
                'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                DENSENET201_WEIGHT_PATH,
                cache_subdir='models',
                file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)

    if not trainable:
        for layer in model.layers:
            if 'block' in layer.name and 'conv' in layer.name:
                layer.trainable = False
                print(layer.name, ' is not trainalbe')

    return model


def dense121_unet(input_shape, weights='imagenet', trainable=True):
    return dense_unet([6, 12, 24, 16], input_shape, weights, trainable)


def dense169_unet(input_shape, weights='imagenet', trainable=True):
    return dense_unet([6, 12, 32, 32], input_shape, weights, trainable)


def dense201_unet(input_shape, weights='imagenet', trainable=True):
    return dense_unet([6, 12, 48, 32], input_shape, weights, trainable)


def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)


if __name__ == "__main__":
    from keras.utils import plot_model

    plot_model(dense121_unet((256, 256, 3)), show_shapes=True)