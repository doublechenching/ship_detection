#encoding: utf-8
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          MaxPool2D, SpatialDropout2D, Deconv2D, add, 
                          UpSampling2D, concatenate)
from keras import backend as K
from keras import Model
from keras import utils as keras_utils
from keras import layers

if K.image_data_format() == 'channels_last':
    ch_axis = -1
    print('channel last')
else:
    ch_axis = 1
    print('channel first')


def load_model_weights(model, weights, include_top=True):
    WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.1/'
                    'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.1/'
                        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    else:
        model.load_weights(weights)
    print("pretain from imagenet----", weights)
    

def conv_bn_relu(x,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 activation='relu',
                 use_bias=False,
                 use_bn=False,
                 name=None
                 ):

    filters = int(filters)
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if(use_bn):
        x = BatchNormalization(axis=ch_axis, name=name+'_bn')(x)
    x = Activation(activation, name=name+'_act')(x)

    return x


def vgg_block(x, filters, kernel_size=3, padding='same', use_bn=True, name=None):
    """x => n*conv -> maxpooling =>
    """
    for i, n_filter in enumerate(filters):
        x = conv_bn_relu(x, n_filter, kernel_size, padding=padding, 
            use_bn=use_bn, name=name+'_conv'+str(i+1))
    x = MaxPool2D((2, 2), strides=(2, 2), name=name+'_pool')(x)
    return x

def conv_up_con(low, up, n_filters, name='None'):
    x = conv_bn_relu(low, n_filters[0], 3, name=name+'_conv1')
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, up])
    x = conv_bn_relu(x, n_filters[1], 3, name=name+'_conv2')

    return x


def vgg_unet(input_shape, base_filter=64, pretrain_weghts='imagenet', trainable=True):
    input_img = Input(input_shape)
    # block1
    block1 = vgg_block(input_img, [base_filter]*2, name='block1')

    # block2
    block2 = vgg_block(block1, [base_filter*2]*2, name='block2')

    # block3
    block3 = vgg_block(block2, [base_filter*4]*3, name='block3')

    # block4
    block4 = vgg_block(block3, [base_filter*8]*3, name='block4')

    # block5
    block5 = vgg_block(block4, [base_filter*16]*3, name='block5')

    up5 = conv_up_con(block5, block4, [base_filter*8, base_filter*8], name='up5')
    up4 = conv_up_con(up5, block3, [base_filter*4, base_filter*4], name='up4')
    up3 = conv_up_con(up4, block2, [base_filter*2, base_filter*2], name='up3')
    up2 = conv_up_con(up3, block1, [base_filter*2, base_filter*2], name='up2')

    conv = conv_bn_relu(block1, 8, 3,  name='reduce')

    attention1 = conv_bn_relu(up5, 1, 3, name='attention1_1')
    attention1 = layers.UpSampling2D((8, 8))(attention1)
    attention1 = layers.concatenate([attention1, conv])
    attention1 = conv_bn_relu(attention1, 1, 3, name='attention1_2')

    attention2 = conv_bn_relu(up4, 1, 3, name='attention2_1')
    attention2 = layers.UpSampling2D((4, 4))(attention2)
    attention2 = layers.concatenate([attention1, attention2])
    attention2 = conv_bn_relu(attention2, 1, 3, name='attention2_2')
    attention2 = layers.concatenate([attention2, conv])
    attention2 = conv_bn_relu(attention2, 1, 3, name='attention2_3')

    attention3 = conv_bn_relu(up3, 1, 3, name='attention3_1')
    attention3 = layers.UpSampling2D((2, 2))(attention3)
    attention3 = layers.concatenate([attention3, attention2])
    attention3 = conv_bn_relu(attention3, 1, 3, name='attention3_2')
    attention3 = layers.concatenate([attention3, conv])
    attention3 = conv_bn_relu(attention3, 1, 3, name='attention3_3')

    dsn = concatenate([attention3, up2])
    dsn = UpSampling2D((2, 2))(dsn)
    dsn = Conv2D(1, 1, activation='sigmoid', name='dsn')(dsn)

    model = Model([input_img], [dsn])
    
    load_model_weights(model, pretrain_weghts)

    if not trainable:
        for layer in model.layers:
            if 'block' in layer.name:
                layer.trainable = False
                if 'block1' in layer.name:
                    layer.trainable = True
            
    return model


if __name__ == "__main__":
    from keras.utils import plot_model

    model = vgg_unet([512, 512, 1])
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
