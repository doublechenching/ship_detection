#encoding: utf-8
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          MaxPool2D, SpatialDropout2D, Deconv2D, add, Multiply, Cropping2D)
from keras import backend as K
from keras import Model
from keras import utils as keras_utils

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
        model.load_weights(weights, by_name=True, skip_mismatch=True)
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


def fcn8s(input_shape, n_class=2, base_filter=64, pretrain_weghts='imagenet'):
    input_img = Input(input_shape)
    height, width = input_shape[0], input_shape[1]
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

    # flatten, if size is (512, 512), kernel size is (7x7)
    kernel_size = (int(height / 32), int(width / 32))
    fc6 = conv_bn_relu(block5, 4096, kernel_size, name='fc6')
    fc6d = SpatialDropout2D(0.5)(fc6)
    
    fc7 = conv_bn_relu(fc6d, 4096, 1, name='f7')
    fc7d = SpatialDropout2D(0.5)(fc7)

    # Gives the classifications scores for each of the 21 classes
    # including background
    fcn32 = Conv2D(n_class, 1, name='fcn32')(fc7d)                      # 1/32

    up_fcn32 = Deconv2D(n_class, 4, strides=2, use_bias=False,
                        padding='same', name='up_fcn32')(fcn32)
    fcn16 = Conv2D(n_class, 1, name='fcn16')(block4)
    fcn16 = add([up_fcn32, fcn16])                                      # 1/16

    up_fcn16 = Deconv2D(n_class, 4, strides=2, use_bias=False, 
                        padding='same', name='up_fcn16')(fcn16)
    fcn8 = Conv2D(n_class, 1, name='fcn8')(block3)
    fcn8 = add([up_fcn16, fcn8])                                        # 1/8
    if n_class > 2:
        fcn1 = Deconv2D(n_class, 16, strides=8, use_bias=False, 
                        padding='same', activation='softmax', 
                        name='output')(fcn8)                            # 1/1
    else:
        fcn1 = Deconv2D(1, 16, strides=8, use_bias=False, 
                        padding='same', activation='sigmoid', 
                        name='output')(fcn8)                            # 1/1
    model = Model([input_img], [fcn1])
    
    load_model_weights(model, pretrain_weghts)

    return model


if __name__ == "__main__":
    from keras.utils import plot_model
    model = fcn8s([512, 512, 1])
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
