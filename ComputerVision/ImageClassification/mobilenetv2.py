import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, DepthwiseConv2D, Dropout, GlobalAveragePooling2D, Input, ReLU 
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, 
                        expansion, 
                        stride, 
                        alpha, 
                        filters, 
                        block_id, 
                        decay=5e-4, 
                        seed=1234):

    prefix = 'block_{}_'.format(block_id)

    channel_axis=-1
    in_channels = K.int_shape(inputs)[channel_axis]
    #in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    if block_id:
        x = Conv2D(expansion * in_channels, 
                   kernel_size=1, 
                   strides=1, 
                   padding='same', 
                   use_bias=False,
                   activation=None, 
                   kernel_initializer=he_normal(seed=seed), 
                   kernel_regularizer=l2(decay), 
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, 
                               momentum=0.999, 
                               name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, 
                        strides=stride, 
                        activation=None, 
                        use_bias=False, 
                        padding='same',
                        depthwise_initializer=he_normal(seed=seed), 
                        depthwise_regularizer=l2(decay), 
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, 
                           momentum=0.999, 
                           name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, 
               kernel_size=1, 
               strides=1, 
               padding='same', 
               use_bias=False, 
               activation=None, 
               kernel_initializer=he_normal(seed=seed), 
               kernel_regularizer=l2(decay), 
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, 
                           momentum=0.999, 
                           name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def MobileNetV2(input_shape=(32, 32, 3),
                alpha=1.0,
                depth_multiplier=1,
                pooling=None,
                classes=10,
                decay=4e-5,
                seed=1234,
                softmax=True):

    # fileter size (first block)
    first_block_filters = _make_divisible(32 * alpha, 8)
    # input shape  (first block)
    img_input = Input(shape=input_shape)

    # model architechture
    x = Conv2D(first_block_filters, 
               kernel_size=3, 
               strides=1, 
               padding='same', 
               use_bias=False, 
               kernel_initializer=he_normal(seed=seed),
               kernel_regularizer=l2(decay), 
               name='Conv1')(img_input)
    x = BatchNormalization(epsilon=1e-3, 
                           momentum=0.999, 
                           name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16,  alpha=alpha, stride=1, expansion=1, block_id=0, decay=decay, seed=seed)

    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=1, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=2, decay=decay, seed=seed)

    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=2, expansion=6, block_id=3, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=4, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=5, decay=decay, seed=seed)

    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=2, expansion=6, block_id=6, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=7, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=8, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=9, decay=decay, seed=seed)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=10, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=11, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=12, decay=decay, seed=seed)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14, decay=decay, seed=seed)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15, decay=decay, seed=seed)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16, decay=decay, seed=seed)
    x = Dropout(rate=0.25)(x)

    # define fileter size (last block)
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters, 
               kernel_size=1, 
               use_bias=False, 
               kernel_initializer=he_normal(seed=seed),
               kernel_regularizer=l2(decay), 
               name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)

    x = GlobalAveragePooling2D(name='global_average_pool')(x)
    if softmax:
        x = Dense(classes, 
                  use_bias=True, 
                  activation='softmax', 
                  kernel_initializer=he_normal(seed=seed), 
                  kernel_regularizer=l2(decay), 
                  name='FC')(x)
    else:
        x = Dense(classes, 
                  use_bias=True, 
                  kernel_initializer=he_normal(seed=seed), 
                  kernel_regularizer=l2(decay), 
                  name='FC')(x)

    # create model of MobileNetV2 (for CIFAR-10)
    model = Model(inputs=img_input, outputs=x, name='mobilenetv2_cifar10')
    return model
