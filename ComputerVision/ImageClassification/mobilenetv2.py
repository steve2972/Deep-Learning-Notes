import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def MobileNetV2(input_shape=None, alpha = 1.0, classes=1000):
    if input_shape == None:
        # Channels last
        input_shape = (224, 224, 3)

    img_input = layers.Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D()(img_input)
    x = layers.Conv2D(first_block_filters, kernel_size=3, strides=2,
                        padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = _inverted_res_block(x, first_block_filters, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    x = _inverted_res_block(x, 16, filters=24, alpha=alpha, stride=2, expansion=1, block_id=1)
    x = _inverted_res_block(x, 24, filters=24, alpha=alpha, stride=1, expansion=1, block_id=2)

    x = _inverted_res_block(x, 24, filters=32, alpha=alpha, stride=2, expansion=1, block_id=3)
    x = _inverted_res_block(x, 32, filters=32, alpha=alpha, stride=1, expansion=1, block_id=4)
    x = _inverted_res_block(x, 32, filters=32, alpha=alpha, stride=1, expansion=1, block_id=5)
    
    x = _inverted_res_block(x, 32, filters=64, alpha=alpha, stride=2, expansion=1, block_id=6)
    x = _inverted_res_block(x, 64, filters=64, alpha=alpha, stride=1, expansion=1, block_id=7)
    x = _inverted_res_block(x, 64, filters=64, alpha=alpha, stride=1, expansion=1, block_id=8)
    x = _inverted_res_block(x, 64, filters=64, alpha=alpha, stride=1, expansion=1, block_id=9)
    
    x = _inverted_res_block(x, 64, filters=96, alpha=alpha, stride=1, expansion=1, block_id=10)
    x = _inverted_res_block(x, 96, filters=96, alpha=alpha, stride=1, expansion=1, block_id=11)
    x = _inverted_res_block(x, 96, filters=96, alpha=alpha, stride=1, expansion=1, block_id=12)

    x = _inverted_res_block(x, 96, filters=160, alpha=alpha, stride=2, expansion=1, block_id=13)
    x = _inverted_res_block(x, 160, filters=160, alpha=alpha, stride=1, expansion=1, block_id=14)
    x = _inverted_res_block(x, 160, filters=160, alpha=alpha, stride=1, expansion=1, block_id=15)

    x = _inverted_res_block(x, 160, filters=320, alpha=alpha, stride=1, expansion=1, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation='sigmoid', name='predictions')(x)

    model = keras.Model(img_input, x, name='mobilenetv2_%0.2f'%(alpha))

    return model


#### BUILDER FUNCTIONS #####


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, in_channels, expansion, stride, alpha, filters, block_id):
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                            use_bias=False, activation=None, name=prefix+'expand')(x)
        x = layers.BatchNormalization(name=prefix+'expand_BN')(x)
        x = layers.ReLU(6., name=prefix+'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise Convolution
    if stride == 2:
        x = layers.ZeroPadding2D()(x)
    
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                                use_bias=False, padding='same' if stride==1 else 'valid')(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    # Projection
    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same',
                        use_bias=False, activation=None)(x)
    x = layers.BatchNormalization()(x)

    if in_channels == pointwise_filters and stride==1:
        return layers.Add()([inputs, x])
    return x
