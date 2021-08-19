import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def ReXNet(input_shape=None, alpha=1.0, classes=1000):
    if input_shape == None:
        # Channels last
        input_shape=(224, 224, 3)

    img_input = layers.Input(shape=input_shape)

    x = ConvBNSwish(img_input, out_channels=32, kernel=3, stride=2, pad='same', name='first_conv')
    x = LinearBottleneck(x, use_se = False, in_channels=32, out_channels=16, t=1, stride=1, name='1')

    x = LinearBottleneck(x, use_se = False, in_channels=16, out_channels=27, t=6, stride=2, name='2')
    x = LinearBottleneck(x, use_se = False, in_channels=27, out_channels=38, t=6, stride=1, name='3')

    x = LinearBottleneck(x, use_se = True, in_channels=38, out_channels=50, t=6, stride=2, name='4')
    x = LinearBottleneck(x, use_se = True, in_channels=50, out_channels=61, t=6, stride=1, name='5')
    x = LinearBottleneck(x, use_se = True, in_channels=61, out_channels=72, t=6, stride=2, name='6')
    x = LinearBottleneck(x, use_se = True, in_channels=72, out_channels=84, t=6, stride=1, name='7')
    x = LinearBottleneck(x, use_se = True, in_channels=84, out_channels=95, t=6, stride=1, name='8')
    x = LinearBottleneck(x, use_se = True, in_channels=95, out_channels=106, t=6, stride=1, name='9')
    x = LinearBottleneck(x, use_se = True, in_channels=106, out_channels=117, t=6, stride=1, name='10')
    x = LinearBottleneck(x, use_se = True, in_channels=117, out_channels=128, t=6, stride=1, name='11')
    x = LinearBottleneck(x, use_se = True, in_channels=128, out_channels=140, t=6, stride=2, name='12')
    x = LinearBottleneck(x, use_se = True, in_channels=140, out_channels=151, t=6, stride=1, name='13')
    x = LinearBottleneck(x, use_se = True, in_channels=151, out_channels=162, t=6, stride=1, name='14')
    x = LinearBottleneck(x, use_se = True, in_channels=162, out_channels=174, t=6, stride=1, name='15')
    x = LinearBottleneck(x, use_se = True, in_channels=174, out_channels=185, t=6, stride=1, name='16')

    pen_channels = int(1280 * alpha)

    x = ConvBNSwish(x, pen_channels)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(classes, name='predict', use_bias=True)(x)
    outputs = layers.Flatten()(x)

    model = keras.models.Model(img_input, outputs, name="ReXNet")
    return model
    

#### BUILDER FUNCTIONS #####


def ConvBNSwish(_in, out_channels, kernel=1, stride=1, pad='valid', num_group=1, name=''):
    x = layers.Conv2D(out_channels, kernel, strides=stride, 
                        padding=pad, groups=num_group, use_bias=False, name=name)(_in)
    x = layers.BatchNormalization(name=name+'BN')(x)
    x = tf.nn.swish(x)
    return x

def ConvBNAct(_in, out_channels, kernel=1, stride=1, pad='valid', num_group=1, active=True, relu6=False):
    x = layers.Conv2D(out_channels, kernel, strides=stride, padding=pad, use_bias=False)(_in)
    x = layers.BatchNormalization()(x)
    if active:
        return layers.ReLU(6.)(x) if relu6 else layers.ReLU()(x)
    else:
        return x

def DepthBNAct(_in, kernel=3, stride=1):
    x = layers.DepthwiseConv2D(kernel, strides=stride, use_bias=False, padding='same')(_in)
    x = layers.BatchNormalization()(x)
    return x

def Squeeze(_in, out_channels, se_ratio=12):
    # First pool
    avg_pool = layers.AvgPool2D(1)(_in)
    
    # Next, fully connected sequential layers
    x = layers.Conv2D(out_channels // se_ratio, kernel_size=1, padding='valid')(avg_pool)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, kernel_size=1, padding='valid', activation='sigmoid')(x)

    return layers.multiply([_in, x])

def LinearBottleneck(_in, in_channels, out_channels, t, 
                        stride, use_se=True, se_ratio=12, name=''):
    use_shortcut = stride == 1 and in_channels <= out_channels
    x = _in

    if t != 1:
        dw_channels = in_channels * t
        x = ConvBNSwish(x, dw_channels, name='Bottleneck'+name)
    else:
        dw_channels = in_channels

    x = DepthBNAct(x, kernel=3, stride=stride)

    if use_se:
        x = Squeeze(x, dw_channels, se_ratio)
    
    x = layers.ReLU(6.)(x)
    x = ConvBNAct(x, out_channels, active=False, relu6=True)

    if use_shortcut:
        res = layers.Conv2D(out_channels, 1, name='bottleneck_shortcut'+name)(_in)
        res = layers.BatchNormalization(name='shortcut_bn'+name)(res)
        return layers.Add(name='residual' + name)([x, res])
    return x
