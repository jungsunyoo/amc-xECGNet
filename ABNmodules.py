from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, Model, Input
# tensorflow.python.keras.layers.noise.GaussianNoise
# from tensorflow.python.keras.layers.noise import GaussianNoise
# from tensorflow.keras import GaussianNoise
from tensorflow.keras.layers import GaussianNoise
import tensorflow as tf

def global_average_pooling(x):
        return K.mean(x, axis = (2, 3))
    
def global_average_pooling_shape(input_shape):
        return input_shape[0:2]
    

    
def get_cam(args):
    x = args[0]
    class_weights = args[1]
    target_classes=args[2]
#     x, class_weights, target_classes
#     cam = K.zeros(x.shape)
    cam=[]
    for curr_class in target_classes:
        for i, w in enumerate(class_weights[:, curr_class]):
            cam += w * x[:,i,:]
    cam %= len(target_classes)
    return cam
    
def CAM_branch(x, n_classes, minimum_len, target_classes, name='cam_branch'): 
    # x is the output image from last convolutional layer
    
    out = layers.GlobalAveragePooling1D(name=name+'_avgpool_1')(x)
    
    out = layers.Dense(n_classes)(out)
    out = layers.Softmax(out)
    class_weights = layers.Layer.get_weights(out)
    cam_out = layers.Lambda(get_cam)([x, class_weights, target_classes])
    

    
    
    return final_out, cam_out
#     out = layers.Lambda(global_average_pooling)(
#     out = global_average_pooling(x)
# att_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, att_out])


#     out = layers.GlobalAveragePooling1D(name=name+'_avgpool_1')(out)
#     out = layers.Dense(256, name=name+'_dense_1')(out)
#     out = layers.Dense(n_classes, name=name+'_dense_2')(out)
#     return layers.Softmax(name='perception_branch_output')(out)

def ieee_baseline_network(x):
    bn_axis=2
    ep = 1.001e-5
    
    # Block 1
    out = layers.Conv1D(64, 3, 1, 'same')(x)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.Conv1D(64, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(out)
    out = layers.Conv1D(128, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.Conv1D(128, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(out)
    
    # Block 2
    for _ in range(3):
        out = layers.Conv1D(256, 3, 1, 'same')(out)
        out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(out)
    
    # Block 3
    for _ in range(3):
        out = layers.Conv1D(512, 3, 1, 'same')(out)
        out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(out)    
    
    # Block 4
    out = layers.Conv1D(512, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.Conv1D(256, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.Conv1D(128, 3, 1, 'same')(out)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)
    out = layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(out)    
    
    return out


def basic_block(x, out_ch, kernel_size=3, stride=1, last_act=True):
    """
    (batch, height, width, channels) => (batch, heigth, width, out_ch)
    """
#     bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    bn_axis=2
    ep = 1.001e-5

    out = layers.Conv1D(out_ch, kernel_size, stride, 'same', use_bias=False)(x)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)

    if last_act is True:
        return layers.Activation(activations.relu)(out)
    else:
        return out


def bottleneck_block(x, out_ch, stride=1):
    """
    (batch, height, width, channels) =>
    stride == 1, (batch, height, width, out_ch)
    stride == 2, (batch, height/2, width/2, out_ch)
    """
    # if x._shape_tuple()[-1] != out_ch:
    if int(x.shape[-1]) != out_ch or stride == 2:
    #     bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        bn_axis=2
        ep = 1.001e-5
        shortcut = layers.Conv1D(out_ch, 1, stride, 'same', use_bias=False)(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(shortcut)
    else:
        shortcut = x

    out = basic_block(x, out_ch//4, 1)
    out = basic_block(out, out_ch//4, 3, stride)
    out = basic_block(out, out_ch, 1, last_act=False)
    out = layers.Add()([out, shortcut])
    return layers.Activation(activations.relu)(out)


def feature_extractor(x, out_ch, n):
    """
    (batch, height, width, channels) =>
        (batch, height/2, width/2, out_ch)
    """
    out = basic_block(x, out_ch)
    out = bottleneck_block(out, out_ch, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, out_ch)

    out = bottleneck_block(out, out_ch*2, 2)
    for _ in range(n-1):
        out = bottleneck_block(out, out_ch*2)
    return out


def attention_branch(x, n, n_classes, name='attention_branch'):
    """
    (batch, height, width, channels) =>
        (batch, n_classes), (batch, height, width, channels)
    """
#     bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    bn_axis=2
    ep = 1.001e-5

    filters = x._shape_tuple()[-1]
    out = bottleneck_block(x, filters*2, 1) # (b,h/2,w/2,f*2*4*2)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1) # (b,h/2,w/2,f*2*4*2)

    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_bn_1')(out)
    out = layers.Conv1D(n_classes, 1, 1, 'same', use_bias=False, activation=activations.relu ,name=name+'_conv_1')(out)

    pred_out = layers.Conv1D(n_classes, 1, 1, 'same', use_bias=False, name=name+'_pred_conv_1')(out)
    pred_out = layers.GlobalAveragePooling1D(name=name+'_gap_1')(pred_out)
    pred_out = layers.Softmax(name='attention_branch_output')(pred_out)

    att_out = layers.Conv1D(1, 1, 1, 'same', use_bias=False, name=name+'_att_conv_1')(out)
    att_out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_att_bn_1')(att_out)
    att_out = layers.Activation(activations.sigmoid, name=name+'_att_sigmoid_1')(att_out)
    # att_out = (x * att_out) + x
    att_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, att_out])
    return pred_out, att_out


def perception_branch(x, cam_map, n, n_classes, name='perception_branch'):
    cam_map = tf.image.resize(cam_map, x.shape)
    filters = cam_map + x
#     cam_map = tf.image.resize(cam_map, filters._shape_tuple())
    
    
    filters = x._shape_tuple()[-1]
#     filters_cam = cam_map.__shape_tuple
#     cam_map.reshape(
    
    
    
    out = bottleneck_block(x, filters*2, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1)

    out = layers.GlobalAveragePooling1D(name=name+'_avgpool_1')(out)
    out = layers.Dense(256, name=name+'_dense_1')(out)
    out = layers.Dense(n_classes, name=name+'_dense_2')(out)
    return layers.Softmax(name='perception_branch_output')(out)


def get_model(input_shape, n_classes, out_ch=256, n=18):
    img_input = Input(shape=input_shape, name='input_image')
#     img_input = GaussianNoise(0.01)(img_input) # YJS added
#     img_input = GaussianNoise(0.01)(img_input)
    backbone = feature_extractor(img_input, out_ch, n)
#     backbone = ieee_baseline_network(img_input, n)
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    per_pred = perception_branch(att_map, n, n_classes)

    model = Model(inputs=img_input, outputs=[att_pred, per_pred])
    return model


def get_custom_model(input_shape, n_classes, minimum_len, target_classes, out_ch=256, n=18):
    img_input = Input(shape=input_shape, name='input_image')
#     img_input = tensorflow.keras.layers.GaussianNoise(0.01)(img_input) # YJS added
#     img_input = GaussianNoise(0.01)(img_input)
#     backbone = feature_extractor(img_input, out_ch, n)
    backbone = ieee_baseline_network(img_input)
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    cam_pred, cam_map = CAM_branch(backbone, n_classes, minimum_len, target_classes)
    per_pred = perception_branch(att_map, cam_map, n, n_classes)
    model = Model(inputs=img_input, outputs=[att_pred, per_pred, cam_pred])
    return model

