from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, Model, Input
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import l2_normalize
import tensorflow as tf
import cProfile
from keras import initializers
from keras.backend import tf as ktf
import numpy as np
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

def attention_branch(x, n, n_classes, name='attention_branch'): # heatmap 없는 원래 버전
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
    pred_out = layers.Activation(activations.sigmoid, name='attention_branch_output')(pred_out)
    
    att_out = layers.Conv1D(1, 1, 1, 'same', use_bias=False, name=name+'_att_conv_1')(out)
    att_out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_att_bn_1')(att_out)
    att_out = layers.Activation(activations.sigmoid, name=name+'_att_sigmoid_1')(att_out)
    att_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, att_out])
    return pred_out, att_out

def attention_branch_edit(x, n, n_classes, heatmap, name='attention_branch'):
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
    pred_out = layers.Activation(activations.sigmoid, name='attention_branch_output')(pred_out)

    att_out = layers.Conv1D(1, 1, 1, 'same', use_bias=False, name=name+'_att_conv_1')(out)
    att_out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_att_bn_1')(att_out)
    att_out = layers.Activation(activations.sigmoid, name=name+'_att_sigmoid_1')(att_out)  # output shape = (batch, 12, 1)

    att_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, att_out]) # 다음에 돌릴땐 heatmap을 이거 전에 넣어주는것도 고려해볼만할듯?
    heat_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, heatmap])
    
    edit_map = layers.concatenate([att_out, heat_out], axis=2) # (batch, 12, 256)
    
    
    return pred_out, edit_map


def perception_branch(x,n, n_classes, name='perception_branch'):

    filters = x._shape_tuple()[-1]
    out = bottleneck_block(x, filters*2, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1)

    out = layers.GlobalAveragePooling1D(name=name+'_avgpool_1')(out)
    out = layers.Dense(512, name=name+'_dense_1')(out)
    out = layers.Dense(n_classes, name=name+'_dense_2')(out)
    return layers.Activation(activations.sigmoid, name='perception_branch_output')(out)

def perception_branch_primitive(x,n, n_classes, name='perception_branch'):
    # 특징: same as perception_branch but no 512 dense layer after GAP (for CAM afterwards)
    filters = x._shape_tuple()[-1]
    out = bottleneck_block(x, filters*2, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1)

    out = layers.GlobalAveragePooling1D(name=name+'_avgpool_1')(out)
#     out = layers.Dense(512, name=name+'_dense_1')(out)
    out = layers.Dense(n_classes, name=name+'_dense_2')(out)
    return layers.Activation(activations.sigmoid, name='perception_branch_output')(out)


def primitive_ABN(input_shape, n_classes, minimum_len, n,out_ch=256):
    # use for training ABN for ABN (extract CAM from this model later)
    img_input = Input(shape=input_shape, name='input_image')
    backbone = ieee_baseline_network(img_input)
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    per_pred = perception_branch_primitive(att_map, n, n_classes)
    model = Model(inputs=img_input, outputs=[att_pred, per_pred])
    return model

def ABN_model(input_shape, n_classes, minimum_len, n, out_ch=256):
    img_input = Input(shape=input_shape, name='input_image')
    backbone = ieee_baseline_network(img_input)
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    per_pred = perception_branch(att_map, n, n_classes)
    model = Model(inputs=img_input, outputs=[att_pred, per_pred])
    return model

def edit_ABN_model(input_shape, n_classes, minimum_len, n,out_ch=256): 
    img_input = Input(shape=input_shape, name='input_image')
    
    backbone = ieee_baseline_network(img_input)
    heatmap = Input(shape=(None,1), name='heatmap_image')    
    att_pred, edit_map = attention_branch_edit(backbone, n, n_classes, heatmap)
    per_pred = perception_branch(edit_map, n, n_classes)
    model = Model(inputs=[img_input, heatmap], outputs=[att_pred, per_pred])
    return model


def custom_loss(heatmap, att_map):
    def loss(y_true, y_pred):
        L_abn = binary_crossentropy(y_true, y_pred)
#         print(l2_normalize((heatmap - att_map), axis=1).shape)
#         L_edit = L_abn + np.linalg.norm((heatmap-att_map), axis=1, ord=2)*0.1
        mapp = tf.math.reduce_sum(tf.math.abs(heatmap-att_map), axis=1)
#         mapp = tf.math.l2_normalize((heatmap-att_map), axis=1)
#         L_edit = L_abn + tf.math.l2_normalize(mapp, axis=1)*0.0001#l2_normalize((heatmap - att_map), axis=(1,2))*0.1
        L_edit = L_abn + tf.math.reduce_sum(mapp, axis=1)*0.0001#l2_normalize((heatmap - att_map), axis=(1,2))*0.1
        return L_edit
    return loss

def edit_ABN_model_loss(input_shape, n_classes, minimum_len, n, out_ch=256): # implement as described in ABN edit paper 
    img_input = Input(shape=input_shape, name='input_image') 
    backbone = ieee_baseline_network(img_input)
    heatmap = Input(shape=(None,1), name='heatmap_image')    
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    per_pred = perception_branch(att_map, n, n_classes)
    model = Model(inputs=[img_input, heatmap], outputs=[att_pred, per_pred])
    
    customLoss = custom_loss(heatmap, att_map)

    return model, customLoss



# def cam_primitive_model(input_shape, n_classes, minimum_len, out_ch=256, n=18): # don't use as backbone anymore (bad performance)
#     img_input = Input(shape=input_shape, name='input_image')
#     backbone = ieee_baseline_network(img_input)
    
#     x = layers.GlobalAveragePooling1D()(backbone)
#     x = layers.Dense(256, activation=None)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation(activation='relu')(x)
#     x = layers.Dense(256, activation=None)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation(activation='relu')(x)
#     x = layers.Dense(n_classes, name='dense_final')(x)
#     out = layers.Softmax(name='output')(x)        

#     model = Model(inputs=img_input, outputs=out)
#     return model

# def get_model(input_shape, n_classes, out_ch=256, n=18):
#     # 원본 ABN model 
#     img_input = Input(shape=input_shape, name='input_image')
#     backbone = feature_extractor(img_input, out_ch, n)
#     att_pred, att_map = attention_branch(backbone, n, n_classes)
#     per_pred = perception_branch(att_map, n, n_classes)

#     model = Model(inputs=img_input, outputs=[att_pred, per_pred])
#     return model