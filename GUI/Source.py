import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    Activation,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Conv2D,
    Conv2DTranspose,
    concatenate,
    add,
)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)

def iou_metric(label, pred):
    return tf.compat.v1.py_func(get_iou_vector, [label, pred>0.5], tf.float64)


#MODEL 5 Layer 128
# ---------------------------------------------------------------------------------------------------
def UNet5_128(input_img):
    c1 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)
    # -----------------------------------------------------------------------------------------------------------------

    c2 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    c2 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)
    # -----------------------------------------------------------------------------------------------------------------

    c3 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    c3 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.1)(p3)

    # -----------------------------------------------------------------------------------------------------------------

    c4 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    c4 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.1)(p4)

    # -----------------------------------------------------------------------------------------------------------------

    c5 = Conv2D(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    c5 = Conv2D(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    # ------UMSAMPLE STARTS---------------------------------------------------------------------------------------------

    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.1)(u6)

    c6 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    c6 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    # ---------------------------------------------------------------------------------------------------

    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.1)(u7)

    c7 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    c7 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    # ---------------------------------------------------------------------------------------------------

    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)

    c8 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    c8 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    # ---------------------------------------------------------------------------------------------------

    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='valid')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)

    c9 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    c9 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(input_img, outputs)
    return model

input_img1 = Input((101, 101, 1), name='img')
model5_128 = UNet5_128(input_img1)
model5_128.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[iou_metric])
model5_128.load_weights("Unet_101_5layers_GD.h5")


#MODEL 112
# ---------------------------------------------------------------------------------------------------
def UNet112(input_img):
    # inputs = Input(input_size)

    c1 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    # p1 = Dropout(0.1)(p1)
    # -----------------------------------------------------------------------------------------------------------------

    c2 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    c2 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    # p2 = Dropout(0.1)(p2)
    # -----------------------------------------------------------------------------------------------------------------

    c3 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    c3 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    c3 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    # p3 = Dropout(0.1)(p3)

    # -----------------------------------------------------------------------------------------------------------------

    c4 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    c4 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    c4 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    # p4 = Dropout(0.1)(p4)

    # -------------------------------------------------------BRIDGE----------------------------------------------------------

    c5 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    c5 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    c5 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    # p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    # p5 = Dropout(0.1)(p5)

    # ------UPSAMPLE STARTS-----------------------------------------------------------------------------------------------------------

    u1 = UpSampling2D(size=(2, 2))(c5)
    u1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)

    merge_u1 = concatenate([u1, c4])

    cu1 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u1)
    cu1 = BatchNormalization()(cu1)
    cu1 = Activation('relu')(cu1)

    cu1 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu1)
    cu1 = BatchNormalization()(cu1)
    cu1 = Activation('relu')(cu1)

    cu1 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu1)
    cu1 = BatchNormalization()(cu1)
    cu1 = Activation('relu')(cu1)

    # --------------------------------------------------------------------------------------------------------------------------
    u2 = UpSampling2D(size=(2, 2))(cu1)
    u2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)

    merge_u2 = concatenate([u2, c3])
    # u1 = Dropout(0.1)(u6)

    cu2 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u2)
    cu2 = BatchNormalization()(cu2)
    cu2 = Activation('relu')(cu2)

    cu2 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu2)
    cu2 = BatchNormalization()(cu2)
    cu2 = Activation('relu')(cu2)

    cu2 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu2)
    cu2 = BatchNormalization()(cu2)
    cu2 = Activation('relu')(cu2)

    # ---------------------------------------------------------------------------------------------------

    u3 = UpSampling2D(size=(2, 2))(cu2)
    u3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)

    merge_u3 = concatenate([u3, c2])
    # u1 = Dropout(0.1)(u6)

    cu3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u3)
    cu3 = BatchNormalization()(cu3)
    cu3 = Activation('relu')(cu3)

    cu3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu3)
    cu3 = BatchNormalization()(cu3)
    cu3 = Activation('relu')(cu3)

    # ---------------------------------------------------------------------------------------------------

    u4 = UpSampling2D(size=(2, 2))(cu3)
    u4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Activation('relu')(u4)

    merge_u4 = concatenate([u4, c1])
    # u1 = Dropout(0.1)(u6)

    cu4 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u4)
    cu4 = BatchNormalization()(cu4)
    cu4 = Activation('relu')(cu4)

    cu4 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu4)
    cu4 = BatchNormalization()(cu4)
    cu4 = Activation('relu')(cu4)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(cu4)

    model = Model(input_img, outputs)
    return model

input_img2 = Input((112, 112, 1), name='img')
model112 = UNet112(input_img2)
model112.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[iou_metric])
model112.load_weights("Unet_112_GD.h5")


#MODEL 6 Layer 128
# ---------------------------------------------------------------------------------------------------
def UNet6_128(input_img):
    # inputs = Input(input_size)

    c1 = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    # p1 = Dropout(0.1)(p1)
    # -----------------------------------------------------------------------------------------------------------------

    c2 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    c2 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    # p2 = Dropout(0.1)(p2)
    # -----------------------------------------------------------------------------------------------------------------

    c3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    c3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    # p3 = Dropout(0.1)(p3)

    # -----------------------------------------------------------------------------------------------------------------

    c4 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    c4 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    # p4 = Dropout(0.1)(p4)

    # -----------------------------------------------------------------------------------------------------------------

    c5 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    c5 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    # p5 = Dropout(0.1)(p5)

    # -------------------------------------------------------BRIDGE----------------------------------------------------------

    c6 = Conv2D(1024, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    c6 = Conv2D(1024, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    # p6 = MaxPooling2D((2, 2))(c6)
    # p6 = Dropout(0.1)(p6)

    # ------UPSAMPLE STARTS---------------------------------------------------------------------------------------------
    u1 = UpSampling2D(size=(2, 2))(c6)
    u1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)

    merge_u1 = concatenate([u1, c5])
    # u1 = Dropout(0.1)(u6)

    cu1 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u1)
    cu1 = BatchNormalization()(cu1)
    cu1 = Activation('relu')(cu1)

    cu1 = Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu1)
    cu1 = BatchNormalization()(cu1)
    cu1 = Activation('relu')(cu1)

    # ---------------------------------------------------------------------------------------------------

    u2 = UpSampling2D(size=(2, 2))(cu1)
    u2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)

    merge_u2 = concatenate([u2, c4])
    # u1 = Dropout(0.1)(u6)

    cu2 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u2)
    cu2 = BatchNormalization()(cu2)
    cu2 = Activation('relu')(cu2)

    cu2 = Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu2)
    cu2 = BatchNormalization()(cu2)
    cu2 = Activation('relu')(cu2)

    # ---------------------------------------------------------------------------------------------------

    u3 = UpSampling2D(size=(2, 2))(cu2)
    u3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)

    merge_u3 = concatenate([u3, c3])
    # u1 = Dropout(0.1)(u6)

    cu3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u3)
    cu3 = BatchNormalization()(cu3)
    cu3 = Activation('relu')(cu3)

    cu3 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu3)
    cu3 = BatchNormalization()(cu3)
    cu3 = Activation('relu')(cu3)

    # ---------------------------------------------------------------------------------------------------

    u4 = UpSampling2D(size=(2, 2))(cu3)
    u4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Activation('relu')(u4)

    merge_u4 = concatenate([u4, c2])
    # u1 = Dropout(0.1)(u6)

    cu4 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u4)
    cu4 = BatchNormalization()(cu4)
    cu4 = Activation('relu')(cu4)

    cu4 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu4)
    cu4 = BatchNormalization()(cu4)
    cu4 = Activation('relu')(cu4)

    # ---------------------------------------------------------------------------------------------------

    u5 = UpSampling2D(size=(2, 2))(cu4)
    u5 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u5)
    u5 = BatchNormalization()(u5)
    u5 = Activation('relu')(u5)

    merge_u5 = concatenate([u5, c1])
    # u1 = Dropout(0.1)(u6)

    cu5 = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge_u5)
    cu5 = BatchNormalization()(cu5)
    cu5 = Activation('relu')(cu5)

    cu5 = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(cu5)
    cu5 = BatchNormalization()(cu5)
    cu5 = Activation('relu')(cu5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(cu5)

    model = Model(input_img, outputs)
    return model

input_img3 = Input((128, 128, 1), name='img')
model6_128 = UNet6_128(input_img3)
model6_128.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[iou_metric])
model6_128.load_weights("Unet_128_6layers_GD.h5")


def res_block(x, filter_size, size):
    conv = Conv2D(size, kernel_size=(filter_size, filter_size), padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(size, kernel_size=(filter_size, filter_size), padding='same')(conv)
    conv = BatchNormalization()(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    res = add([shortcut, conv])
    res = Activation('relu')(res)
    return res
def Res_UNet(input_img):
    c1 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = res_block(c1, 3, 16)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)
    # -----------------------------------------------------------------------------------------------------------------

    c2 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    c2 = res_block(c2, 3, 32)

    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)
    # -----------------------------------------------------------------------------------------------------------------

    c3 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    c3 = res_block(c3, 3, 64)

    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.1)(p3)

    # -----------------------------------------------------------------------------------------------------------------

    c4 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    c4 = res_block(c4, 3, 128)

    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.1)(p4)

    # -----------------------------------------------------------------------------------------------------------------

    c5 = Conv2D(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    c5 = res_block(c5, 3, 256)

    # ------UMSAMPLE STARTS---------------------------------------------------------------------------------------------

    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.1)(u6)

    c6 = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    c6 = res_block(c6, 3, 128)

    # ---------------------------------------------------------------------------------------------------

    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.1)(u7)

    c7 = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    c7 = res_block(c7, 3, 64)

    # ---------------------------------------------------------------------------------------------------

    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)

    c8 = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    c8 = res_block(c8, 3, 32)

    # ---------------------------------------------------------------------------------------------------

    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='valid')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)

    c9 = Conv2D(16, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    c9 = res_block(c9, 3, 16)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(input_img, outputs)
    return model

input_img4 = Input((101, 101, 1), name='img')
modelRes = Res_UNet(input_img4)
modelRes.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[iou_metric])
modelRes.load_weights("Res_Unet_101_GD.h5")


def Run(img, modelnum):

    img_ap = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if modelnum == 112:
        img = cv2.resize(img, (112, 112))
        img_tensor = img_to_array(img)
        img_tensor = img_tensor/255.0
        img_ap.append(img_tensor)
        img_ap = np.asarray(img_ap)
        res = model112.predict(img_ap, verbose=1)
    elif modelnum == 5128:
        img = cv2.resize(img, (101, 101))
        img_tensor = img_to_array(img)
        img_tensor = img_tensor / 255.0
        img_ap.append(img_tensor)
        img_ap = np.asarray(img_ap)
        res = model5_128.predict(img_ap, verbose=1)
    elif modelnum == 6128:
        img = cv2.resize(img, (128, 128))
        img_tensor = img_to_array(img)
        img_tensor = img_tensor / 255.0
        img_ap.append(img_tensor)
        img_ap = np.asarray(img_ap)
        res = model6_128.predict(img_ap, verbose=1)
    else:
        img = cv2.resize(img, (101, 101))
        img_tensor = img_to_array(img)
        img_tensor = img_tensor / 255.0
        img_ap.append(img_tensor)
        img_ap = np.asarray(img_ap)
        res = modelRes.predict(img_ap, verbose=1)

    res_f = (res > 0.5).astype(np.uint8)
    img_res = res_f.squeeze()
    for c1 in range(img_res.shape[0]):
        for c2 in range(img_res.shape[1]):
            if img_res[c1][c2] == 1:
                img_res[c1][c2] = 255

    return img_res
