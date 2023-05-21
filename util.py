import math
import os
import sys

import numpy as np
import cv2
import tensorflow as tf
import PIL.Image as pilimg
from tensorflow.python.framework.type_spec import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.python.keras import applications
from tensorflow.keras import losses
from tensorflow.python.ops.variable_scope import get_variable


def l1_norm(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.math.abs(y_true - y_pred))


def l2_norm(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.math.square(y_true - y_pred))


def l1_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return K.mean(K.abs(y_pred - y_true), axis=-1)


def l2_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return K.mean(K.square(y_pred - y_true), axis=-1)


def psnr_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def perceptual_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    low = perceptual_low_loss(y_true, y_pred)
    high = perceptual_high_loss(y_true, y_pred)
    return 0.2 * low + 0.8 * high


def perceptual_low_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    image_shape = (256, 256, 3)
    img_width = 256
    img_height = 256

    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
    loss_model.trainable = False
    return K.sum(K.square(loss_model(y_true) - loss_model(y_pred))) / (img_width * img_height)


def perceptual_high_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    image_shape = (256, 256, 3)
    img_width = 256
    img_height = 256

    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv2').output)
    loss_model.trainable = False
    return K.sum(K.square(loss_model(y_true) - loss_model(y_pred))) / (img_width * img_height)


def wasserstein_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return K.mean(y_true * y_pred)


def mix_loss(y_true, y_pred):
    alpha = 0.85

    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return (alpha * tf.keras.backend.mean(tf.norm(y_true - y_pred, axis=1))) + ((1 - alpha) * (1. - tf.keras.backend.mean(tf.image.ssim(y_true, y_pred, 1.0))))


def mse_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


def epe_loss(y_true, y_pred):
    # EPE
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.norm(y_true - y_pred, axis=1))


def EPE_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(l2_norm(y_true - y_pred))


def SSIMLoss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def custom_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    alph = 0.84
    squared_difference = tf.square(y_true - y_pred)
    if alph > 0.:
        return alph*tf.reduce_mean(squared_difference, axis=-1)+(1.-alph)*(1.-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0), axis=-1))
    else:
        beta=-alph
        return beta*tf.reduce_mean(squared_difference, axis=-1)+(1.-beta)*(1.-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred,1.0), axis=-1))


def ssim_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return 1.-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def msssim_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    return 1.-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))


def load_data(images_dir, start=0, end=100000):
    name_list = []
    image_list = []
    # np_images = None

    files = os.listdir(images_dir)

    # images = np.empty((256, 256, 3))

    cnt = 0
    for file in files[start:end]:
        try:
            path = os.path.join(images_dir, file)

            image = cv2.imread(path, cv2.IMREAD_COLOR)

            name_list.append(file)
            image_list.append(image)

            cnt += 1
            print(cnt, 'filename:', file, images_dir)
            del image, file

        except FileNotFoundError as e:
            print('ERROR : ', e)

    names = np.array(name_list)
    images = np.stack(image_list)

    return names, images


def save_images(names, images, save_dir):
    for name, image in zip(names, images):
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, image)
        # cv2.imwrite(save_path, image * 255)


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.config.run_functions_eagerly(True)
    tf.config.experimental_run_functions_eagerly(True)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)




