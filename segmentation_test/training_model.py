import os
from fastcore.all import *
from datetime import datetime
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
from tensorflow.keras import backend as K
# from keras_unet_collection import models
# import segmentation_models as sm
# from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Lambda, Concatenate
# from keras.metrics import MeanIoU
# import matplotlib as mpl
from pathlib import Path
import tensorflow as tf
from collections import Counter
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pathlib import Path
from functools import partial
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import os
import albumentations as A
import random
import cv2
import argparse
from fastcore.basics import patch
from datetime import datetime
from fastcore.all import *
# from fastai.vision.all import *
from dataclasses import dataclass, field
import tensorflow_addons as tfa
from typing import Union, List, Tuple, Optional, Callable, Dict, Any
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from tqdm.notebook import tqdm
import cv2
from fastcore.all import *
# from ai_vision_tool.model_creation import *
import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


# os.environ['SSL_CERT_FILE']="/etc/pki/tls/certs/ca-bundle.crt"
# os.environ['MLFLOW_IGNORE_TLS'] = 'true'
# os.environ['MLFLOW_TRACKING_URI'] = 'https://mlflow-tracking-aihps.eu-de-5.icp.infineon.com/'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3mucceph02.infineon.com'
# os.environ['S3_BUCKET_PATH'] = "s3://rddl-mlwifi/mlflow"
# os.environ['AWS_ACCESS_KEY_ID'] =  "D0MWI7VUDQ1G3X8K00A1"
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'QTqAjHDBOCE5I1oYlB3zixXRCIE0yimCmH2ePTAH'
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'BLRmlflowaihps'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'wsa@AML@03091994'


################  mlfow part #########################
# import mlflow
# import mlflow.tensorflow
# import logging
#######################################################
# mlflow.tensorflow.autolog(every_n_iter=1, log_models=True, log_datasets=False)
# logging.getLogger('mlflow').setLevel(logging.INFO)
################  mlfow environment variable part #########################
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

def pooling(
        inputs,
        max_pool_only=False,
        both=True,
        pool_size=2):
    if both:
        p1 = tf.keras.layers.MaxPooling2D(
            (pool_size, pool_size))(inputs)
        p2 = tf.keras.layers.AvgPool2D(
            pool_size=(pool_size, pool_size))(inputs)
        return tf.keras.layers.concatenate([p1, p2])
    elif max_pool_only:
        return tf.keras.layers.MaxPooling2D(
            (pool_size, pool_size))(inputs)
    else:
        return tf.keras.layers.AvgPool2D(
            pool_size=(pool_size, pool_size))(inputs)
# | export


def conv_block(inputs, filter_no, kernel_size, batch_nm=True, dropout=True, drp_rt=0.1):
    c1 = tf.keras.layers.Conv2D(
        filter_no,
        (kernel_size, kernel_size),

        kernel_initializer='he_normal',
        padding='same',
        activation=None)(inputs)
    if batch_nm:
        c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    if dropout:
        c1 = tf.keras.layers.Dropout(drp_rt)(c1)
    return c1
# | export


def encoder(input_size):

    inputs = tf.keras.layers.Input(input_size)
    s = inputs
    c1 = conv_block(
        inputs=s,
        filter_no=16,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)

    c1 = conv_block(
        inputs=c1,
        filter_no=16,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0.001)  # NQA

    p1 = pooling(inputs=c1, both=True, pool_size=2)

    c2 = conv_block(
        inputs=p1,
        filter_no=32,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)

    c2 = conv_block(
        inputs=c2,
        filter_no=32,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0.0001  # will not be used
    )

    p2 = pooling(inputs=c2, both=True, pool_size=2)

    c3 = conv_block(
        inputs=p2,
        filter_no=64,
        kernel_size=3,

        dropout=True,
        drp_rt=0.2)

    c3 = conv_block(
        inputs=c3,
        filter_no=64,
        kernel_size=3,
        dropout=False,
        drp_rt=0.001  # will not be used
    )
    p3 = pooling(inputs=c3, both=True, pool_size=2)

    c4 = conv_block(
        inputs=p3,
        filter_no=128,
        batch_nm=True,
        kernel_size=3,
        dropout=True,
        drp_rt=0.2
    )
    c4 = conv_block(
        inputs=c4,
        filter_no=128,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0.001  # will not be used
    )

    p4 = pooling(both=True, inputs=c4, pool_size=2)

    c5 = conv_block(
        inputs=p4,
        filter_no=256,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.3)
    c5 = conv_block(
        inputs=c5,
        filter_no=256,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0.01  # will not be used
    )
    return s, c1, c2, c3, c4, c5


def encoder_decoder(
        input_size,
        n_classes=2):

    s, c1, c2, c3, c4, c5 = encoder(input_size)

    c9 = decoder_block(c1, c2, c3, c4, c5)

    outputs = tf.keras.layers.Conv2D(
        n_classes, (1, 1), activation='sigmoid')(c9)
    return tf.keras.models.Model(inputs=s, outputs=outputs)

# | export


def decoder_block(c1, c2, c3, c4, c5):

    # one concat block
    u6 = tf.keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    if u6.shape != c4.shape:
        c4_ = c4[:, :, :152, :]
    else:
        c4_ = c4
    u6 = tf.keras.layers.concatenate([u6, c4_])

    c6 = conv_block(
        inputs=u6,
        filter_no=128,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)

    c6 = conv_block(
        inputs=c6,
        filter_no=128,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0.001,  # will not be used
    )

    # second concat block
    u7 = tf.keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c6)
    if u7.shape != c3.shape:
        paddings = tf.constant([[0, 0], [0, 0], [0, 2], [0, 0]])
        u7_padded = tf.pad(u7, paddings)
    else:
        u7_padded = u7

    u7 = tf.keras.layers.concatenate([u7_padded, c3])
    c7 = conv_block(
        inputs=u7,
        filter_no=64,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)
    c7 = conv_block(
        inputs=c7,
        filter_no=64,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0)

    # third concat layer
    u8 = tf.keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = conv_block(
        inputs=u8,
        filter_no=32,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)
    c8 = conv_block(
        inputs=c8,
        filter_no=32,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0)
    # 4th concat
    u9 = tf.keras.layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = conv_block(
        inputs=u9,
        filter_no=16,
        kernel_size=3,
        batch_nm=True,
        dropout=True,
        drp_rt=0.2)
    c9 = conv_block(
        inputs=c9,
        filter_no=16,
        kernel_size=3,
        batch_nm=True,
        dropout=False,
        drp_rt=0)
    return c9
# | export


def normalize(
    image,  # :Union[np.ndarray, tf.Tensor],
        min=0):
    def _normalize(im):
        img = tf.cast(im, tf.float32)
        return img / 255.0

    if min == 0:
        return _normalize(image)
    else:
        return (_normalize(image) * 2.0) - 1.0


def read_normalize_and_resize(
    im_file,
    cfg,
):
    im = tf.io.read_file(im_file)
    im = tf.image.decode_png(im, channels=cfg['channel_number'])

    im = tf.cast(tf.image.resize(im,
                                 [cfg['IMAGE_HEIGHT'], cfg['IMAGE_WIDTH']],
                                 method=tf.image.ResizeMethod.BILINEAR, antialias=False), tf.float32)
    im = normalize(im, cfg['min'])

    return im


def read_and_binarize_mask(
    mask_file,
    cfg,
):
    mask = tf.io.read_file(mask_file)
    mask = tf.image.decode_png(
        mask,
        channels=cfg['channel_number'])
    mask = tf.image.resize(
        mask,
        (cfg['IMAGE_HEIGHT'], cfg['IMAGE_WIDTH']),
        antialias=True
    )
    mask = tf.cast(mask > 127, tf.float32)

    return mask


def process_image_and_mask(
        image_path,
        mask_path,
        cfg):
    image = read_normalize_and_resize(
        image_path,
        cfg)
    mask = read_and_binarize_mask(
        mask_path,
        cfg)
    return image, mask


def process_(image_path, mask_path, cfg):
    image, mask = process_image_and_mask(
        image_path,
        mask_path,
        cfg
    )
    return image, mask


def augmentation_(
    im_height: int,
    im_width: int,
    image: tf.Tensor,
    mask: tf.Tensor,
):

    scale = (0.8, 1.0)
    ratio = (1.0, 1.0)
    aug = A.Compose([
                    A.OneOf([
                        A.ShiftScaleRotate(
                            shift_limit=0.04,
                            scale_limit=0.04,
                            rotate_limit=12,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=0.8),
                        A.ShiftScaleRotate(
                            shift_limit=0.14,
                            scale_limit=0.04,
                            rotate_limit=20,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=0.1)
                    ], p=0.8),
                    A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    # A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    # A.RandomContrast(p=0.5, limit=(-0.5, 0.5)),
                    A.ColorJitter(brightness=0.1, p=0.02),
                    # A.Salt(p=0.2, per_channel=True),
                    # A.GaussNoise(var_limit=(10, 50), mean=0, p=0.5),
                    A.GaussNoise(var_limit=(0.01, 0.05), mean=0, p=0.5),
                    # A.GaussianBlur(blur_limit=(3,7), p=0.5),
                    # A.MultiplicativeNoise(multiplier=(0.05, 0.19), per_channel=False, p=0.5),
                    ])

    aug_data = aug(image=image, mask=mask)
    image, mask = aug_data['image'], aug_data['mask']
    return image, mask


class MiniumPixelPenaltlyLoss(tf.keras.losses.Loss):

    def __init__(
            self,
            min_foreground_pixels=40,
            **kwargs):
        super().__init__(**kwargs)
        self.smooth = 1.0e-5
        self.crop_entropy_loss = focal_tversky_loss_r
        self.min_forground_pixels = min_foreground_pixels
        # self.weight = weight if not None else [1.0, 10.0]

    def call(self, y_true, y_pred):
        tversky_focal_loss = focal_tversky_loss_r(y_true, y_pred)
        # bce_loss = tf.keras.losses.BinaryCrossentropy(
        # weight=self.weight
        # )(y_true, y_pred)
        forground_pixels = tf.reduce_sum(y_true)
        pixel_penalty = tf.math.maximum(
            self.min_forground_pixels - forground_pixels,
            0
        )
        total_loss = tversky_focal_loss + pixel_penalty
        return total_loss


class BasnetLoss(tf.keras.losses.Loss):
    """BASNet hybrid loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.smooth = 1.0e-9

        # Binary Cross Entropy loss.
        self.cross_entropy_loss = focal_tversky_loss_r
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        #  Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return K.mean(
            (intersection + self.smooth) / (union + self.smooth), axis=0
        )

    def call(self, y_true, y_pred):
        # cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred)
        cross_entropy_loss = focal_tversky_loss_r(y_true, y_pred)

        ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
        ssim_loss = K.mean(1 - ssim_value + self.smooth, axis=0)

        iou_value = self.iou_value(y_true, y_pred)
        iou_loss = 1 - iou_value

        # Add all three losses.
        return cross_entropy_loss + ssim_loss + iou_loss


def augment_data(image, mask, im_height, im_width, config):

    image_shape = (im_height, im_width, 1)
    aug_img, aug_mask = tf.numpy_function(
        func=augmentation_,
        inp=[
            im_height,
            im_width,
            image,
            mask], Tout=(tf.float32, tf.float32))
    aug_img.set_shape(image_shape)
    aug_mask.set_shape(image_shape)
    aug_img = tf.image.resize(
        aug_img, [config['IMAGE_HEIGHT'], config['IMAGE_WIDTH']])
    aug_mask = tf.image.resize(
        aug_mask, [config['IMAGE_HEIGHT'], config['IMAGE_WIDTH']])
    aug_mask = tf.cast(aug_mask > 0.5, tf.float32)
    return aug_img, aug_mask


def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.8
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss_r(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 2
    return K.pow((1-pt_1), gamma)


def get_m_name(fn, pat=re.compile(r'(IN|In|OUT|Out).(\d+)')):
    if pat.search(fn) is not None:
        return pat.search(Path(fn).name).group(2)
    else:
        return "undefined"


def model_train(
        model_save_path,
        model_experiment_name,
        model, epochs, train_ds,
        test_ds,
        config):
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="epoch",
    )
    # MODEL_EXPERIMENT_NAME='easy_pin_detection'
    # mlflow.set_experiment(MODEL_EXPERIMENT_NAME)
    # with mlflow.start_run():
    WD_RATIO = 0.05
    LR_START = 0.0001
    LR_MAX = 2e-4
    LR_MIN = 2e-5
    LR_RAMPUP_EPOCHS = 8
    LR_SUSTAIN_EPOCHS = 5
    EPOCHS = epochs

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
            decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5,
        clipnorm=1.0)
    model.compile(
        # loss=focal_tversky_loss_r,
        loss=BasnetLoss(),
        # loss=MiniumPixelPenaltlyLoss(),
        # loss =tf.keras.losses.BinaryFocalCrossentropy(),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.BinaryIoU(name='foreground',
                                            target_class_ids=[1],
                                            threshold=0.5)]

    )
    current_datetime = datetime.now()
    time = current_datetime.strftime("%H_%M_%S")

    Path(model_save_path).mkdir(exist_ok=True, parents=True)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_save_path}/time_{time}_val_frGrnd' +
        '{val_foreground:.4f}_epoch_{epoch}.h5',
        monitor='val_foreground', verbose=1, save_best_only=False, save_freq='epoch')
    callbacks = [
        lr_callback,
        # WeightDecayCallback(),
        model_checkpoint]

    his = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs if epochs else config['EPOCHS'],
        steps_per_epoch=config['steps_per_epoch'],
        validation_steps=config['validation_steps'],
        callbacks=callbacks
    )
    # Log the parameters
    # mlflow.log_params(
    # {
    # "epochs": epochs,
    # "foreground": foreground,
    # "val_foreground": val_ground,
    # }
    # )
    # Log the final metrics
    # mlflow.log_metrics(
    # {
    # "final_train_loss": train_loss.numpy(),
    # "final_test_loss": val_loss.numpy(),
    # }
    # )
    # )
    tf.keras.models.save_model(model, 'last.h5')
    return model, his


###############################################################################################################################
def create_datset_from_folder(
    config,
    training_folder,
    test_folder,
    training=True
):
    "create dataset from a folder"
    def get_data(config, training_folder, test_folder, training=True):

        train_images = sorted([str(i)for i in Path(
            training_folder/'images').ls(file_exts='.png')])
        train_im_name = [Path(i).name for i in train_images]
        train_labels = sorted([str(i)for i in Path(
            training_folder/'masks').ls(file_exts='.png') if Path(i).name in train_im_name])

        test_images = sorted([str(i)for i in Path(
            test_folder/'images').ls(file_exts='.png')])
        test_im_name = [Path(i).name for i in test_images]
        test_labels = sorted([str(i)for i in Path(
            test_folder/'masks').ls(file_exts='.png') if Path(i).name in test_im_name])
        if training:
            return train_images, train_labels
        else:
            return test_images, test_labels
    images, masks = get_data(config, training=training)

    dataset = tf.data.Dataset.from_tensor_slices(
                                                (images, masks)
    )
    # if training:
    # dataset = dataset.map(lambda x, y: (x, y, tf.py_function(repeat_image, [x], [tf.int64])))
    # dataset = dataset.flat_map(lambda x, y, n: tf.data.Dataset.from_tensors((x, y)).repeat(n[0]))

    dataset = dataset.map(
        partial(process_, cfg=config),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if training:
        dataset = dataset.map(partial(
            augment_data,
            im_height=config['IMAGE_HEIGHT'],
            im_width=config['IMAGE_WIDTH'], config=config),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(buffer_size=2)
    dataset = dataset.repeat()

    print(f" number of batches found {'#'*10}")
    print(f'config[bs] =  {config["bs"]}]')
    dataset = dataset.batch(config['bs'])
    # if training:
    # dataset = dataset.map(lambda x, y: mixup(x, y, alpha=0.2))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


###############################################################################################################################


def create_dataset(config, training=True):
    def get_data(config, training):

        mask_names = [i.name for i in Path(
            config['label_path']).ls(file_exts='.png')]
        print(f"{'#'*10}")
        print(f' number of masks found {len(mask_names)}')
        images = sorted([str(i)for i in Path(config['image_path']).ls(
            file_exts='.png') if Path(i).name in mask_names])
        image_names = [Path(i).name for i in images]
        masks = sorted([str(i)for i in Path(config['label_path']).ls(
            file_exts='.png') if Path(i).name in image_names])
        print(f' number of images found {len(masks)}')

        train_images, test_images, \
            train_labels, test_labels = train_test_split(
                images,
                masks,
                test_size=config['test_size'],
                random_state=42)

        if training:

            return train_images, train_labels
        else:
            return test_images, test_labels
    images, masks = get_data(config, training=training)

    dataset = tf.data.Dataset.from_tensor_slices(
                                                (images, masks)
    )
    # if training:
    # dataset = dataset.map(lambda x, y: (x, y, tf.py_function(repeat_image, [x], [tf.int64])))
    # dataset = dataset.flat_map(lambda x, y, n: tf.data.Dataset.from_tensors((x, y)).repeat(n[0]))

    dataset = dataset.map(
        partial(process_, cfg=config),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if training:
        dataset = dataset.map(partial(
            augment_data,
            im_height=config['IMAGE_HEIGHT'],
            im_width=config['IMAGE_WIDTH'], config=config),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(buffer_size=2)
    dataset = dataset.repeat()

    print(f" number of batches found {'#'*10}")
    print(f'config[bs] =  {config["bs"]}]')
    dataset = dataset.batch(config['bs'])
    # if training:
    # dataset = dataset.map(lambda x, y: mixup(x, y, alpha=0.2))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def process_(image_path, mask_path, cfg):
    image, mask = process_image_and_mask(
        image_path,
        mask_path,
        cfg
    )
    return image, mask


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-imn',
        '--initial_model_name',
        type=str,
        default='progressive_resizing_2023-07-26 12_33_160.8407_epoch_15.h5',
        help='current loaded model name'
    )
    parser.add_argument(
        '-en',
        '--epoch_number',
        type=int,
        default=200,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '-imp',
        '--initial_model_path',
        type=str,
        default='/home/ai_easypid/data/projects/easy_pin_detection/models1024_1224/gamma2',
        help='current loaded model path'
    )

    parser.add_argument(
        '-mp',
        '--model_path',
        type=str,
        default='/home/ai_easypid/data/projects/easy_pin_detection/models1024_1224/gamma2',
        help='model path'
    )

    parser.add_argument(
        '-ptrn',
        '--pretrained',
        action='store_true',
        help='whether to use pretrained or scratch model'

    )

    parser.add_argument(
        '-mn', '--model_name', type=str, default='higer_learning', help='experiment description')
    parser.add_argument(
        '-ip', '--image_path', type=str, default='/home/ai_easypid/data/projects/easy_pin_detection/ali1_new_download/images', help='name of image path')
    parser.add_argument(
        '-p', '--mask_path', type=str, default='/home/ai_easypid/data/projects/easy_pin_detection/ali1_new_download/masks', help='name of mask path')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=3, help='batch size')
    parser.add_argument(
        '-ih', '--image_height', type=int, default=256, help='height of image resolution ')
    parser.add_argument(
        '-iw', '--image_width', type=int, default=256, help='width of image resolution ')
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    config = {}
    config['image_path'] = args.image_path
    config['label_path'] = args.mask_path
    config['class_names'] = ['Pin']
    config['test_size'] = 0.2
    config['train_count'] = int(
        len(Path(config['image_path']).ls()) * (1 - config['test_size']))
    print(f"{'#'*50} train_count == {config['train_count']}")
    # 5055 - config['train_count']
    config['test_count'] = int(len(Path(config['image_path']).ls())*0.2)
    config['num_classes'] = 1
    config['IMAGE_HEIGHT'] = int(args.image_height)
    config['IMAGE_WIDTH'] = int(args.image_width)
    config['channel_number'] = 1
    config['one_channel'] = True
    config['im_size'] = (config['IMAGE_HEIGHT'],
                         config['IMAGE_WIDTH'], config['channel_number'])
    # Minimum value of tensor [whether normalize 0,1 or -1 to 1
    config['min'] = 0
    config['EPOCHS'] = int(args.epoch_number)

    # tf.keras.mixed_precision.set_global_policy(
    # "mixed_float16"
    # )
    config['bs'] = int(args.batch_size)
    print(f"{'#'*10}")
    print(f" training count == {config['train_count']}")
    print(f"{'#'*10}")
    config['steps_per_epoch'] = config['train_count']//config['bs']
    config['validation_steps'] = config['test_count']//config['bs']
    print(f"{'#'*10}")
    print(f"batch size == {config['bs']}")
    print(f"Steps per epoch size == {config['steps_per_epoch']}")

    train_ds = create_dataset(config=config, training=True)
    test_ds = create_dataset(config, training=False)
    model_path = args.model_path
    model_name = args.model_name

    optimizer = tfa.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5,
        clipnorm=1.0)

    IMAGE_HEIGHT = config['IMAGE_HEIGHT']
    IMAGE_WIDTH = config['IMAGE_WIDTH']
    input_size = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    if args.pretrained:
        loaded_model = tf.keras.models.load_model(
            f"{args.initial_model_path}/{args.initial_model_name}",
            custom_objects={
                'optimizer': optimizer,
                'BasnetLoss': BasnetLoss,
                'focal_tversky_loss_r': focal_tversky_loss_r}
        )
        # print(loaded_model.summary())
        print('pretrained model is used')
    else:

        loaded_model = encoder_decoder(input_size=input_size, n_classes=1)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model, his = model_train(
        model_save_path=args.model_path,
        model_experiment_name=args.model_name,
        model=loaded_model,
        epochs=config['EPOCHS'],
        train_ds=train_ds,
        test_ds=test_ds,
        config=config)


if __name__ == '__main__':
    main()
