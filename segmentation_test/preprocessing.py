"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['IMAGE_HEIGHT', 'IMAGE_WIDTH', 'EPOCS', 'BATCH_SIZE', 'BUFFER_SIZE', 'class_names', 'train_count', 'test_count',
           'num_classes', 'steps_per_epoch', 'validation_steps', 'Preprocess', 'convert_np_and_uint8',
           'convert_one_channel', 'foo']

# %% ../nbs/00_core.ipynb 3
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import albumentations as A
import random
import cv2
from fastcore.basics import patch
from fastcore.all import *
#from fastai.vision.all import *
from dataclasses import dataclass, field

from typing import Union, List, Tuple, Optional, Callable, Dict, Any

# %% ../nbs/00_core.ipynb 5
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
EPOCS = 5
BATCH_SIZE = 8
BUFFER_SIZE = 30
class_names = ['Pin']
train_count = 107
test_count = 27
num_classes = len(class_names)
steps_per_epoch = train_count // BATCH_SIZE
validation_steps = test_count // BATCH_SIZE


# %% ../nbs/00_core.ipynb 8
@dataclass
class Preprocess:
    image_path:Union[Path, str]
    label_path:Union[Path, str]
    im_height:int = IMAGE_HEIGHT
    im_width:int = IMAGE_WIDTH
    bf_size:int = BUFFER_SIZE
    bs:int = BATCH_SIZE
    one_channel:bool=False
    test_size:float = 0.2
    img_ext:str = field(default_factory=str, init=False, repr=True)

    def __post_init__(self):
        self.img_ext = Path(self.image_path).ls()[0].suffix
        pat = f'*{self.img_ext}'
        self.images = [str(i) for i in Path(self.image_path).rglob(pat)]
        self.labels = [str(i) for i in Path(self.label_path).rglob(pat)]
        self.train_images, self.test_images, \
            self.train_labels, self.test_labels = train_test_split(
                         self.images,
                         self.labels, 
                         test_size=self.test_size,
                         random_state=42)


# %% ../nbs/00_core.ipynb 11
@patch_to(Preprocess)
def show_image(self,im_file):
    #image = self.from_file_to_image(im_file)
    plt.imshow(im_file)
    plt.axis('off')
    plt.show()

# %% ../nbs/00_core.ipynb 12
@patch_to(Preprocess)
def read_image(self,im_file, one_channel=False):
    if one_channel:
        im = tf.io.read_file(im_file)
        im = tf.image.decode_png(im, channels=1)
    else:
        im = tf.io.read_file(im_file)
        im = tf.image.decode_png(im, channels=3)
    return im

# %% ../nbs/00_core.ipynb 15
@patch_to(Preprocess)
def augmentation_(
        self,
        im_height:int,
        im_width:int,
        image:tf.Tensor,
        mask:tf.Tensor,
        ):
    aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(250, 250), height=im_height, width=im_width, p=0.5),
        A.PadIfNeeded(
                      #min_height=im_height,
                     # min_width=im_width,
                        p=0.5)
    ], p=1),    
    A.HorizontalFlip(p=0.5),              
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.8),
    #A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8)])
    aug_data = aug(image=image.numpy(), mask=mask.numpy())
    image, mask = aug_data['image'], aug_data['mask']
    #mask = tf.expand_dims(mask, axis=-1)
    return image, mask

# %% ../nbs/00_core.ipynb 16
@patch_to(Preprocess)
def show_aug(
    self,
    image,
    mask,
    original_image=None,
    original_mask=None
    ):
    
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[0, 0].axis('off')
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[1, 0].axis('off')
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[0, 1].axis('off')

        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        ax[1, 1].axis('off')
    f.tight_layout()

# %% ../nbs/00_core.ipynb 17
@patch_to(Preprocess)
def read_aug(
            self,
            im_file:str,
            lbl_file:str,
            one_channel:bool=False, 
            aug:bool=False):
    img = self.read_image(im_file=im_file, one_channel=one_channel)
    mask = self.read_image(im_file=lbl_file, one_channel=one_channel)
    if aug:
        return self.augmentation_(im_height=img.shape[1], im_width=img.shape[0], image=img, mask=mask)
    else:
        return img, mask

# %% ../nbs/00_core.ipynb 20
@ patch_to(Preprocess)
def normalize(
              self,
              image:Union[np.ndarray, tf.Tensor], 
              min=0):
    def _normalize(im):
        img = tf.cast(im, tf.float32)
        return img / 255.0

    if min == 0:
        return _normalize(image)
    else:
        return (_normalize(image) * 2.0) -1.0

# %% ../nbs/00_core.ipynb 23
@ patch_to(Preprocess)
def process_image_and_mask(
                       self,
                       im_file:str,
                       lbl_file:str,
                       norm:bool=True,
                       one_channel:bool=False,
                       aug_data:bool=True
                       ):
    image,mask = self.read_aug(im_file, lbl_file, one_channel=one_channel, aug=aug_data)
    image, mask = tf.image.resize(image, (self.im_height, self.im_width)), tf.image.resize(mask, (self.im_height, self.im_width))
    if norm:
        image = self.normalize(image)
        mask = self.normalize(mask)
    if one_channel:
        image = tf.reshape(image, (self.im_height, self.im_width, 1,))
        mask = tf.reshape(mask, (self.im_height, self.im_width, 1,))
    else:
        image = tf.reshape(image, (self.im_height, self.im_width, 3,))
        mask = tf.reshape(mask, (self.im_height, self.im_width, 3,))
    return image,mask

# %% ../nbs/00_core.ipynb 28
@ patch_to(Preprocess)
def process_data(
                 self,
                 image,
                 label,
                 norm:bool=True,
                 one_channel:bool=False,
                 aug_data:bool=True):
    #@tf.function
    aug_img, aug_lbl = tf.numpy_function(func=self.process_image_and_mask, inp=[image, label, norm, one_channel, aug_data], Tout=(tf.float32, tf.float32))
    return aug_img, aug_lbl

# %% ../nbs/00_core.ipynb 29
@patch_to(Preprocess)
def set_shapes(
              self,
              img, 
              label, 
              img_shape):
    img.set_shape(img_shape)
    label.set_shape(img_shape)
    return img, label

# %% ../nbs/00_core.ipynb 30
@patch_to(Preprocess)
def create_dataset(
                  self,
                  images, labels,
                  train:bool=True,
                  norm:bool=True,
                  aug:bool=True
                  ):
    _dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    _dataset = _dataset.map(partial(self.process_data, aug_data=aug, norm=norm), num_parallel_calls=tf.data.AUTOTUNE)
    
    if self.one_channel:
        _dataset = _dataset.map(partial(self.set_shapes, img_shape=(self.im_height, self.im_width, 1)), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        _dataset = _dataset.map(partial(self.set_shapes, img_shape=(self.im_height, self.im_width, 3)), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        return  _dataset\
                   .cache()\
                   .shuffle(
                            self.bf_size,
                            reshuffle_each_iteration=True)\
                   .batch(self.bs)\
                   .repeat()\
                   .prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        #_dataset = _dataset.map(self.process_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
        return  _dataset.batch(self.bs).repeat()


# %% ../nbs/00_core.ipynb 31
@patch_to(Preprocess)
def create_train_test_dataset(self):
    self.train_dataset = self.create_dataset(
                                            images=self.train_images, 
                                            labels=self.train_labels,
                                            norm=True,
                                            aug=True,
                                            train=True)
    self.test_dataset = self.create_dataset(
                                           images=self.test_images,
                                           labels=self.test_labels,
                                           norm=True,
                                           aug=False,
                                           train=False)
    return self.train_dataset, self.test_dataset

# %% ../nbs/00_core.ipynb 36
def convert_np_and_uint8(img:tf.Tensor)->Tuple[np.array, np.array]:
    "Convert img to np.array and uint8"

    if isinstance(img, tf.data.Dataset) or isinstance(img, tf.Tensor):
        m_scale1=img.numpy()
        m_scale255=(img * 255).numpy().astype(np.uint8)
    elif isinstance(img, np.ndarray):
            
        if img.dtype in [np.float16, np.float32, np.float64]:
            m_scale255 = (img * 255).astype(np.uint8)
            m_scale1 = img
            raise Exception("unknown dtype:", img.dtype)

    return m_scale1, m_scale255

# %% ../nbs/00_core.ipynb 37
def convert_one_channel(img:tf.Tensor):
    "Convert image to one channel"
    img = tf.image.rgb_to_grayscale(img)
    img = tf.reshape(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    return img

# %% ../nbs/00_core.ipynb 38
@patch_to(Preprocess)
def create_color_mask(
             self,
             mask:Union[np.array, tf.Tensor], 
             img:Union[np.array, tf.Tensor],
             threshold:float=0.5):
    "Creating color mask for segmentation"
    overlay_mask = np.ones((self.im_height, self.im_width, 3,), dtype=np.uint8)
    overlay_mask[:, :, 0] = img.reshape(*(self.im_height, self.im_width))
    overlay_mask[:, :, 1] = img.reshape(*(self.im_height, self.im_width))
    overlay_mask[:, :, 2] = img.reshape(*(self.im_height, self.im_width))
    match = mask.reshape(*(self.im_height, self.im_width)) > (threshold * 255)
    overlay_mask[match] = [0,255,0]
    return overlay_mask

# %% ../nbs/00_core.ipynb 39
@patch_to(Preprocess)
def display_np_batch(
                    self,
                    images:np.ndarray,
                    masks:np.ndarray,
                    threshold:float=0.5):
    "Displaying batch of images and masks"
    n_batch = images.shape[0]
    for i in range(n_batch):
        if images[i].shape[2] ==3:
            img = images[i][:,:,0]
            mask = masks[i][:,:,0]
        else:
            img = images[i].reshape(self.im_height, self.im_width )
            mask = masks[i].reshape(self.im_height, self.im_width )
        
        _, ax = plt.subplots(1, 3, figsize=(15, 10)) 
        ax[0].imshow(img, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('only image')
        clr_mask_ = self.create_color_mask(mask=mask, img=img, threshold=threshold)
        ax[1].imshow(mask)
        ax[1].axis('off')
        ax[1].set_title('only mask')
        ax[2].imshow(clr_mask_)
        ax[2].axis('off')
        ax[2].set_title('image with mask')
        plt.tight_layout()

# %% ../nbs/00_core.ipynb 40
@patch_to(Preprocess)
def display_ds(self,
            ds:tf.data.Dataset):

    images, masks = next(iter(ds))

    # convert to numpy and uint8 and getting scaled(255) image and mask
    images_np, images_255_np = convert_np_and_uint8(images)
    masks_np, masks_255_np = convert_np_and_uint8(masks)
    self.display_np_batch(
                          images=images_255_np,
                          masks=masks_255_np, 
                          threshold=0.5)

    

# %% ../nbs/00_core.ipynb 42
def foo(): pass
