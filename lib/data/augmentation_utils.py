import torchvision.transforms.functional as transformFuncs
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import random
import torch

from torchvision import transforms
# the range of values for alpha, sigma and alpha affine 
# have to be CAREFULLY ASSESSED in order to obtain MEANINGFUL results from the deformation
def helper_elastic_transform(image, segmentation, alpha=(1,10), sigma=(0.08, 0.5), alpha_affine=(0.01, 0.2), random_state=None):
  '''
  Apply the same random elastic deformations to the given image-mask pair
  Values of the parameters of the deformations that preserve the image information are:
          -alpha in [1, 5]
          -sigma in [0.08,0.13]
          -alpha_affine in [0.07, 0.13]
  Values outside these ranges can result in too much distortion in the resulting augmented image 
  Args: 
          -image (np.array dtype=np.uint8)
          -mask (np.array dtype=np.uint8)
          -alpha in [1, 5] (tuple)
          -sigma in [0.08,0.13]  (tuple)
          -alpha_affine in [0.07, 0.13] (tuple)
          -random_state (default None)
  Returns:
          -transformed image (np.array dtype=np.uint8)
          -transformed mask (np.array dtype=np.uint8)
  '''
  # passed as np.uint8 usually, as images come from cv2 imread
  # but for the sake of generality, initial type is stored here,
  # in order to convert the images produced by the elastic transformation (which are of float type)
  # to their original type
  # this is also needed as pytorch .ToTensor only normalizes in the range 0,1 image
  image_dtype = image.dtype
  segmentation_dtype = segmentation.dtype

  n_image_channels = image.shape[-1]
  
  # image can be grayscale or RGB but segmentation mask is grayscale
  im_merge = np.concatenate((image, segmentation[...,None]), axis=2)

  alpha_multiplier = random.randint(*alpha) 
  sigma_multiplier = random.uniform(*sigma)
  alpha_affine_multiplier = random.uniform(*alpha_affine)
  im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * alpha_multiplier, im_merge.shape[1] * sigma_multiplier, im_merge.shape[1] * alpha_affine_multiplier)

  im_t = im_merge_t[...,:n_image_channels]
  #print(im_merge_t.shape)
  #print(im_merge_t[...,-1].shape)
  im_mask_t = im_merge_t[...,-1]
  
  #after operations, data becomes of type float 
  # so it has to be restored to its original type
  return im_t.astype(image_dtype), im_mask_t.astype(segmentation_dtype)



def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def paired_rand_rot(image, segmentation, prob=0.5):
  '''
  Apply the same random rotation on the pair image-mask
  Images are rotated of [0, 90, 180, 270] degrees
  with probability p 
  Args:
      -image (torch.Tensor)
      -mask (torch.Tensor)
      -p (probability of the rotation) (default is 0.5)
  Returns:
      -transformed image (torch.Tensor)
      -transformed mask (torch.Tensor)
  '''
  if random.random() > (1. - prob):
    #angle = random.randint(-angle_range, angle_range)
    angles = [0, 90, 180, 270]
    k = random.randint(0,3)
    angle = angles[k]
    image = transformFuncs.rotate(image, angle)
    
    segmentation = transformFuncs.rotate(segmentation, angle)
  return image, segmentation

def paired_gauss_blur(image, segmentation, prob=0.5, kernel_size=3, sigma=(0.1, 2.0)):
  '''
  Apply the same random gaussian blur on the pair image-mask
  parameters:
      -image  (torch.Tensor)
      -mask   (torch.Tensor)
      -kernel_size (default is 3) (int)
      -sigma (default range [0.1,2] ) (tuple (min,max))
  with probability p 
  '''
  if random.random() > 1. -prob:
    std = random.uniform(*sigma)
    image = transformFuncs.gaussian_blur(image, kernel_size=kernel_size, sigma=std)
    segmentation = transformFuncs.gaussian_blur(segmentation, kernel_size=kernel_size, sigma=std)
  return image, segmentation

def paired_rand_crop(image, segmentation, prob=0.5, original_kept_crop_percent=(0.75,1.0)):
  '''
  Apply the same random crop and resize to the image and the mask
  parameters:
      -image
      -mask
      -prob : probability of the transformation (default is 0.5)
      -original_kept_crop_percent: percent of the original image that is kept and then resized to original shape (default range is [0.75, 1.0])
  '''
  if random.random() > 1. - prob:
    if torch.is_tensor(image):
      dim = (1,2)
      h,w = image.shape[dim[0]:dim[1]+1]
    else:
      dim = (0,1)
      h,w = image.shape[dim[0]:dim[1]+1]
    
    keep_percent = random.uniform(*original_kept_crop_percent)
    removed_percent = 1. - keep_percent
    x_coord_pixel = random.randint(0, int(removed_percent*w) )
    y_coord_pixel = random.randint(0, int(removed_percent*h) )
    new_width = w - x_coord_pixel
    new_height = h - y_coord_pixel

    #image = transformFuncs.crop(image, y_coord_pixel, x_coord_pixel, new_height, new_width)
    #print(image.shape)
    #image = transformFuncs.resize(image, (h,w))#, interpolation)
    image = transformFuncs.resized_crop(image,  top=y_coord_pixel, left=x_coord_pixel, height=new_height, width=new_width, size=(h,w))    
    segmentation = transformFuncs.resized_crop(segmentation,  top=y_coord_pixel, left=x_coord_pixel, height=new_height, width=new_width, size=[h,w])
  return image, segmentation 


class AugmentTransform(object):
  '''
  Helper class that performs data augmentation according to the keyword arguments given when the object is allocated 
  Args (can be set to None to skip the corresponding transformation):
      -resize_crop   (default to {'prob':1.0, 'original_kept_crop_percent':(0.75,0.9)})  (dictionary)
      -rotate   (default to {'prob':1.0})  (dictionary)
      -gauss_blur   (default{'prob':1.0,'kernel_size':3, 'sigma':(0.1, 2.0)}) (dictionary)
      -elastic_deform (default {'alpha':(1,10), 'sigma':(0.08, 0.5), 'alpha_affine':(0.01, 0.2), 'random_state':None}) (dictionary)
  '''

  def __init__(self,
                  resized_crop= {'prob':1.0, 'original_kept_crop_percent':(0.75,0.9)},
                  rotate = {'prob':1.0},
                  gauss_blur={'prob':1.0,'kernel_size':3, 'sigma':(0.1, 2.0)},
                  elastic_deform={'alpha':(1,10), 'sigma':(0.08, 0.5), 'alpha_affine':(0.01, 0.2), 'random_state':None},
               ):
    self.totensorTransform = transforms.ToTensor()
    self.resized_crop = resized_crop
    self.rotate = rotate
    self.gauss_blur= gauss_blur
    self.elastic_deform = elastic_deform
  def transform(self, image, segmentation):
    '''
    data augmentation pipeline that applies the same transformation to the paired image-mask sample
    elastic deformation is done with numpy arrays, which are then transformed to torch tensors for the consecutive transformations.
    The pytorch ToTensor() transform automatically swaps the axes from channel last to channel first and if the image is of type uint8, it also rescales its values\n so that
    they are in the range [0...1] (by dividing by 255). In any case, wheter the elastic deformation is applied or not, images are uint8, so the ToTensor transformation is
    correctly applied.  
    Args:
        -image (np.array uint8) HWC (height,width,channel)
        -mask (np.array uint8) HWC (height,width,channel)
    Returns:
        -image (torch.Tensor) CHW (channel, height,width)
        -mask (torch.Tensor)  CHW (channel, height,width)

    '''
    if self.elastic_deform is not None:
      
      image, segmentation = helper_elastic_transform(image, segmentation, **self.elastic_deform)

    image = self.totensorTransform(image)
    #print("tr: ", segmentation.shape)
    segmentation = self.totensorTransform(segmentation)
    #print("tr2: ", segmentation.shape)
    if self.resized_crop is not None:
      image, segmentation = paired_rand_crop(image, segmentation, **self.resized_crop)
    if self.rotate  is not None:
      image, segmentation = paired_rand_rot(image, segmentation, **self.rotate)
    if self.gauss_blur is not None:
      image, segmentation = paired_gauss_blur(image, segmentation, **self.gauss_blur)

  
    return image,segmentation
