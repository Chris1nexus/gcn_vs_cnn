import cv2
import numpy as np

from PIL import Image
import cv2
import torch
import numpy as np

def pil_loader(path, resize_dim, color_mapping):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert(color_mapping).resize((resize_dim,resize_dim))
        return img #'RGB')

def read_image(path, resize_dim, color_mapping, img_format='RGB'):
    img_item = cv2.imread(path, color_mapping)
    resized_img_item = cv2.resize(img_item, (resize_dim,resize_dim))
    if color_mapping == cv2.IMREAD_COLOR and img_format == 'RGB':
      resized_img_item = cv2.cvtColor(resized_img_item, cv2.COLOR_BGR2RGB)
    return resized_img_item


def np_recompose_tensor(crops,  image_shape_getter_fn, n_channels_getter_fn,
                        paste_fn, init_output_fn ):
  coords, cropped_tensor = crops[0]
  crop_dim = image_shape_getter_fn(cropped_tensor)
  #num_crops_per_side = original_side_len//crop_dim[0]  #original_shape[1]//crop_dim[0]
  
  crop_len = crop_dim[0] #crops[0][1].shape[1]

  max_idx_crop = 0
  for coords, cropped_tensor in crops:
    i,j = coords
    max_idx_crop = max(max(max_idx_crop, i),j)
  num_crops_per_side = max_idx_crop + 1
  original_side_len = num_crops_per_side * crop_dim[0]

  n_original_channels = n_channels_getter_fn(cropped_tensor)
  destination_tensor = init_output_fn(original_side_len,n_original_channels)  

  crop_idx = 0
  for i in range(num_crops_per_side):
    for j in range(num_crops_per_side):
        row_start = i*crop_len
        row_end = (i+1)*crop_len
        col_start = j*crop_len
        col_end = (j+1)*crop_len

        coords, curr_crop = crops[crop_idx]
        row_indices = slice(row_start,row_end)
        col_indices = slice(col_start,col_end)
        paste_fn(destination_tensor,row_indices, col_indices, curr_crop)
        #destination_tensor[:, row_start:row_end,  col_start:col_end ] = curr_crop

        crop_idx += 1
  return destination_tensor

def np_make_crops(tensor_img,image_shape_getter_fn,
                        crop_fn,split_side_in=4):
  crops = []

  img_shape = image_shape_getter_fn(tensor_img)
  # crop in squares (suppose tensor_img is square)
  crop_len = img_shape[0]//split_side_in

  for i in range(split_side_in):
    for j in range(split_side_in):
        row_start = i*crop_len
        row_end = (i+1)*crop_len
        col_start = j*crop_len
        col_end = (j+1)*crop_len

        #print(row_start,":", row_end)
        #print(col_start,":", col_end)
        row_indices = slice(row_start,row_end)
        col_indices = slice(col_start,col_end)
   
        tensor_crop = crop_fn(tensor_img,row_indices, col_indices)

        crops.append( ((i,j),tensor_crop)  )
  return crops
def img_paste_fn(destination_tensor,row_indices, col_indices, curr_crop):
              destination_tensor[row_indices,col_indices,:] = curr_crop
def seg_paste_fn(destination_tensor,row_indices, col_indices, curr_crop):
              destination_tensor[row_indices,col_indices] = curr_crop

              

def estimate_channel_statistics(image_dataset):
    """
    returns mean and variance of each color channel
    """
    mean_k = 0
    var_k = 0
    K = 0

    M = 0
    N = 0
    # dimensions of the image, along which mean and std are computed for the channels
    dim = (0,1)
    for (path_img, path_seg, img, seg_img, seg_gt), label in image_dataset:
        
        # channel last
        if torch.is_tensor(img):
          dim = (1,2)#img.shape[1:]
        else:
          dim = (0,1)#img.shape[:-1]
          
        P,Q =img.shape[dim[0]:dim[1]+1] #img.shape[0],img.shape[1]
        N = P*Q
        curr_mean = img.mean(dim)#axis=(0,1))

        old_stat_weight = M/(M+N)
        new_stat_weight = N/(M+N)
        mean_k_new = mean_k*old_stat_weight + curr_mean*new_stat_weight


        curr_var = img.var(dim)#axis=(0,1))
        var_k = old_stat_weight*var_k +new_stat_weight*curr_var + old_stat_weight*new_stat_weight*(mean_k-curr_mean)**2

        mean_k = mean_k_new
        M += N
    return mean_k, var_k



def is_on(image,i,j):
    return image[i][j] == 255

def remove_isolated(img):
    h, w = img.shape
    image = img.copy()
    
    for i in range(h):
        for j in range(w):
            isolated = True
            if image[i][j] == 255:
                if i + 1 < h:
                    if is_on(image,i+1,j):
                        isolated = False
                    if j + 1 < w and is_on(image,i+1,j+1):
                        isolated = False
                    if j - 1 >= 0 and is_on(image,i+1,j-1):
                        isolated = False
                if i - 1 >= 0:
                    if is_on(image,i-1,j):
                        isolated = False
                    if j + 1 < w and is_on(image,i-1,j+1):
                        isolated = False
                    if j - 1 >= 0 and is_on(image,i-1,j-1):
                        isolated = False
                if j + 1 < w and is_on(image,i ,j+1):
                    isolated = False

                if j - 1 >= 0 and is_on(image,i ,j-1):
                    isolated = False
                if isolated:
                    image[i][j] = 0
    return image




def binarize_to_numpy(segmentation_image, seg_color_mapping, img_format='RGB'):
  if seg_color_mapping == cv2.IMREAD_COLOR:
    if img_format == 'RGB':
      segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR)
    gray_seg = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)#segmentation_image.convert('L')#cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
  else:
    gray_seg = segmentation_image
    
  _, binarized = cv2.threshold(gray_seg,1,255,cv2.THRESH_BINARY)

  return binarized


def get_segmentation_mask(segmentation, seg_color_mapping, img_format):
  binarized_255_0 = binarize_to_numpy(segmentation, seg_color_mapping, img_format)
  
  binarized_255_0[binarized_255_0==255] = 1
  binarized_255_0 = binarized_255_0.astype(np.float)
  return binarized_255_0
