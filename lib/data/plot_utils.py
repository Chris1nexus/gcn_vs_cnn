from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
from IPython.display import display, HTML
from matplotlib.animation import PillowWriter
import imageio

from .augmentation_utils import helper_elastic_transform

def ipyDisplay(torchTensor): 
  '''
  plot torch tensor passing through the pil image API
  Args:
    -torchTensor (torch.Tensor) image to plot

  '''
  topil = transforms.ToPILImage()
  pil_img = topil(torchTensor)
  display(pil_img)



def plot_segmentation_history(segmentation_progress, gif_filepath, figsize=(8,8)):
    '''
    Helper function used by segmentation procedures.
    Args:
      -segmentation_progress (  list of (epoch_id, (idx, pred_grid, true_grid )) ) pred_grid, true_grid are numpy arrays composed by means of torchvision.utils makegrid function
      -gif_filepath (str) path where to save the result
      -figsize (tuple (int,int) ) image shape
    
    '''

    pred_history_for_idx = {}
    mask_history_for_idx = {}
    for epoch_idx, items in segmentation_progress:
        for idx, pred_grid, mask_grid in items:
          pred_progress_list = pred_history_for_idx.get(idx, None)
          mask_progress_list = mask_history_for_idx.get(idx, None) 

          if pred_progress_list is None:
            pred_progress_list = []
            pred_history_for_idx[idx] = pred_progress_list
          if mask_progress_list is None:
            mask_progress_list = []
            mask_history_for_idx[idx] = mask_progress_list

          pred_progress_list.append(pred_grid)
          mask_progress_list.append(mask_grid)

    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    pred_progress_list = pred_history_for_idx[idx]
    mask_progress_list = mask_history_for_idx[idx]

    ims = [[plt.imshow( pred  , animated=True), plt.imshow(mask, animated=True, cmap='jet',alpha=0.2) ] for pred,mask in zip(pred_progress_list,mask_progress_list) ]
    #ims = [[plt.imshow( pred_progress_list  , animated=True) ] for i in pred_progress_list ]
    ims = ims + [ims[-1] for _ in range(5)]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=100000, blit=True)
    writer = PillowWriter(fps=2)  
    ani.save(gif_filepath, writer=writer)  
    HTML(ani.to_jshtml())

"""# plot elastic deformations"""

def plot_elastic_transform(img, seg_gt, alpha=(1,10), sigma=(0.08, 0.5), alpha_affine=(0.01, 0.2), random_state=None):
    '''
    Args:
        -img  (np.array) image to deform
        -seg_gt (np.array) mask to deform
        -alpha=(1,10)  (tuple (int,int))
        -sigma=(0.08, 0.5) (tuple (float,float))
        -alpha_affine=(0.01, 0.2) (tuple (float,float))
        random_state=None  (int)
    '''
    res_img, res_seg_gt = helper_elastic_transform(img, seg_gt, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, random_state=random_state)
    fig = plt.figure(figsize=(10,8))
    raw = fig.add_subplot(221)
    raw.imshow(img)

    seg_gt_transformed = fig.add_subplot(222)
    seg_gt_transformed.imshow(seg_gt, cmap='gray')

    raw_transformed = fig.add_subplot(223)
    raw_transformed.imshow(res_img)

    seg_gt_transformed = fig.add_subplot(224)
    seg_gt_transformed.imshow(res_seg_gt, cmap='gray')


def grid_plot_elastic_transform(image, segmentation, alpha=(1,4), sigma=(0.08, 0.13), alpha_affine=(0.07, 0.13), random_state=None):
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

  alpha_multiplier = random.randint(*alpha) 
  sigma_multiplier = random.uniform(*sigma)
  alpha_affine_multiplier = random.uniform(*alpha_affine)

  im = (np.mean(img,axis=2)*255).astype(np.uint8)
  im_mask = (segmentation*255).astype(np.uint8)
  draw_grid(im, 50)
  draw_grid(im_mask, 50)

  im_merge_plot = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
  im_merge_t_plot = elastic_transform(im_merge_plot, im_merge_plot.shape[1] * alpha_multiplier, im_merge_plot.shape[1] * sigma_multiplier, im_merge_plot.shape[1] * alpha_affine_multiplier)

  im_t_plot = im_merge_t_plot[...,0]
  im_mask_t_plot = im_merge_t_plot[...,-1]

  #im_t_plot = #(np.mean(im_t,axis=2)*255).astype(np.uint8)
  #im_mask_t_plot = #(im_mask_t*255).astype(np.uint8)
  
  # Display result
  plt.figure(figsize = (16,14))
  plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t_plot, im_mask_t_plot]], cmap='gray')

  
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


