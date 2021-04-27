from torchvision import transforms

import matplotlib.animation as animation  
from IPython.display import display, HTML
from matplotlib.animation import PillowWriter
import imageio

from augmentation_utils import helper_elastic_transform

def ipyDisplay(torchTensor): 
  topil = transforms.ToPILImage()
  pil_img = topil(torchTensor)
  display(pil_img)



def plot_segmentation_history(segmentation_progress, gif_filepath, figsize=(8,8)):

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
