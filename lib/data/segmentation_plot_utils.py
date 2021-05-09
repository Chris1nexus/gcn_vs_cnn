
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import torchvision.utils as vutils
import matplotlib.pyplot as plt

from .log_utils import OverlaySegmentationLogger
from .log_utils import ExperimentLogger
from .metrics import IoU
from .processing_utils import UnNormalize


def test_segmentation_overlay_plots(test_dataloader, net, img_means=None, img_std=None, figsize=(14,14)):
  '''
  
  Args:
      -test_dataloader (torch.utils.data.Dataloader)
      -net (nn.Module) 
      -img_means tuple of size number of channels in 0...1 
      -img_std   tuple of size number of channels in 0...1
      -figsize tuple:(int,int) (14,14)
  Returns
      -list of OverlaySegmentationLogger objects
      -list of matplotlib Figure objects
  '''
  if img_means is not None:
    unnormalizer = UnNormalize(img_means, img_std)
  
  device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
  net.to(device)
  net.eval()
  IOU_each_test_image = []
  figures = []
  segmentation_logs = []

  for (path_img, path_seg, img, seg, seg_gt),label in test_dataloader:
    
    img = img.to(device)
    seg_gt = seg_gt.long().to(device)
    
    label_pred = net(img)
 
    log_softmax = nn.LogSoftmax(dim=1)
    y_pred_binarized = log_softmax(label_pred).argmax(dim=1)

    mapping = test_dataloader.dataset.dataset.mapping_id_to_label

    N, c, h, w = img.shape

    for idx in range(N):    
      
      fig = plt.figure(figsize=figsize)
      if img_means is not None:
        curr_img = unnormalizer(img[idx])
      else:
        curr_img = img[idx]


      curr_img = curr_img.cpu().permute(1,2,0).numpy()
      curr_seg_gt = seg_gt[idx]
      curr_pred = y_pred_binarized[idx]
      curr_gt = mapping[label[idx].data.item()]

      curr_IOU = IoU(curr_pred, curr_seg_gt )

      curr_seg_gt = curr_seg_gt.cpu().numpy().squeeze()
      curr_pred = curr_pred.cpu().numpy().squeeze()


      raw = fig.add_subplot(221)
      raw.imshow( curr_img)
      raw.title.set_text('Raw wsi slide: '+  mapping[label[idx]]  )

      raw_ground_truth_overlay = fig.add_subplot(222)
      raw_ground_truth_overlay.imshow( curr_img)
      raw_ground_truth_overlay.imshow(curr_seg_gt, cmap='jet', alpha=0.6)#cmap='gray')
      raw_ground_truth_overlay.title.set_text('Overlap of raw image and ground truth(red)' )

      raw_pred_overlay = fig.add_subplot(223)
      raw_pred_overlay.imshow( curr_img)
      raw_pred_overlay.imshow(curr_pred, cmap='jet', alpha=0.6)
      raw_pred_overlay.title.set_text('Overlap of predicted(red) and raw image' )

      ground_truth_pred_overlay = fig.add_subplot(224)
      ground_truth_pred_overlay.imshow( curr_seg_gt,cmap='gray')
      ground_truth_pred_overlay.imshow(curr_pred, cmap='jet', alpha=0.6)
      ground_truth_pred_overlay.title.set_text('Overlap of predicted(red) and ground truth(blue)' + "%.3f"%(curr_IOU))

      IOU_each_test_image.append(curr_IOU)
      log = OverlaySegmentationLogger(fig,mapping.copy(), curr_IOU )

      segmentation_logs.append(log)
      figures.append(fig)
    del seg
    del img
    del seg_gt
    del label_pred
  return segmentation_logs,figures