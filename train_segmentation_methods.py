
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from log_utils import ExperimentLogger
import torchvision.utils as vutils

import cv2
from managers import RCCDatasetManager


from metrics import IoU


def train_segmentation_model(log_weights_path,  train_dataloader, val_dataloader, model, 
                              learning_rate=0.0001,
                             n_epochs=10,
                             verbose=True, verbose_loss_acc=False,
                             weights_filename="torch_weights.pt"):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    torch.cuda.empty_cache() 
    

    if not os.path.exists(log_weights_path):
      os.makedirs(log_weights_path)
    

    loaders = {"train": train_dataloader, "valid": val_dataloader}
    

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

    
    best_validaton_BCE = sys.float_info.max

    

    # metrics to track
    IOU_train = []
    IOU_validation = []
    loss_train = []
    loss_validation = []

    # track segmentation progress by means of plots of the predicted masks
    segmentation_progress = []
    segmentation_progress_pred = []
    segmentation_progress_true_mask = []
    val_progress_dataloader = DataLoader(val_dataloader.dataset, batch_size=val_dataloader.batch_size, 
                                         shuffle=False,  num_workers=2, drop_last=True)


    step = 0
    epochs = n_epochs
    for epoch in tqdm(range(epochs), total=epochs,position=0, leave=True):
        loss_train_epoch = []
        loss_validation_epoch = []
        IOU_train_epoch = []
        IOU_validation_epoch = []
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                (path_img, path_seg, img, seg, seg_gt),label  = data

                img, seg_gt  = img.to(device).float(), seg_gt.long().squeeze(dim=1).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(img)
                    loss = criterion(y_pred,seg_gt)
                    logsoftmax = nn.LogSoftmax(dim=1)
                    y_pred_binarized = logsoftmax(y_pred).argmax(dim=1, keepdim=True)
                    curr_IOU = IoU(y_pred_binarized, seg_gt)

                    if phase == "valid":
                        loss_validation_epoch.append(loss.item())
                        IOU_validation_epoch.append(curr_IOU)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        loss_train_epoch.append(loss.item())
                        IOU_train_epoch.append(curr_IOU)
                    del img
                    del seg_gt
                    del y_pred
            if phase == "train" :
                    loss_train.append(np.mean(loss_train_epoch))
                    IOU_train.append(np.mean(IOU_train_epoch))
                    
                    loss_train_epoch = []
                    IOU_train_epoch = []
            
            if phase == "valid":
                # track best model so far
                mean_validation_loss = np.mean(loss_validation_epoch)
                if mean_validation_loss < best_validaton_BCE:
                  best_validaton_BCE = mean_validation_loss
                  torch.save(model.state_dict(), os.path.join(log_weights_path, weights_filename))
                if verbose_loss_acc:
                  print("valid: ", mean_validation_loss)
                  print("iou_valid: ", np.mean(IOU_validation_epoch))
                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []

       
                # track segmentation progress
                progress = []
                progress_preds =[]
                progress_true_masks = []
                for idx, batch in enumerate(val_progress_dataloader):
                      (path_img, path_seg, img, seg, mask),label = batch 
                                
                      img, mask  = img.to(device).float(), mask.long().squeeze(dim=1).to(device)
            
                      with torch.set_grad_enabled(phase == "train"):
                                    y_pred = model(img)
                                    logsoftmax = nn.LogSoftmax(dim=1)
                                    y_pred_binarized = logsoftmax(y_pred).argmax(dim=1, keepdim=True)
           

                                    pred_grid = np.transpose(vutils.make_grid(
                                        y_pred_binarized.detach().float(), padding=2, nrow=5)
                                        .cpu(),(1,2,0))
                                    true_grid = np.transpose(vutils.make_grid(
                                        mask.unsqueeze(1).float(), padding=2, nrow=5)
                                        .cpu(),(1,2,0))
                                    
                                    
                                    progress.append( (idx, pred_grid, true_grid ) )

                                    progress_preds.append( (idx, y_pred_binarized.detach().cpu() ) )
                                    progress_true_masks.append( (idx, mask.cpu()) )
                        
                      del img
                      del mask
                      del y_pred
                segmentation_progress.append( (epoch, progress)  )
                segmentation_progress_pred.append( (epoch, progress_preds) )
                segmentation_progress_true_mask.append( (epoch, progress_true_masks) )
    return loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress, segmentation_progress_pred, segmentation_progress_true_mask

"""## validate segmentation model method"""

def validate_segmentation(model, test_dataloader):
  model.eval()
  device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
  IoU_test =[]
  
  for (path_img, path_seg, img, seg, seg_gt),label in test_dataloader:
    seg = seg.to(device)
    img = img.float().to(device)
    # long() is important as transformation may have distorted some pixels
    seg_gt = seg_gt.long().to(device)
    seg_pred = model(img)

    log_softmax = nn.LogSoftmax(dim=1)
    y_pred_binarized = log_softmax(seg_pred).argmax(dim=1, keepdim=True)
    IoU_test.append(IoU(y_pred_binarized, seg_gt))
    del seg
    del img
    del seg_gt
    del seg_pred
    
  return IoU_test






def train_crop_segmentation_model(log_weights_path,  train_dataloader, val_dataloader, model,
                              learning_rate=0.00001,
                             n_epochs=10,
                             verbose=True, verbose_loss_acc=False,
                             weights_filename="torch_weights.pt"):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    torch.cuda.empty_cache() 
    
    unet_weights_folder_path = log_weights_path
    weights_filename = weights_filename
    if not os.path.exists(unet_weights_folder_path):
      os.makedirs(unet_weights_folder_path)
    

    loaders = {"train": train_dataloader, "valid": val_dataloader}
    in_channels = 3
    criterion = nn.CrossEntropyLoss()


    out_channels=2
   
 
    model.to(device)
  

    log_dir = "./logs"
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)



    
    best_validaton_BCE = sys.float_info.max
    #0.0001 goood
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    IOU_train = []
    IOU_validation = []
    loss_train = []
    loss_validation = []
    
    evaluation_multi_crop_batch = next(iter(val_dataloader))
    segmentation_progress = []
    step = 0
    epochs = n_epochs
    for epoch in tqdm(range(epochs), total=epochs,position=0, leave=True):
        loss_train_epoch = []
        loss_validation_epoch = []
        IOU_train_epoch = []
        IOU_validation_epoch = []
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            

            for i, multi_crop_batch in enumerate(loaders[phase]):
                    step += 1
          
                    loss_train_batch = []
                    loss_validation_batch = []
                    IOU_train_batch = []
                    IOU_validation_batch = []
                    for batch in multi_crop_batch:
                      img, mask = batch 
                      
                      img, mask  = img.to(device).float(), mask.long().squeeze(dim=1).to(device)
                      
                      optimizer.zero_grad()

                      with torch.set_grad_enabled(phase == "train"):
                          y_pred = model(img)
                          loss = criterion(y_pred,mask)
                          softmax = nn.LogSoftmax(dim=1)
                          y_pred_binarized = softmax(y_pred).argmax(dim=1, keepdim=True)
                          
                          curr_IOU = IoU(y_pred_binarized, mask)


                          if phase == "valid":
                              loss_validation_epoch.append(loss.item())
                              IOU_validation_epoch.append(curr_IOU)
                            
                              loss_validation_batch.append(loss.item())
                              IOU_validation_batch.append(curr_IOU)

                          if phase == "train":
                              loss.backward()
                              optimizer.step()
                         
                              loss_train_epoch.append(loss.item())
                              IOU_train_epoch.append(curr_IOU)

                              loss_train_batch.append(loss.item())
                              IOU_train_batch.append(curr_IOU)
                              
                          del img
                          del mask
                          del y_pred


        if phase == "valid":
              progress = []
              for idx, batch in enumerate(evaluation_multi_crop_batch):
                    img, mask = batch 
                              
                    img, mask  = img.to(device).float(), mask.long().squeeze(dim=1).to(device)
          
                    with torch.set_grad_enabled(phase == "train"):
                                  y_pred = model(img)
                                  softmax = nn.LogSoftmax(dim=1)
                                  y_pred_binarized = softmax(y_pred).argmax(dim=1, keepdim=True)
                                  
                                  pred_grid = np.transpose(vutils.make_grid(
                                      y_pred_binarized.detach().float(), padding=2, nrow=5)
                                      .cpu(),(1,2,0))
                                  true_grid = np.transpose(vutils.make_grid(
                                      mask.unsqueeze(1).float(), padding=2, nrow=5)
                                      .cpu(),(1,2,0))
                                  progress.append( (idx, pred_grid, true_grid ) )
                      
                    del img
                    del mask
                    del y_pred
              segmentation_progress.append( (epoch, progress)  )

  

        if phase == "train" :
                    loss_train.append(np.mean(loss_train_epoch))
                    IOU_train.append(np.mean(IOU_train_epoch))
                    
                    loss_train_epoch = []
                    IOU_train_epoch = []
            
        if phase == "valid":
                
                mean_validation_loss = np.mean(loss_validation_epoch)
                if mean_validation_loss < best_validaton_BCE:
                  best_validaton_BCE = mean_validation_loss
                  torch.save(model.state_dict(), os.path.join(unet_weights_folder_path, weights_filename))
                if verbose_loss_acc:
                  print("valid: ", mean_validation_loss)
                  print("iou_valid: ", np.mean(IOU_validation_epoch))
                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []
    return loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress

"""## segmentation training plotting function"""














