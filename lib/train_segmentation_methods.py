
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import sys
import cv2
from tqdm import tqdm

from .data.log_utils import ExperimentLogger
from .data.metrics import IoU


def train_segmentation_model(log_weights_path,  train_dataloader, val_dataloader, model, 
                              learning_rate=0.0001,
                             n_epochs=10,
                             verbose=True, verbose_loss_acc=False,
                             weights_filename="torch_weights.pt"):
    '''
    Args:
        log_weights_path (str) where to save the torch weights
        train_dataloader (torch.utils.data.Dataloader)
        val_dataloader (torch.utils.data.Dataloader)
        model (nn.Module)
        learning_rate=0.0001 (float)
        n_epochs=10 (int)
        verbose_loss_acc=True (print results for each epoch of training in the shell)
        weights_filename="torch_weights.pt"
    
    Returns:
        loss_train (list of floats)
        loss_validation (list of floats)
        IOU_train (list of floats)
        IOU_validation (list of floats)
        model   (nn.Module)
        segmentation_progress  list of tuples: epoch_id,(idx, pred_grid, true_grid )
        segmentation_progress_pred list of tuples: epoch_id,(idx, pred_grid )
        segmentation_progress_true_mask  list of tuples: epoch_id,(idx, true_grid )
    '''
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

                  PATH = os.path.join(log_weights_path, weights_filename)
                  torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
 
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
  '''
    Args:
        model (nn.Module)
        test_dataloader (torch.utils.data.Dataloader)
    Returns:
        IOU_test (list of floats)

  '''
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
    '''
    Args:
        log_weights_path (str) where to save the torch weights
        train_dataloader (torch.utils.data.Dataloader)
        val_dataloader (torch.utils.data.Dataloader)
        model (nn.Module)
        learning_rate=0.0001 (float)
        n_epochs=10 (int)
        verbose_loss_acc=True (print results for each epoch of training in the shell)
        weights_filename="torch_weights.pt"
    
    Returns: 
        loss_train (list of floats)
        loss_validation (list of floats)
        IOU_train (list of floats)
        IOU_validation (list of floats)
        model   (nn.Module)
        segmentation_progress  list of tuples: epoch_id,(idx, pred_grid, true_grid )
    '''
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
                  PATH = os.path.join(unet_weights_folder_path, weights_filename)
                  torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
                if verbose_loss_acc:
                  print("\nvalidation loss: ", mean_validation_loss)
                  print("validation iou: ", np.mean(IOU_validation_epoch))
                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []
    return loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress

"""## segmentation training plotting function"""

















def train_cell_segmentation_model(log_weights_path,  train_dataloader, val_dataloader, model, 
                              learning_rate=0.0001,
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

    model.to(device)
  

    log_dir = "./logs"
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)



    
    best_validaton_BCE = sys.float_info.max

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    IOU_train = []
    IOU_validation = []
    loss_train = []
    loss_validation = []

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

                images,labels  = data
                
                
                img, seg_gt  = images.to(device).float(), labels.long().squeeze().to(device)
                 
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(img)
                   
                    loss = criterion(y_pred,seg_gt)
                    softmax = nn.Softmax(dim=1)
                    y_pred_binarized = softmax(y_pred).argmax(dim=1, keepdim=True)
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
                
                mean_validation_loss = np.mean(loss_validation_epoch)
                if mean_validation_loss < best_validaton_BCE:
                  best_validaton_BCE = mean_validation_loss

                  PATH = os.path.join(unet_weights_folder_path, weights_filename)
                  torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
                  
                if verbose_loss_acc:
                  print("\nvalidation loss: ", mean_validation_loss)
                  print("validation iou: ", np.mean(IOU_validation_epoch))
                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []
    return loss_train, loss_validation, IOU_train, IOU_validation, model








def train_adaptation_unet(model , 
                      train_source_dataloader,
                      val_source_dataloader,
                      train_target_dataloader,
                      val_target_dataloader,
                      lambda_ = 1.,
                      log_weights_path="log_dir" ,
                      learning_rate=0.00005,
                      n_epochs=50,
                      verbose=True,
                      verbose_loss_acc=False,
                      weights_filename="torch_weights.pt"):
    def get_batch(target_data, num_samples=4):
            target_images_list = []
            target_masks_list = []
            [ target_images_list.extend(target_img_crops)for target_img_crops, target_mask_crops in target_data ]
            [ target_masks_list.extend(target_mask_crops)for target_img_crops, target_mask_crops in target_data ]

            # subsample the crop batch 
            indices = np.array([i for i in range(len(target_images_list) ) ] )
            np.random.shuffle(indices)
            subsample_size = num_samples

            target_img_sample = torch.cat([ target_images_list[idx].unsqueeze(0) for idx in indices[:subsample_size] ] ,
                                            dim=0)
            return target_img_sample
    

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    unet_weights_folder_path = log_weights_path
    weights_filename = weights_filename
    if not os.path.exists(unet_weights_folder_path):
      os.makedirs(unet_weights_folder_path)
    
    def cycle(iterable):
      while True:
          for x in iterable:
              yield x

    source_train_dataiter = train_source_dataloader
    target_train_dataiter = iter(cycle(train_target_dataloader))  

    source_validation_dataiter = val_source_dataloader
    target_validation_dataiter = iter(cycle(val_target_dataloader))  

    loaders = {"train": (source_train_dataiter, target_train_dataiter), "valid": (source_validation_dataiter, target_validation_dataiter)}
    

    lambda_ = lambda_
  
    criterion = nn.CrossEntropyLoss()

    model.to(device)
  

    log_dir = "./logs"
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)



    
    best_validaton_BCE = sys.float_info.max

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    #source_val_iter = iter(source_validation_dataiter)
    evaluation_batch = [batch for batch in source_validation_dataiter][:4]
    segmentation_progress = []

    IOU_train = []
    IOU_validation = []
    loss_train = []
    loss_validation = []

    pretext_loss = []
    pretext_acc = []

    epochs = n_epochs
    for epoch in tqdm(range(epochs), total=epochs,position=0, leave=True):
        loss_train_epoch = []
        loss_validation_epoch = []
        IOU_train_epoch = []
        IOU_validation_epoch = []


        pretext_loss_epoch = []
        pretext_acc_epoch = []
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            source_dataiter, target_dataiter = loaders[phase]
            for source_data, target_data in zip(source_dataiter, target_dataiter):
                optimizer.zero_grad()

                
                #main task
                source_images,source_labels  = source_data
                #print(source_labels.shape)
                source_img = source_images.float().to(device)
                source_seg_gt  = source_labels.long().squeeze().to(device)

                
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(source_img, 'main')
                   
                    main_loss = criterion(y_pred,source_seg_gt)
                    softmax = nn.Softmax(dim=1)
                    y_pred_binarized = softmax(y_pred).argmax(dim=1)
                    curr_IOU = IoU(y_pred_binarized, source_seg_gt)

                    
                    if phase == "train":
                        main_loss.backward()
                        

                        loss_train_epoch.append(main_loss.item())
                        IOU_train_epoch.append(curr_IOU)
                    if phase == "valid":
                       
                        loss_validation_epoch.append(main_loss.item())
                        IOU_validation_epoch.append(curr_IOU)
                    
                    del source_seg_gt
                    del y_pred
                    

                if phase == 'train':
                      
                      # pretext task
                      ### source domain            
                      source_img = source_img
                      source_img_shape = source_img.shape
                      source_labels = torch.zeros(source_img.shape[0]).long().to(device)

                      corrects = 0 
                      total = 0
                      loss_sum = 0
                      with torch.set_grad_enabled(phase == "train"):
                          y_pred = model(source_img, 'pretext')
                        
                          pretext_source_loss = criterion(y_pred,source_labels)
                          softmax = nn.Softmax(dim=1)
                          y_pred_binarized = softmax(y_pred).argmax(dim=1)
                        
                          if phase == "train":
                            (pretext_source_loss*lambda_).backward()
                            loss_sum += pretext_source_loss.item()

                            corrects += torch.sum(y_pred_binarized == source_labels).data.item()
                            total += len(source_labels)
                          

                          del source_img
                          del source_labels
                          del y_pred

                      ### target domain
                      target_img = get_batch(target_data, num_samples=source_img_shape[0])
                        
                      target_img = target_img.float().to(device)
                      target_labels = torch.ones(target_img.shape[0]).long().to(device)


                      with torch.set_grad_enabled(phase == "train"):
                          y_pred = model(target_img, 'pretext')
                        
                          pretext_target_loss = criterion(y_pred,target_labels)
                          softmax = nn.Softmax(dim=1)
                          y_pred_binarized = softmax(y_pred).argmax(dim=1)
                        
                          if phase == "train":
                              (pretext_target_loss*lambda_).backward()
                              loss_sum += pretext_target_loss.item()

                              corrects += torch.sum(y_pred_binarized == target_labels).data.item()
                              total += len(target_labels)
                          
                          
                          del target_img
                          del target_labels
                          del y_pred
                      
                      pretext_loss_epoch.append((loss_sum )/2 )
                      pretext_acc_epoch.append(  corrects/total  )

             
                      optimizer.step()
            if phase == "train" :
                    loss_train.append(np.mean(loss_train_epoch))
                    IOU_train.append(np.mean(IOU_train_epoch))
                    pretext_loss.append(np.mean(pretext_loss_epoch))
                    pretext_acc.append(np.mean(pretext_acc_epoch))

                    print("train IoU: ",IOU_train[-1])
                    print("pretext_loss: ", pretext_loss[-1])
                    print("pretext_acc: ", pretext_acc[-1])


                    pretext_loss_epoch = []
                    pretext_acc_epoch = []
                    loss_train_epoch = []
                    IOU_train_epoch = []
            
            if phase == "valid":
                
                mean_validation_loss = np.mean(loss_validation_epoch)
                if mean_validation_loss < best_validaton_BCE:
                  best_validaton_BCE = mean_validation_loss

                  PATH = os.path.join(unet_weights_folder_path, weights_filename)
                  torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)

                if verbose_loss_acc:
                  print("\nvalidation loss: ", mean_validation_loss)
                  print("validation iou: ", np.mean(IOU_validation_epoch))

                progress = []
                for idx, batch in enumerate(evaluation_batch):
                                
                                img, mask = batch 
                                          
                                img, mask  = img.to(device).float(), mask.long().squeeze().to(device)
                      
                                with torch.set_grad_enabled(phase == "train"):
                                              y_pred = model(img)
                                              softmax = nn.Softmax(dim=1)
                                              y_pred_binarized = softmax(y_pred).argmax(dim=1,keepdim=True)
                    
                                              pred_grid = np.transpose(vutils.make_grid(
                                                  y_pred_binarized.detach().float(), padding=2, nrow=5)
                                                  .cpu(),(1,2,0))
                                              true_grid = np.transpose(vutils.make_grid(
                                                  mask.unsqueeze(1).float(), padding=2, nrow=5)
                                                  .cpu(),(1,2,0))
                                              progress.append( (idx, pred_grid, true_grid ) )


                                
                                print(pred_grid.shape)
                                print(true_grid.shape)

                                del img
                                del mask
                                del y_pred
                segmentation_progress.append( (epoch, progress)  )

                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []

    return loss_train, loss_validation, IOU_train, IOU_validation, pretext_loss, pretext_acc, model, segmentation_progress