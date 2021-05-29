import argparse


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf

from lib.data.plot_utils import ipyDisplay
from lib.data.graph_utils import estimate_graph_statistics
from lib.data.graph_utils import GraphItem, ConnectedComponentCV2, GraphItemLogs
from lib.data.image_utils import is_on, remove_isolated
from lib.data.processing_utils import UnNormalize
from lib.data.processing_utils import ToGraphTransform
from lib.models.neural_nets import UNet


from lib.managers import RCCDatasetManager, ExperimentManager
import numpy as np
import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import sys
from sklearn import model_selection
import copy


from lib.data.utils import DriveDownloader
from lib.models.neural_nets import AdaptUnet
import torch
import os
import argparse
import numpy as np
import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
from IPython.display import HTML
from matplotlib.animation import PillowWriter

from lib.models.neural_nets import AdaptUnet
from lib.data.datasets import CellDataset
from lib.data.datasets import CropDataset
from torch.utils.data import DataLoader
from lib.data.datasets import import_data


from lib.data.image_utils import np_recompose_tensor, img_paste_fn, seg_paste_fn
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from lib.models.neural_nets import UNet


import sys
import torch.optim as optim
import torch.nn as nn
from lib.data.metrics import IoU
import torchvision.utils as vutils

def train_segmentation_model(log_weights_path,  train_dataloader, val_dataloader, model,
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
  
    criterion = nn.CrossEntropyLoss()


 
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

            

            for i, batch in enumerate(loaders[phase]):
                      loss_train_batch = []
                      loss_validation_batch = []
                      IOU_train_batch = []
                      IOU_validation_batch = []
                    
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
              for idx, batch in enumerate(val_dataloader):
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
                  print("\nvalidation loss: ", mean_validation_loss)
                  print("validation iou: ", np.mean(IOU_validation_epoch))
                loss_validation.append(mean_validation_loss)
                IOU_validation.append(np.mean(IOU_validation_epoch))

                loss_validation_epoch = []
                IOU_validation_epoch = []
    return loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress



def main(args):

        #download domain adaptation network weights
        weights_downloader = DriveDownloader()
        weights_downloader.download_file_from_google_drive("15cbuIsknQAywkusaPve8qhkhLY3aowAR", "adaptunet_weights.pt")
        
        adaptunet = AdaptUnet(in_channels=1)
        #state_dict = torch.load("/content/gcn_vs_cnn_bioinformatics/weights/torch_weights.pt", map_location='cuda')
        PATH = "adaptunet_weights.pt"
        checkpoint = torch.load(PATH,map_location='cuda:0')
        adaptunet.load_state_dict(checkpoint['model_state_dict'])
        adaptunet.to('cuda')


        ###################
        ################### LOAD CROPPED DATASET
        def collate(batch):
                    cropped_images = []
                    cropped_masks = []
                    for img_crops, seg_crops in batch:
                        for img_crop_data, seg_crop_data in zip(img_crops, seg_crops):
                          (i,j), img_crop = img_crop_data
                          (i,j), seg_crop = seg_crop_data
                          cropped_images.append(img_crop.unsqueeze(dim=0))
                          cropped_masks.append(seg_crop.unsqueeze(dim=0))
                  
                    x_img = torch.cat(cropped_images, dim=0)
                    x_mask = torch.cat(cropped_masks, dim=0)
                    batch_len = len(batch)
                    l1 = [  x_img[idx:(idx+batch_len),...] for idx in range(0, x_img.shape[0], batch_len) ]
                    l2 = [  x_mask[idx:(idx+batch_len),...] for idx in range(0, x_mask.shape[0], batch_len) ]
                    return list(zip(l1,l2))

        img_train_transform = transforms.Compose([transforms.ToTensor(), 
                                                          transforms.Grayscale(num_output_channels=1)])
        seg_train_transform = transforms.Compose([transforms.ToTensor()])
        img_test_transform = transforms.Compose([transforms.ToTensor(), 
                                                         transforms.Grayscale(num_output_channels=1)])
        seg_test_transform = transforms.Compose([transforms.ToTensor()])

        dataset_root_path = "./rccdataset"
        IMAGE_SIDE_LEN = 2048
        CROPS_PER_SIDE = 4
        VERBOSE = True
        batch_size=args.batch_size
        num_workers=args.workers
        train = CropDataset(root_path=dataset_root_path,in_memory=False,resize_dim=IMAGE_SIDE_LEN,
                                              num_crops_per_side=CROPS_PER_SIDE,
                                              partition="Train",
                                              img_transform=img_train_transform,
                                              target_transform=seg_train_transform,
                                              verbose=VERBOSE)
        test =  CropDataset(root_path=dataset_root_path,in_memory=False,resize_dim=IMAGE_SIDE_LEN,
                                              partition="Test",
                                              num_crops_per_side=CROPS_PER_SIDE,
                                              img_transform=img_test_transform,
                                              target_transform=seg_test_transform,
                                      verbose=VERBOSE)
       

        train_crop_dataloader = DataLoader(train,batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate)
        test_crop_dataloader = DataLoader(test,batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate)


        SIDE_SHAPE = 512
        train_cleaned = get_cell_cleaned_dataset(train, adaptunet, SIDE_SHAPE=512)
        test_cleaned = get_cell_cleaned_dataset(test, adaptunet, SIDE_SHAPE=512)

        img_transform = transforms.Compose([transforms.ToTensor()] ) 
        seg_transform = transforms.Compose([transforms.ToTensor()] ) 

        train_transformed_sample_list = []
        for img, mask in tqdm(train_cleaned):
          img = img_transform(img)
          mask = seg_transform(mask)
          train_transformed_sample_list.append((img, mask) )
        test_transformed_sample_list = []
        for img, mask in tqdm(test_cleaned):
          img = img_transform(img)
          mask = seg_transform(mask)
          test_transformed_sample_list.append((img, mask) )







        unet_model = UNet(in_channels=1, out_channels=2, init_features=64)



        train_processed_dataloader = DataLoader(train_transformed_sample_list,batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_processed_dataloader = DataLoader(test_transformed_sample_list,batch_size=batch_size, shuffle=True, num_workers=num_workers)

        print("Unet training started: storing results in {}".format(args.weights_dir))
        loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress = train_segmentation_model("log_weights_dir",  train_processed_dataloader, test_processed_dataloader, unet_model,
                              learning_rate=0.00005,
                             n_epochs=40,
                             verbose=True, verbose_loss_acc=True,
                             weights_filename="unet_weights.pt")
        
        print("Unet training completed")
        plot_segmentation_progress(segmentation_progress, args.weights_dir)
        plot_results(args, loss_train, loss_validation, IOU_train, IOU_validation, metric_name='IoU')
                    








    



""" segmentation plotting"""
def plot_segmentation_progress(segmentation_progress, log_dir):
        finals = []
        pred_progress_list,mask_progress_list = [],[]
        for epoch, items in segmentation_progress:
            cat_tensor = torch.cat(  [  torch.where( pred > 0., pred, true*0.5 )  for idx, pred , true in items if pred.shape[1] == 2058  ]    , dim=0)
            
            pred_cat = torch.cat(  [ pred for idx, pred , true in items if pred.shape[1] == 2058  ]    , dim=0)
            true_cat = torch.cat(  [ true for idx, pred , true in items if pred.shape[1] == 2058  ]    , dim=0)
            pred_progress_list.append(pred_cat)
            mask_progress_list.append(true_cat)
            finals.append(cat_tensor)
        

        fig = plt.figure(figsize=(10,10))
        plt.axis("off")

        ims = [[plt.imshow(pred ,animated=True), plt.imshow(mask, animated=True, cmap='jet',alpha=0.2) ] for pred,mask in zip(pred_progress_list,mask_progress_list) ]

        ims = ims + [ims[-1] for _ in range(5)]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=100000, blit=True)
        writer = PillowWriter(fps=2)  

        path = os.path.join(log_dir, "segmentation_progress.gif")
        ani.save(path, writer=writer)  







    
def greedy_select(i, j, img, n_rows, n_cols):
    # select closest with least empty neighbors
    min_empty = 8
    min_i = -1
    min_j = -1
    choices_list = []
    # up  
    if i-1 >= 0:
        if img[i-1,j] == 0.:
            avg, num_samples = neigh_avg(i-1,j, img, n_rows, n_cols)
            if (8-num_samples < min_empty) and (8-num_samples >=1):
                min_empty = 8-num_samples
                min_i = i-1
                min_j = j
            #choices_list.append( (8-num_samples,(i-1,j) )   )
    #down
    if i+1 < n_rows:
        if img[i+1,j] == 0.:
            avg, num_samples = neigh_avg(i+1,j, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i+1,j) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i+1
                min_j = j
    #left
    if j-1 >= 0:
        if img[i,j-1] == 0.:
            avg, num_samples = neigh_avg(i,j-1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i,j-1) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i
                min_j = j-1
    #right
    if j+1 < n_cols:
        if img[i,j+1] == 0.:
            avg, num_samples = neigh_avg(i,j+1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i,j+1) )   )  
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i
                min_j = j+1
    
    #left_diag up
    if i-1 >= 0 and j-1 >= 0:
        if img[i-1,j-1] == 0.:
            avg, num_samples = neigh_avg(i-1,j-1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i-1,j-1) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i-1
                min_j = j-1
    #right_diag up
    if i-1 >= 0 and j+1 < n_cols:
        if img[i-1,j+1] == 0.:
            avg, num_samples = neigh_avg(i-1,j+1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i-1,j+1) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i-1
                min_j = j+1
    #left_diag down
    if i+1 < n_rows and j-1 >= 0:
        if img[i+1,j-1] == 0.:
            avg, num_samples = neigh_avg(i+1,j-1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples, (i+1,j-1) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i+1
                min_j = j-1
    #right_diag down
    if i+1 < n_rows and j+1 < n_cols:
        if img[i+1,j+1] == 0.:
            avg, num_samples = neigh_avg(i+1,j+1, img, n_rows, n_cols)
            #choices_list.append( (8-num_samples,(i+1,j+1) )   )
            if 8-num_samples < min_empty and 8-num_samples >=1 :
                min_empty = 8-num_samples
                min_i = i+1
                min_j = j+1
    
    #if len(choices_list) > 0:
    #  greedy_select_item = min(choices_list,key=lambda x: x[0])
    #  greedy_select_position = greedy_select_item[1]
    #  return greedy_select_position
    if min_i >= 0 and min_j >= 0:
        return min_i, min_j
    else:
        return -1,-1
    
def neigh_avg(i,j, img, n_rows, n_cols):
  neigh_total = 0.
  num_samples = 0

  # up
  if i-1 >= 0:
    if img[i-1,j] > 0.:
      neigh_total += img[i-1,j]
      num_samples += 1
  
  #down
  if i+1 < n_rows:
    if img[i+1,j] > 0.:
      neigh_total += img[i+1,j]
      num_samples += 1

  #left
  if j-1 >= 0:
    if img[i,j-1] > 0.:
      neigh_total += img[i,j-1]
      num_samples += 1

  #right
  if j+1 < n_cols:
    if img[i,j+1] > 0.:
      neigh_total += img[i,j+1]
      num_samples += 1
    
  
  #left_diag up
  if i-1 >= 0 and j-1 >= 0:
    if img[i-1,j-1] > 0.:
      neigh_total += img[i-1,j-1]
      num_samples += 1
  
  #right_diag up
  if i-1 >= 0 and j+1 < n_cols:
    if img[i-1,j+1] > 0.:
      neigh_total += img[i-1,j+1]
      num_samples += 1
  #left_diag down
  if i+1 < n_rows and j-1 >= 0:
    if img[i+1,j-1] > 0.:
      neigh_total += img[i+1,j-1]
      num_samples += 1
  #right_diag down
  if i+1 < n_rows and j+1 < n_cols:
    if img[i+1,j+1] > 0.:
      neigh_total += img[i+1,j+1]
      num_samples += 1
  
  #num_samples = max(num_samples,1)
  return neigh_total/max(num_samples,1), num_samples 

def fill_blanks_greedy(img, mask, n_rows, n_cols):

  #contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  #contours_mask = np.zeros(mask.shape,dtype=np.uint8)
  #contours_mask = cv2.drawContours(contours_mask, contours, -1,255, 1) 
  #row_locations, col_locations = np.where(contours_mask == 255)

  
  #"""
  filled_locations =  np.where(img > 0)
  avg_value = img[filled_locations].mean()

  empty_locations = np.where(img == 0)
  img[empty_locations] = avg_value
  #"""
  """
  #for i_loc, j_loc in zip(row_locations, col_locations):
  print("d")
  for i_loc in range(n_rows):
    
    for j_loc in range(n_cols):
      
      # if contours_mask[i_loc][j_loc] ....
      avg_val, samples_ = neigh_avg(i_loc,j_loc, img, n_rows, n_cols)
      if img[i_loc,j_loc] > 0 and samples_ > 0 and  samples_ < 8:# less than 8 are actually colored, means that we are in a not completely filled neighborhood
        
        queue = [(i_loc,j_loc)]
        while len(queue) > 0:
          location = queue.pop()#heapq.heappop(heap)
          i_new,j_new = location

          start = time.time()
          avg, num_filled_neigh = neigh_avg(i_new,j_new, img, n_rows, n_cols)
          end = time.time()
          #print(end-start)

          img[i_new,j_new] = avg
          #contours_mask[i_loc,j_loc] = 0

          start = time.time()
          i_new, j_new = greedy_select(i_new,j_new, img, n_rows, n_cols)
          end = time.time()
          #print("second", end-start)
          if i_new >= 0 and j_new >= 0:
            queue.append((i_new, j_new))
  for i_loc in range(n_rows):
    
    for j_loc in range(n_cols):
     
      avg_val, samples_ = neigh_avg(i_loc,j_loc, img, n_rows, n_cols)
      if img[i_loc,j_loc] == 0: 
          avg, num_filled_neigh = neigh_avg(i_loc,j_loc, img, n_rows, n_cols)
          img[i_loc,j_loc] = avg
  """
def remove_cells_from_image(img,mask):
    kernel = np.ones((7,7),np.uint8)

    curr_mask_dilated = cv2.dilate(mask,kernel,iterations = 1)

    #contours, hierarchy = cv2.findContours(curr_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #contours_img = np.zeros(curr_mask_dilated.shape,dtype=np.uint8)
    #contours_img = cv2.drawContours(curr_mask_dilated, contours, -1,255, 1)


    image_without_cells = np.where(curr_mask_dilated > 0, 0, img)

    fill_blanks_greedy(image_without_cells,curr_mask_dilated, *mask.shape)
    return image_without_cells

def reconstruct(image_crops):
            (i,j), crop = image_crops[0]    
            n_dim = len(crop.shape)
            if torch.is_tensor(crop):
              if n_dim == 2:
                init_output_fn = lambda original_side_len,n_original_channels: torch.zeros( (original_side_len,original_side_len) ) 
              else:
                init_output_fn = lambda original_side_len,n_original_channels: torch.zeros( (n_original_channels, original_side_len, original_side_len) ) 
            else:
              if n_dim == 2:
                init_output_fn = lambda original_side_len,n_original_channels: torch.zeros( (original_side_len,original_side_len) ) 
              else:
                init_output_fn = lambda original_side_len,n_original_channels: torch.zeros( (original_side_len, original_side_len, n_original_channels) ) 

          
            recomposed_img = np_recompose_tensor(image_crops,  image_shape_getter_fn=lambda x: x.shape[:-1], n_channels_getter_fn= lambda x:x.shape[-1],
                                    paste_fn= img_paste_fn,  init_output_fn=init_output_fn )

            return recomposed_img
def paste_fn(destination_tensor,row_indices, col_indices, curr_crop):
    destination_tensor[:,row_indices,col_indices] = curr_crop

def get_cell_cleaned_dataset(dataset, adaptunet, SIDE_SHAPE=512):
    img_recomposed_list = []
    seg_recomposed_list = []
    samples_list = []

    for img_crops, seg_crops in tqdm(dataset,leave=True, position=0, total=len(dataset)):
      
      img_processed_crops = []
      
      for (i,j),img_crop in img_crops:
       
        img_crop = img_crop.unsqueeze(0).to('cuda')
        cell_segmentation_mask = adaptunet(img_crop).argmax(dim=1).detach().cpu().squeeze(0)
       
        img_crop = img_crop.to('cpu')

        cell_segmentation_mask = (cell_segmentation_mask*255).numpy().astype(np.uint8)
        #print(cell_segmentation_mask.shape)
        #print(img_crop.shape)
        #print(img_crop.dtype)
        #print(cell_segmentation_mask.shape)
        #print(cell_segmentation_mask.dtype)
     
        cleaned_img_crop = torch.Tensor(remove_cells_from_image(img_crop.squeeze(), cell_segmentation_mask) ) .unsqueeze(0)
       
        #print(cleaned_img_crop.shape)
        #print(cleaned_img_crop.dtype)
        
        img_processed_crops.append( ((i,j),cleaned_img_crop) )
      """
      image_tensor = None
      for (i,j),img_crop in img_crops:
        img_crop = img_crop.unsqueeze(0)
        if image_tensor == None:
          image_tensor = torch.cat([img_crop], dim=0)
        else:
          image_tensor = torch.cat([image_tensor, img_crop],dim=0)

      image_tensor = image_tensor.to('cuda')
      #print(image_tensor.shape)
      cell_segmentation_masks = adaptunet(image_tensor).argmax(dim=1).detach().cpu()#.squeeze(0)
      #print(cell_segmentation_masks.shape)
      for idx, (i,j), img_crop in enumerate(img_crops):
        cell_segmentation_mask = cell_segmentation_masks[idx].squeeze(0)

        cell_segmentation_mask = (cell_segmentation_mask*255).numpy().astype(np.uint8)
        #print(cell_segmentation_mask.shape)
        #print(img_crop.shape)
        #print(img_crop.dtype)
        #print(cell_segmentation_mask.shape)
        #print(cell_segmentation_mask.dtype)
        cleaned_img_crop = torch.Tensor(remove_cells_from_image(img_crop.squeeze(), cell_segmentation_mask) ) .unsqueeze(0)
        #print(cleaned_img_crop.shape)
        #print(cleaned_img_crop.dtype)
        
        img_processed_crops.append( ((i,j),cleaned_img_crop) )   
      del image_tensor 
      """
      img_recomposed = np_recompose_tensor(img_processed_crops, image_shape_getter_fn=lambda img: img.shape[1:], n_channels_getter_fn= lambda img:img.shape[0],
                            paste_fn= paste_fn,  init_output_fn=lambda side_len, channels: torch.zeros((channels,side_len,side_len)) )
      seg_recomposed = np_recompose_tensor(seg_crops,image_shape_getter_fn=lambda x: x.shape[1:], n_channels_getter_fn= lambda x:x.shape[0],
                            paste_fn= paste_fn,  init_output_fn=lambda side_len, channels: torch.zeros((channels, side_len,side_len) ) )

      #print(img_recomposed.dtype)
      #print(img_recomposed.numpy().dtype)
      #print(set(img_recomposed.numpy().flatten().tolist()) )
      img_resized = cv2.resize( (img_recomposed.squeeze().numpy()*255).astype(np.uint8), (SIDE_SHAPE,SIDE_SHAPE))
      seg_resized = cv2.resize(seg_recomposed.squeeze().numpy(), (SIDE_SHAPE,SIDE_SHAPE))

      img_recomposed_list.append(img_resized[np.newaxis,...])
      seg_resized = np.where(seg_resized >= 0.5, 1, 0  )
      seg_recomposed_list.append(seg_resized)
      samples_list.append( (img_resized, seg_resized) )

    return samples_list









def plot_results(img_file_path, train_loss, val_loss, train_metric, val_metric, metric_name='accuracy'):
        fig = plt.figure(figsize=(12,4))
        loss_fig = fig.add_subplot(121)
        loss_fig.plot(train_loss)
        loss_fig.plot(val_loss)
        loss_fig.set_title('model loss')
        loss_fig.set_ylabel('loss')
        loss_fig.set_xlabel('epoch')
        loss_fig.legend(['train', 'validation'], loc='upper left')

        acc_fig = fig.add_subplot(122)
        acc_fig.plot(train_metric)
        acc_fig.plot(val_metric)
        acc_fig.set_title('model ' + metric_name)
        acc_fig.set_ylabel(metric_name)
        acc_fig.set_xlabel('epoch')
        acc_fig.legend(['train', 'validation'], loc='upper left')
        plt.savefig(img_file_path)



def setup_dataset(args):
    ROOT_PATH = args.images 
    #assert os.path.exists(ROOT_PATH), "Error: given path does not exist"
    RESIZE_DIM = 512
    # dataset setup
    datasetManager = RCCDatasetManager(ROOT_PATH,
                        download_dataset=True,
                        standardize_config={'by_patient':args.std_by_patient, 
                                             'by_patient_train_avg_stats_on_test':args.std_by_patient,
                                             'by_single_img_stats_on_test':False}, 
                        load_graphs=False,
                    
                        verbose=True)
    return datasetManager
def setup_augmentation(args):
    augment_params_dict = dict()
    
    if args.rand_rot:
            augment_params_dict['rotate'] = {'prob':1.0}
            
    if args.rand_crop:
            augment_params_dict['resized_crop'] = {'prob':1.0, 'original_kept_crop_percent':(0.7,1.0)}
            
    if args.rand_elastic_deform:
            augment_params_dict['elastic_deform'] = {'alpha':(1,10), 'sigma':(0.07, 0.13), 'alpha_affine':(0.07, 0.13), 'random_state':None}
            
    return augment_params_dict






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN experiment"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--cross-val",
        type=bool,
        default=False,
        help="Perform 5-fold cross-validation: [True,False] (default to False)",
    )







    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )


    parser.add_argument(
        "--weights-dir", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--weights-fname", type=str, default="torch_weights.pt", help="weights filename"
    )
    parser.add_argument(
        "--images", type=str, default="./vascular_segmentation", help="root folder with train and test folders"
    )


    args = parser.parse_args()
    main(args)