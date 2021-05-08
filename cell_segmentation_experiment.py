import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf

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



from neural_nets import AdaptUnet



from datasets import CellDataset
from datasets import CropDataset
from torch.utils.data import DataLoader
from datasets import import_data

def main(args):
        
        os.system('mkdir content')
        os.system('cd content')
        os.system('git clone https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet.git')
        os.system('cp -r ./Histology-Image-Segmentation-using-UNet/. .')
        #os.system('cd /content')

        if os.path.exists('gcn_vs_cnn_bioinformatics'):
            os.system('rm -rf gcn_vs_cnn_bioinformatics')
        os.system('git clone https://chris1nexus:Android560ti@github.com/Chris1nexus/gcn_vs_cnn_bioinformatics.git')
        os.system('cd gcn_vs_cnn_bioinformatics')

        images, labels = import_data("../data", resize_shape=(512,512))

        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
        seg_transform = transforms.Compose([transforms.ToTensor()])
        cell_dataset = CellDataset(data_path="../data", img_transform=img_transform, seg_transform=seg_transform)
        cell_test_dataset = CellDataset(data_path="../", mode="test", img_transform=img_transform, seg_transform=seg_transform)
        train_cell_dataloader = DataLoader(cell_dataset, batch_size=4, shuffle=True)
        test_cell_dataloader = DataLoader(cell_test_dataset, batch_size=4, shuffle=True)

       

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
        batch_size=4
        num_workers=2
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
        train_crop_dataloader = DataLoader(train,batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
        test_crop_dataloader = DataLoader(test,batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)

    
        in_channels = 1
        out_channels = 2

        
        kwargs_dict = {'learning_rate':args.lr, 'batch_size':args.batch_size,
                 'n_epochs':args.epochs,
                 'log_weights_path':args.weights_dir,
                 'weights_filename':args.weights_fname,
                 'verbose':True,
                  'verbose_loss_acc': False}
        
        
        train_source_dataloader = train_cell_dataloader
        val_source_dataloader = test_cell_dataloader
        train_target_dataloader = train_crop_dataloader
        val_target_dataloader = test_crop_dataloader
      
        model = AdaptUnet( in_channels=1, out_channels=2)

        print("AdaptUnet training started: storing results in {}".format(args.weights_dir))
        loss_train, loss_validation, IOU_train, IOU_validation, pretext_loss, pretext_acc, model = train_adaptation_unet(model , 
                      train_source_dataloader,
                      val_source_dataloader,
                      train_target_dataloader,
                      val_target_dataloader,
                      **kwargs_dict)
        print("AdaptUnet training completed")
        
        plot_results(args, loss_train, loss_validation, IOU_train, IOU_validation, metric_name='IoU')
        plot_results(args, pretext_loss, pretext_loss, pretext_acc, pretext_acc, metric_name='pretext_accuracy')



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
        #HTML(ani.to_jshtml())






def plot_results(args, train_loss, val_loss, train_metric, val_metric, metric_name='accuracy'):
        img_file_path = os.path.join(args.weights_dir, args.weights_fname +"_train.png")
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


def setup_dataset(args, load_graphs=False):
    ROOT_PATH = args.images 
    #assert os.path.exists(ROOT_PATH), "Error: given path does not exist"
    
    # dataset setup
    datasetManager = RCCDatasetManager(ROOT_PATH,
                     
                        download_dataset=True,
                        standardize_config={'by_patient':args.std_by_patient, 
                                             'by_patient_train_avg_stats_on_test':args.std_by_patient,
                                             'by_single_img_stats_on_test':False}, 
                        load_graphs=load_graphs,
                    
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

def setup_preprocessing(args):
    img_means = RCCDatasetManager.img_means
    img_std = RCCDatasetManager.img_std


    gray_mean = RCCDatasetManager.gray_mean
    gray_std = RCCDatasetManager.gray_std

    totensor = transforms.ToTensor() 
    img_train_transforms_list = [totensor]
    img_test_transforms_list = [totensor]

    means = img_means
    st_deviations = img_std
    if args.format == 'gray': # images are from the start in RGB format, so a transformation is required only for the Grayscale case
            grayscale = transforms.Grayscale()
            img_train_transforms_list.append(grayscale)
            img_test_transforms_list.append(grayscale)
            means = gray_means
            st_deviations = gray_std


    if args.std:
            normalize = transforms.Normalize(means, st_deviations)
            img_train_transforms_list.append(normalize)
            img_test_transforms_list.append(normalize)

    img_train_transform = transforms.Compose(img_train_transforms_list) 
    img_test_transform = transforms.Compose(img_test_transforms_list)

    seg_train_transform = transforms.Compose([transforms.ToTensor()] )
    seg_test_transform = transforms.Compose([transforms.ToTensor()] )

    return (img_train_transform, img_test_transform), (seg_train_transform, seg_test_transform)








if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNET experiment"
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
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.0001)",
    )




    parser.add_argument(
        "--std",
        type=bool,
        default=True,
        help="standardize slide images according to the channel statistics (default: True)",
    )
    parser.add_argument(
        "--std-by-patient",
        type=bool,
        default=False,
        help="compute mean and variance for each 'train' split patient\n and standardize each of their samples by their own statistics: test samples are standardized according to the average mean and pooled variance (default: False)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="rgb",
        help="slide image format:['rgb','gray'] (default is rgb)",
    )







    parser.add_argument(
        "--rand-rot",
        type=bool,
        default=False,
        help="random rotations (90,180,270 degrees) data augmentation  (default: False)",
    )
    parser.add_argument(
        "--rand-crop",
        type=bool,
        default=False,
        help="random crop and zoom (keep from 0.7 to 1.0 of the original image ) data augmentation (default: False)"
    )

    parser.add_argument(
        "--rand-elastic-deform",
        type=bool,
        default=False,
        help="elastic deformation (default: False):\n\t\t\t"+\
                                            "alpha in [1,4]\n\t\t\t" +\
                                            "sigma in [0.07, 0.13]\n\t\t\t"+\
                                             "alpha affine in [0.07, 0.13]"
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