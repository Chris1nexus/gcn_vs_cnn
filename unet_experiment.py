import argparse


from plot_utils import ipyDisplay
from graph_utils import estimate_graph_statistics
from graph_utils import GraphItem, ConnectedComponentCV2, GraphItemLogs
from image_utils import is_on, remove_isolated
from processing_utils import UnNormalize
from processing_utils import ToGraphTransform
from neural_nets import UNet


from managers import RCCDatasetManager, ExperimentManager
import numpy as np
import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt




def main(args):


        # dataset setup
        datasetManager = setup_dataset(args)

        augment_params_dict = setup_augmentation(args)
        augment = len(augment_params_dict) > 0
        # experiment setup
        experiment_mgr = ExperimentManager( datasetManager )


        if args.format == 'rgb':
            in_channels = 3
        else:
            in_channels = 1
        out_channels = 2
        init_features = 64


        unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)


        
        kwargs_dict = {'learning_rate':args.lr, 'batch_size':args.batch_size, 'img_train_transform':img_train_transform, 'seg_train_transform':seg_train_transform,
                                                                  'img_test_transform':img_test_transform,
                                                                  'seg_test_transform':seg_test_transform,
                                                                  'epochs':args.epochs,
                                                                  'log_weights_path':args.weights_dir,
                                                                  'weights_filename':args.weight_fname,
                                                                  'augment':augment, 
                                                                  'augment_params_dict':augment_params_dict,
                                                                  'verbose_loss_acc': False}

        loss_train, loss_validation, IOU_train, IOU_validation, IOU_test, unet, segmentation_progress, segmentation_progress_pred, segmentation_progress_true_mask = experiment_mgr.train_unet(unet, **kwargs_dict)
        plot_results(args, loss_train, loss_validation, IOU_train, IOU_validation, metric_name='IoU')
					








	








        
def plot_results(args, train_loss, val_loss, train_metric, val_metric, metric_name='accuracy'):
        img_file_path = os.path.join(args.weights_dir, args.weights_fname +"_train.png")
        fig = plt.figure(figsize=(12,4))
        loss_fig = fig.add_subplot(121)
        loss_fig.plot(loss_train)
        loss_fig.plot(loss_validation)
        loss_fig.set_title('model loss')
        loss_fig.set_ylabel('loss')
        loss_fig.set_xlabel('epoch')
        loss_fig.legend(['train', 'validation'], loc='upper left')

        acc_fig = fig.add_subplot(122)
        acc_fig.plot(acc_train)
        acc_fig.plot(acc_validation)
        acc_fig.set_title('model ' + metric_name)
        acc_fig.set_ylabel(metric_name)
        acc_fig.set_xlabel('epoch')
        acc_fig.legend(['train', 'validation'], loc='upper left')
        plt.savefig(img_file_path)

def setup_dataset(args, load_graphs=False):
    ROOT_PATH = args.images 
    assert os.path.exists(ROOT_PATH), "Error: given path does not exist"
    RESIZE_DIM = 512
    # dataset setup
    datasetManager = RCCDatasetManager(ROOT_PATH,
                        resize_dim=RESIZE_DIM,
                        download_dataset=True,
                        standardize_config={'by_patient':args.std_by_patient, 
                                             'by_patient_train_avg_stats_on_test':args.std_by_patient,
                                             'by_single_img_stats_on_test':False}, 
                        load_graphs=load_graphs,
                        img_format='RGB',
                        verbose=True)
    return datasetManager

def setup_augmentation(args):
    augment_params_dict = dict()

    if args.rand_rot:
            augment_params_dict['rotate'] = {'prob':1.0,'angle_range': 30}
           
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
        help="standardize slide images according to the channel statistics",
    )
    parser.add_argument(
        "--std-by-patient",
        type=bool,
        default=False,
        help="compute mean and variance for each 'train' split patient\n and standardize each of their samples by their own statistics: test samples are standardized according to the average mean and pooled variance",
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
        help="random rotations (90,180,270 degrees) data augmentation",
    )
    parser.add_argument(
        "--rand-crop",
        type=bool,
        default=False,
        help="random crop and zoom (keep from 0.7 to 1.0 of the original image ) data augmentation"
    )

    parser.add_argument(
        "--rand-elastic-deform",
        type=bool,
        default=False,
        help="elastic deformation:\n\t\t\t"+\
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