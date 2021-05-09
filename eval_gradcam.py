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
from lib.data.gradcam_utils import test_classifier_gradcam

from lib.managers import RCCDatasetManager, ExperimentManager
import numpy as np
import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from torchvision import models


def main(args):

        
        # dataset setup
        datasetManager = setup_dataset(args)

 
        # preprocessing setup
        (img_train_transform, img_test_transform), (seg_train_transform, seg_test_transform) = setup_preprocessing(args)
        
        
        # experiment setup
        experiment_mgr = ExperimentManager( datasetManager )

        
        
        if args.format == 'rgb':
            in_channels = 3
            means = RCCDatasetManager.img_means
            st_deviations = RCCDatasetManager.img_std
        else:
            in_channels = 1
            means = RCCDatasetManager.gray_mean
            st_deviations = RCCDatasetManager.gray_std
      
        model = models.vgg16_bn(num_classes= 2)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        target_layer = model.features[40]

        assert args.vgg_weights_path is not None and os.path.exists(args.vgg_weights_path), "Error: path to the vgg16 weights has not been given or does not exist."

        state_dict = torch.load(args.vgg_weights_path)
        model.load_state_dict(state_dict)
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        model.to(device)

        (train_dataset, validation_dataset, test_dataset), (train_gen, validation_gen, test_gen) = datasetManager.init_train_val_split(0.1, 
                                                batch_size=args.batch_size,
                                                img_train_transform = img_train_transform,
                                                seg_train_transform = seg_train_transform,
                                                img_test_transform = img_test_transform,
                                                seg_test_transform = seg_test_transform,
                                                train_augment=False,
                                                    )


        train_dataloader = DataLoader( train_dataset , batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_dataloader = DataLoader(validation_dataset , batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        test_dataloader = DataLoader( test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)  

        


        gradcam_logs, figures = test_classifier_gradcam(model, target_layer,val_dataloader, 
                                     means, 
                                     st_deviations,
                                     gradcam_method= 'gradcam++')

        save_gradcams(gradcam_logs, args)

def save_gradcams(gradcam_logs, args):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    for i, gradcam_log in enumerate(gradcam_logs):
        path = os.path.join(args.logs_dir, "gradcam_log_{}.png".format(i) )
        description = f"image path: {gradcam_log.img_path}\n mask path: {gradcam_log.seg_path}"
        gradcam_log.figure.text(.5, .001, description, ha='center')
        gradcam_log.figure.savefig(path)     

    reference_file =  os.path.join(args.logs_dir, "reference_gradcam_logs.txt" )
    with open(reference_file, "w+") as outfile:
        outfile.write("id;image_path;mask_path\n")
        for i, gradcam_log in enumerate(gradcam_logs):
            outfile.write(f"{i};{gradcam_log.img_path};{gradcam_log.seg_path}\n" )

def setup_dataset(args):
    ROOT_PATH = args.images 
    #assert os.path.exists(ROOT_PATH), "Error: given path does not exist"
   
    # dataset setup
    datasetManager = RCCDatasetManager(ROOT_PATH,
                        download_dataset=True,
                        standardize_config={'by_patient': False, 
                                             'by_patient_train_avg_stats_on_test':False,
                                             'by_single_img_stats_on_test':False}, 
                        load_graphs=False,
                        verbose=True)
    return datasetManager

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
    if args.format == 'gray':
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
        description="CNN gradcam plotting (de-standardization by-patient not supported)"
    )

    parser.add_argument(
        "--vgg-weights-path",
        type=str,
        default=None,
        help="path to the weights of the vgg16 classifier",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="input batch size for training (default: 4)",
    )
  




    parser.add_argument(
        "--std",
        type=bool,
        default=True,
        help="standardize images according to the channel statistics",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="rgb",
        help="image format:['rgb','gray'] (default is rgb)",
    )





    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )


    parser.add_argument(
        "--logs-dir", type=str, default="./logs", help="folder to save weights"
    )

    parser.add_argument(
        "--images", type=str, default="./vascular_segmentation", help="root folder with train and test folders"
    )


    args = parser.parse_args()
    main(args)