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
from lib.data.utils import str2bool

def main(args):

        # dataset setup
        datasetManager = setup_dataset(args)

        # augmentation setup
        augment_params_dict = setup_augmentation(args)

        augment = len(augment_params_dict) > 0
        # preprocessing setup
        (img_train_transform, img_test_transform), (seg_train_transform, seg_test_transform) = setup_preprocessing(args)
        
        
        # experiment setup
        experiment_mgr = ExperimentManager( datasetManager )


        if args.format == 'rgb':
            in_channels = 3
        else:
            in_channels = 1


        model = models.vgg16_bn(num_classes= 2)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        if args.cross_val:
                    folds = 5
                    stratified_folds = model_selection.RepeatedStratifiedKFold(
                      n_splits=folds, n_repeats=1
                        ).split(datasetManager.sample_dataset.y_labels, datasetManager.sample_dataset.y_labels)


                    train_acc_folds = []
                    val_acc_folds = []
                    test_acc_folds = []

                    train_loss_folds = []
                    val_loss_folds = []
                    test_loss_folds = []


                    best_model = model
                    min_loss = sys.float_info.max



                    for i, (train_index, validation_index) in enumerate(stratified_folds):
                          print(f"Fold {i+1} of {folds}")
                          curr_model = copy.deepcopy(model)
                
                          (train_dataset, validation_dataset, test_dataset), _ = datasetManager.init_train_val_split( 
                                                                img_train_transform = img_train_transform,
                                                            seg_train_transform = seg_train_transform,
                                                              img_test_transform = img_test_transform,
                                                            seg_test_transform = seg_test_transform,
                                                         
                                                                  batch_size=args.batch_size,
                                                                    train_indices = train_index,
                                                                    validation_indices = validation_index,
                                                         
                                                                    train_augment=augment,
                                                                        **augment_params_dict)
                          train_dataloader = DataLoader( train_dataset , batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                          val_dataloader = DataLoader(validation_dataset , batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                          test_dataloader = DataLoader( test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

                          curr_fold_weights_fname = f"fold_{i}_" + args.weights_fname
                          loss_train, loss_validation, acc_train, acc_validation, test_accuracy, model = experiment_mgr.train_convnet(model, learning_rate=args.lr, 
                                                  batch_size=args.batch_size, 
                                                  train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                                                  epochs=args.epochs,
                                                  log_weights_path=args.weights_dir,
                                                  weights_filename=curr_fold_weights_fname,
                                                  augment=augment, augment_params_dict=augment_params_dict,
                                                   verbose=False)
                    
                          train_acc_folds.append(acc_train)
                          val_acc_folds.append(acc_validation)
              

                          train_loss_folds.append(loss_train)
                          val_loss_folds.append(loss_validation)
                         
                          test_acc_folds.append( np.mean([ acc.item() for acc in test_accuracy  ]) )

                          print(f"validation accuracy: {acc_validation[-1]}")
                          if loss_validation[-1] <  min_loss:
                            min_loss = loss_validation[-1]
                            best_model = curr_model
                            torch.save(best_model.state_dict(), os.path.join(args.weights_dir, "best_model_" + args.weights_fname))
                    train_acc_folds_average = np.array(train_acc_folds).mean(axis=0)
                    val_acc_folds_average = np.array(val_acc_folds).mean(axis=0)
                    #test_acc_folds_average = np.array(test_acc_folds).mean(axis=0)

                    train_loss_folds_average = np.array(train_loss_folds).mean(axis=0)
                    val_loss_folds_average = np.array(val_loss_folds).mean(axis=0)
                    #test_loss_folds_average = np.array(test_loss_folds).mean(axis=0)
                    img_file_path = os.path.join(args.weights_dir, args.weights_fname +"_train.png")
                    plot_results(img_file_path, train_loss_folds_average, val_loss_folds_average, train_acc_folds_average, val_acc_folds_average, metric_name='accuracy')
                    boxplot_file_path = os.path.join(args.weights_dir, "test_accuracy_boxplot.png")
                    
                    plt.figure()
                    plt.boxplot(test_acc_folds)
                    plt.savefig(boxplot_file_path)
                    #best_model, train_acc_folds_average, val_acc_folds_average, train_loss_folds_average, val_loss_folds_average, test_acc_folds
        #model =  torch.hub.load('pytorch/vision:v0.9.0', 'vgg', pretrained=False,num_classes=2)
        else:
                    #model = models.vgg16_bn(num_classes= 2)
                    #model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

                    #model = resnet50(pretrained=False, num_classes=2)
                    #model.fc = nn.Linear(in_features=2048, out_features=out_channels, bias=True)
                    #model = torchvision.models.resnet18()#torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
                    #model.fc = nn.Linear(in_features=512, out_features=out_channels, bias=True)
             
                    # experiment run
                    print("CNN training started")
                    loss_train, loss_validation, acc_train, acc_validation, test_accuracy, model = experiment_mgr.train_convnet(model, learning_rate=args.lr, 
                                                              batch_size=args.batch_size, img_train_transform=img_train_transform, seg_train_transform=seg_train_transform,
                                                              img_test_transform=img_test_transform,
                                                              seg_test_transform=seg_test_transform,
                                                              epochs=args.epochs,
                                                              log_weights_path=args.weights_dir,
                                                              weights_filename=args.weights_fname,
                                                              augment=augment, augment_params_dict=augment_params_dict,
                                                               verbose=False)
                    print("CNN training completed")
                    # save results
                    img_file_path = os.path.join(args.weights_dir, args.weights_fname +"_train.png")
                    plot_results(img_file_path, loss_train, loss_validation, acc_train, acc_validation, metric_name='accuracy')


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
            means = gray_mean
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
        type=str2bool,
        default=False,
        help="Perform 5-fold cross-validation: [True,False] (default to False)",
    )





    parser.add_argument(
        "--std",
        type=str2bool,
        default=True,
        help="standardize images according to the channel statistics  [True, False] (default True)  ",
    )
    parser.add_argument(
        "--std-by-patient",
        type=str2bool,
        default=False,
        help=" [True, False] (default to False) compute mean and variance for each 'train' split patient\n and standardize each of their samples by their own statistics: test samples are standardized according to the average mean and pooled variance ",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="rgb",
        help="image format:['rgb','gray'] (default is rgb)",
    )





    parser.add_argument(
        "--rand-rot",
        type=str2bool,
        default=False,
        help="random rotations (90,180,270 degrees) data augmentation,  [True, False] (default to False) ",
    )
    parser.add_argument(
        "--rand-crop",
        type=str2bool,
        default=False,
        help="random crop and zoom (keep from 0.7 to 1.0 of the original image ) data augmentation,  [True, False] (default to False)"
    )

    parser.add_argument(
        "--rand-elastic-deform",
        type=str2bool,
        default=False,
        help=" [True, False] (default to False) elastic deformation:\n\t\t\t"+\
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