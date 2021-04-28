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




from plot_utils import ipyDisplay
from graph_utils import estimate_graph_statistics
from graph_utils import GraphItem, ConnectedComponentCV2, GraphItemLogs
from image_utils import is_on, remove_isolated
from processing_utils import UnNormalize
from processing_utils import ToGraphTransform
from neural_nets import UNet
from managers import RCCDatasetManager, ExperimentManager




def main(args):


        # dataset setup
        if args.dataset == 'unet':
            load_graphs = False
            (img_train_transform, img_test_transform), (seg_train_transform, seg_test_transform) = setup_preprocessing(args)
        else:
            load_graphs = True
        datasetManager = setup_dataset(args, load_graphs)


        # experiment setup
        experiment_mgr = ExperimentManager( datasetManager )




        sample_dataset_graph_items, sample_dataset_graph_labels = None, None
        out_of_sample_dataset_graph_items, out_of_sample_dataset_graph_labels = None, None

        if args.dataset == 'unet':
                if args.format == 'rgb':
                    in_channels = 3
                else:
                    in_channels = 1
                out_channels = 2
                init_features = 64

                resize_dim = 512
                model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)

                if args.state_dict_path is not None:
                        #assert args.state_dict_path is not None, "Error: must provide the path to the trained unet model weights"
                        state_dict = torch.load(args.state_dict_path)
                        unet.load_state_dict(state_dict)
                        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
                        unet.to(device)
                else:

                        log_weights_path = "./experiment_log_folder"
                        weights_filename = "unet_weights.pt"
                        augment_params_dict = {'resized_crop': None,
                        'rotate' : {'prob':1.0,'angle_range': 30},
                          'gauss_blur' : None,
                          'elastic_deform': None
                            }
                        kwargs_dict = {'learning_rate':args.lr, 'batch_size':4, 'img_train_transform':img_train_transform, 'seg_train_transform':seg_train_transform,
                                                                  'img_test_transform':img_test_transform,
                                                                  'seg_test_transform':seg_test_transform,
                                                                  'epochs':20,
                                                                  'log_weights_path':log_weights_path,
                                                                  'weights_filename':weights_filename,
                                                                  'augment':True, 
                                                                  'augment_params_dict':augment_params_dict,
                                                                  'verbose_loss_acc': True}

                        print("UNet training started: storing results in '{}' ".format(log_weights_path))						
                        loss_train, loss_validation, IOU_train, IOU_validation, IOU_test, unet, segmentation_progress, segmentation_progress_pred, segmentation_progress_true_mask = experiment_mgr.train_unet(unet, **kwargs_dict)
                        print("Unet training completed\n")
                        plot_results(args, loss_train, loss_validation, IOU_train, IOU_validation, metric_name='IoU')
                        plot_segmentation_progress(segmentation_progress, log_weights_path)




                mask_pred_kwargs_dict = {'batch_size': 4, 
                                                          'img_train_transform':img_train_transform, 
                                                          'seg_train_transform':seg_train_transform,
                                                          'img_test_transform':img_test_transform,
                                                          'seg_test_transform':seg_test_transform,
                                                          'validation_split_size': 0.1}
                print("Creating unet segmentation dataset")	
                (train_pred_graphs, train_pred_graph_labels), (val_pred_graphs, val_pred_graph_labels), (test_pred_graphs, test_pred_graph_labels) = \
                                                    experiment_mgr.get_segmented_masks_graph_items(unet,  
                                                      **mask_pred_kwargs_dict)
                print("Created segmentation dataset")

                pred_segmentation_graph_dataset = train_pred_graphs + val_pred_graphs
                pred_segmentation_graph_labels = train_pred_graph_labels + val_pred_graph_labels
                sample_dataset_graph_items = pred_segmentation_graph_dataset
                sample_dataset_graph_labels  = pred_segmentation_graph_labels
                out_of_sample_dataset_graph_items = test_pred_graphs
                out_of_sample_dataset_graph_labels = test_pred_graph_labels

        else:				
                sample_dataset_graph_items, sample_dataset_graph_labels = datasetManager.sample_dataset_graph_items,datasetManager.sample_dataset_graph_labels
                out_of_sample_dataset_graph_items, out_of_sample_dataset_graph_labels = datasetManager.out_of_sample_dataset_graph_items,datasetManager.out_of_sample_dataset_graph_labels









		if args.gcn_type == 'sg':
				sg_train, train_labels = RCCDatasetManager.make_stellargraph_dataset(sample_dataset_graph_items, sample_dataset_graph_labels,
				                                  loading_prompt_string=None)
				sg_test, test_labels = RCCDatasetManager.make_stellargraph_dataset(out_of_sample_dataset_graph_items,  out_of_sample_dataset_graph_labels,
				                                  loading_prompt_string=None)

				print("GCN training started")
				best_model, train_histories, train_acc_folds, val_acc_folds, test_acc_folds = experiment_mgr.train_sg_gcn( validation_size=0.1,    
				                      train_sg_graphs=sg_train, train_graphs_labels=train_labels,
				                      val_sg_graphs=None, val_graphs_labels=None,  
				                      test_sg_graphs=sg_test, test_graphs_labels=test_labels,
				                      cross_validation=True,
				                      batch_size=args.batch_size,
				                      learning_rate=args.lr,
				                      epochs=args.epochs,
				                      folds = 5,
				                      n_repeats = 1,
				                      verbose=True,verbose_epochs_accuracy=False)
				print("GCN training completed")

		else:
				torch_train, train_labels = RCCDatasetManager.make_torch_graph_dataset(sample_dataset_graph_items, sample_dataset_graph_labels,
				                                  loading_prompt_string=None)
				torch_test, test_labels = RCCDatasetManager.make_torch_graph_dataset(out_of_sample_dataset_graph_items, out_of_sample_dataset_graph_labels,
				                                  loading_prompt_string=None)

				gcn = GCN(hidden_channels=64, num_node_features=3, num_classes=2, dropout=0.2)
				print("GCN training started")
				best_model, train_acc_folds, val_acc_folds, test_acc_folds = experiment_mgr.train_torch_gcn( gcn, validation_size=0.1,   
				                      train_torch_graphs=torch_train, train_graphs_labels=train_labels,
				                      val_torch_graphs=None, val_graphs_labels=None,  
				                      test_torch_graphs=torch_test, test_graphs_labels=test_labels,
				                      cross_validation=True,
				                      batch_size=args.batch_size,
				                      learning_rate=args.lr,
				                      epochs=args.epochs,
				                      folds = 5,
				                      n_repeats = 1,
				                      verbose=True,verbose_epochs_accuracy=False)
				print("GCN training completed")








        
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







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GCN experiment"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )


	parser.add_argument(
        "--gcn-type",
        type=str,
        default="torch",
        help="gcn library that implementst the GCN: ['torch', 'sg']\n\t\t 'torch' is the torch geometric library"+\
        														"\n\t\t 'sg' is the stellargraph library",
    )





    parser.add_argument(
        "--dataset",
        type=str,
        default="unet",
        help="dataset on which the GCN is trained: ['unet', 'gt']\n\t\t 'unet' is the graph dataset created from predicted segmentation masks"+\
        														"\n\t\t 'gt' is the graph dataset created from ground truth masks",
    )
    parser.add_argument(
        "--state-dict-path",
        type=str,
        default=None,
        help="(ONLY VALID IF unet dataset) Path to the weights of the trained unet model, else a unet will be trained with:\n\t\t lr=0.00005, batch_size=4, epochs=20"+\
        														"\n\t\t all remaining parameters are configurable by means of the available command line arguments" 
    )





    parser.add_argument(
        "--std",
        type=bool,
        default=True,
        help="(ONLY VALID IF unet dataset) standardize slide images according to the channel statistics",
    )
    parser.add_argument(
        "--std-by-patient",
        type=bool,
        default=False,
        help="(ONLY VALID IF unet dataset) compute mean and variance for each 'train' split patient\n and standardize each of their samples by their own statistics: test samples are standardized according to the average mean and pooled variance",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="rgb",
        help="(ONLY VALID IF unet dataset) slide image format:['rgb','gray'] (default is rgb)",
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