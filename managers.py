import os
import cv2
import numpy as np
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
import random
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.keras.callbacks import EarlyStopping

from utils import DriveDownloader, recursive_visit
from datasets import RCCStorage, RCCImageSubset, RCCImageDataset, CropDataset
from patient_utils import Patient, find_position, map_patients_to_ids, reduce_bykey
from itertools import groupby
from functools import  reduce
from operator import itemgetter
from processing_utils import ToGraphTransform
from image_utils import binarize_to_numpy
from torch_geometric.data import Data
from train_segmentation_methods import train_segmentation_model, train_crop_segmentation_model,  validate_segmentation
from train_clf_methods import train_classifier, test_classifier
from train_graph_nn import train_test_torch_gcn
from train_sg_graph_nn import create_graph_classification_model
from metrics import IoU
from image_utils import read_image


class RCCDatasetManager(object):
    
    X_TRAIN = "Train"
    X_TEST = "Test"
    PIL_GRAYSCALE = 'L'
    PIL_RGB  = 'RGB'
    def __init__(self, 
                 root_path,
                 download_dataset=False,
                 standardize_config={'by_patient':False,   # compute mean and standard dev statistics for each of the patient sample groups and standardize the respective images by these statistics
                                                      # for each patient annotation group of images, a mean and a standard deviation are computed and these are used to standardize the 
                                                      #   respective annotation group. These values are then stored for later de-standardization or computation of the overall  mean and 
                                                          # standard deviation, by taking the average of the means and the square root of the average of the Variances.
                                                          # IF by_patient is TRUE, behavior on test is:
                                                          #            - to standardize each of the TEST images by the average mean and  square root of average of the variances, computed on the train patients
                                                          #                     with ###########by_patient_train_stats_avg_on_test=True################
                                                          #            - to standardize each of the test image by its own mean and std as if each test image comes from its own patient.
                                                          #               The logic behind this procedure is that i want to avoid too large differences in color intensity of
                                                          #                 different patients' biopsy material: inside one cancer class, colors vary widely from a patient to another
                                                          #                 but the overall features that distinguish one class from the other seem, from observation, more dependent on geometrical
                                                          #                 features.
                                     
                                     'by_patient_train_avg_stats_on_test':False,# if this is true, each test image is standardized according to 
                                                                          # the average mean and average variance computed on the training set  
                                     
                                     'by_single_img_stats_on_test':False},  # if true, a test sample is standardized according to its own provided information
                 
                 load_graphs = True,
                img_format = 'RGB',                  
                resize_dim=512,
                verbose=False):

        self.load_graphs = load_graphs
        



        if download_dataset == True:
          drive_file_id = '1jz9lC2j4CfH9oF3CRVHHLs8C5k6yw46i'
          tmp_destination = './dataset.zip'
          target_directory = './rccdataset'
          target_path = os.path.join(target_directory, "vascular_segmentation")
          if (not os.path.exists(target_path) and root_path is None) or \
                              ( root_path is not None and ("vascular_segmentation" not in root_path or not os.path.exists(root_path))  ):
            # if root path is none and the downloaded dataset folder does not exist OR 
            # root path is NOT none and (it does not contain vascular segmentation OR it does not exist)
            # we download the dataset on the folder
            self.__download_dataset__(drive_file_id, tmp_destination, target_directory)
          self.root_path = target_path
        else:
          assert root_path is not None and "vascular_segmentation" in root_path, "Error: root path must point to the 'vascular_segmentation' folder"
          self.root_path = root_path

     
        self.resize_dim = resize_dim
        self.img_format = img_format

        self.verbose = verbose

        

        
        # mean and standard deviation of colors in the RCC train dataset
        # CHANNEL ORDER IS BGR (cv2 imread convention)
        self.img_means = (0.7297678 , 0.4809845 , 0.65480953)
        var = (0.02753073, 0.04772836, 0.02944909)
        self.img_std = np.sqrt(var)

        
        
        if 'by_patient' in standardize_config and standardize_config['by_patient'] is True:
            # if by patient is true the other two settings for the test set must be either true or false but not both true or both false (invalid standardization configurations)
            assert standardize_config.get('by_patient_train_avg_stats_on_test', False) ^\
                   standardize_config.get('by_single_img_stats_on_test', False), "pre-standardizing by patient requires to setup also a test set preprocessing standardization procedure for the patients"
        else:
            assert standardize_config.get('by_patient_train_avg_stats_on_test', False) == False and \
                   standardize_config.get('by_single_img_stats_on_test', False)== False, "without pre-standardizaton on train patients annotation groups, test patients cannot be standardized"
      
      
   
        # components of the rcc dataset are loaded into a storage
        # train_avg_mean and train_avg_std are computed in this call (whenever a train dataset is loaded)
        self.rccTrainStorage = self.__load_rcc_dataset__(
                     partition=RCCDatasetManager.X_TRAIN, 
                     **standardize_config
                    )

        self.rccTestStorage = self.__load_rcc_dataset__(
                     partition=RCCDatasetManager.X_TEST,
                     **standardize_config
                    )
   
        
        # IMAGE DATASET CREATION
        # torch Datasets are initialized with the components that have previously been loaded though load_rcc_dataset
        self.sample_dataset = RCCImageDataset( 
                 self.rccTrainStorage,
              
                resize_dim=self.resize_dim,
                img_format=self.img_format,
                img_color_mapping=cv2.IMREAD_COLOR,  #self.img_color_mapping,
                seg_color_mapping=cv2.IMREAD_GRAYSCALE #self.seg_color_mapping
                )
        self.out_of_sample_dataset = RCCImageDataset( 
                 self.rccTestStorage,
         
                resize_dim=self.resize_dim,
                img_format=self.img_format,
                img_color_mapping=cv2.IMREAD_COLOR,  #self.img_color_mapping,
                seg_color_mapping=cv2.IMREAD_GRAYSCALE #self.seg_color_mapping
                )
        
        self.N_sample_dataset = len(self.sample_dataset)
        self.N_out_of_sample_dataset = len(self.out_of_sample_dataset)
    
        
        # GRAPH DATASET CREATION
        dataset_loading_prompt = None
        test_dataset_loading_prompt = None
        if self.verbose:
            dataset_loading_prompt = str("Loading " + RCCDatasetManager.X_TRAIN + " partition graphs\n")
            test_dataset_loading_prompt = str("Loading " +  RCCDatasetManager.X_TEST +  " partition graphs\n") 

        if self.load_graphs:
          self.sample_dataset_graph_items, self.sample_dataset_graph_labels = RCCDatasetManager.load_graph_items(self.rccTrainStorage.X_seg,
                                                                                                      self.rccTrainStorage.y_numeric,
                                                                                                      self.resize_dim,
                                                                                                      self.img_format,
                                                                                                    cv2.IMREAD_COLOR, 
                                                                                                    loading_prompt_string= dataset_loading_prompt)
          self.out_of_sample_dataset_graph_items, self.out_of_sample_dataset_graph_labels = RCCDatasetManager.load_graph_items(self.rccTestStorage.X_seg,
                                                                                              self.rccTestStorage.y_numeric,
                                                                                              self.resize_dim,
                                                                                              self.img_format,
                                                                                              cv2.IMREAD_COLOR,
                                                                                              loading_prompt_string= test_dataset_loading_prompt)
          

          self.dataset_graphs, self.dataset_graph_labels = RCCDatasetManager.make_stellargraph_dataset(self.sample_dataset_graph_items,
                                                                                                      self.sample_dataset_graph_labels)
          
          self.test_graphs, self.test_graph_labels = RCCDatasetManager.make_stellargraph_dataset(self.out_of_sample_dataset_graph_items,
                                                                                                self.out_of_sample_dataset_graph_labels)
          
          self.dataset_generator = PaddedGraphGenerator(graphs=self.dataset_graphs)
          self.test_generator = PaddedGraphGenerator(graphs=self.test_graphs)  
        
        



    def __get_generators__(self, train_index, validation_index, test_index, graph_labels, test_graph_labels, batch_size):
        train_gen = self.dataset_generator.flow(
            train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
        )
        validation_gen = self.dataset_generator.flow(
            validation_index, targets=graph_labels.iloc[validation_index].values, batch_size=batch_size
        )
        test_gen = self.test_generator.flow(
            test_index, targets=test_graph_labels.iloc[test_index].values, batch_size=batch_size
        )

        return train_gen, validation_gen, test_gen
    

        
    def init_train_val_split(self, validation_size, 
                                    img_train_transform = None,
                                seg_train_transform = None,
                                  img_test_transform = None,
                                seg_test_transform = None,
                             
                                      batch_size=16,
                                        train_indices = None,
                                        validation_indices = None,
                             
                                        train_augment=False,
                                            resized_crop=None,
                                            rotate=None,
                                            gauss_blur=None,
                                            elastic_deform=None):
        if train_indices is None:
          train_indices, validation_indices = train_test_split(range(self.N_sample_dataset),
                                                      test_size=validation_size,
                                                      stratify=self.sample_dataset.y_labels)
        
        test_indices = [i for i in range(self.N_out_of_sample_dataset)]
        

        train_dataset = RCCImageSubset(self.sample_dataset, train_indices, 
                                            img_train_transform,
                                            seg_train_transform,
                                            augment=train_augment,
                                            resized_crop=resized_crop,
                                            rotate=rotate,
                                            gauss_blur=gauss_blur,
                                            elastic_deform=elastic_deform)
        
        validation_dataset = RCCImageSubset(self.sample_dataset, validation_indices,
                                            img_test_transform,
                                            seg_test_transform)
        
        test_dataset = RCCImageSubset(self.out_of_sample_dataset, test_indices,
                                            img_test_transform,
                                            seg_test_transform)

   
        train_gen, validation_gen, test_gen = None,None,None
        if self.load_graphs:
          train_gen, validation_gen, test_gen = self.__get_generators__(train_indices, validation_indices, test_indices,
                                                                 self.dataset_graph_labels, self.test_graph_labels, batch_size)
        
        
        
        return (train_dataset, validation_dataset, test_dataset), (train_gen, validation_gen, test_gen)

  
    def get_torch_geom_dataset(self, validation_size, 
                                        train_indices = None,
                                        validation_indices = None):
      torch_dataset_graphs, torch_dataset_graph_labels = RCCDatasetManager.make_torch_graph_dataset(self.sample_dataset_graph_items, self.sample_dataset_graph_labels,
                                                      loading_prompt_string=None)
      torch_test_graphs, test_graph_labels = RCCDatasetManager.make_torch_graph_dataset(self.out_of_sample_dataset_graph_items, self.out_of_sample_dataset_graph_labels,
                                                      loading_prompt_string=None)

      if train_indices is None:
          train_indices, validation_indices = train_test_split(range(len(torch_dataset_graph_labels)),
                                                      test_size=validation_size,
                                                      stratify=torch_dataset_graph_labels)
      test_indices = [i for i in range(self.N_out_of_sample_dataset)]

      X_torch_train = []
      y_torch_train = []

      X_torch_validation = []
      y_torch_validation = []

      X_torch_test = []
      y_torch_test = []
      for idx in train_indices:
        X_torch_train.append(torch_dataset_graphs[idx])
        y_torch_train.append(train_graphs_labels[idx])
      for idx in validation_indices:
        X_torch_validation.append(torch_dataset_graphs[idx])
        y_torch_validation.append(train_graphs_labels[idx])
      for idx in test_indices:
        X_torch_test.append(torch_test_graphs[idx])
        y_torch_test.append(test_graphs_labels[idx])
                      
      return  (X_torch_train,y_torch_train), (X_torch_validation,y_torch_validation), (X_torch_test,y_torch_test)
    
    def __load_rcc_dataset__(self,
                     partition=X_TRAIN,

                     # patient sample group standardization happens while loading training set 
                     by_patient=False, # train partition only setting
                     by_patient_train_avg_stats_on_test=False,  # test partition only setting
                     by_single_img_stats_on_test=False,         # test partition only setting
                    ):
      
        labels_dict = dict()
        id_to_labels = []

        num_labels = 0
        X_img = []
        X_seg = []
        y_cancer_ids = []
        y_labels = []


        assert partition == RCCDatasetManager.X_TRAIN or partition == RCCDatasetManager.X_TEST, "Error: dataset split must either be 'Train' or 'Test' "
        
        
        folder_path = os.path.join(self.root_path, partition)

        img_paths = []
        seg_paths =[]

        # pattern required to identify the attributes of a given file
        # -sample type= segmentation or image
        # -id = expressed in x[0-9]_y[0-9]
        # -file type = {.png} is the only one considered but also .roi exist
        train_pattern = re.compile('([\w\W]+)([a-zA-Z][0-9]+_[a-zA-Z][0-9]+)\.([a-zA-Z0-9\.]+)$')
        test_pattern = re.compile('([\w\W]+_)([a-zA-Z0-9]+)\.([a-zA-Z0-9\.]+)$')



        replacement_string = None


        filename_pattern = None
        if partition == RCCDatasetManager.X_TEST :
            filename_pattern = test_pattern
        else:
            filename_pattern = train_pattern




        samples = []

        folder_iterable = os.listdir(folder_path)
        if self.verbose:
            print("Scanning " + partition + " dataset directories")
            folder_iterable = tqdm(folder_iterable, leave=True,position=0 )


        # first pass to cleanup images or segmentations that have no matching counterpart 
        # and to an easily iterable set of samples 
        for directory in folder_iterable:
            if directory not in labels_dict:
                    labels_dict[directory] = num_labels
                    id_to_labels.append(directory)
                    # pRCC or cRCC
                    category = directory
                    if partition == RCCDatasetManager.X_TEST:
                        replacement_string = "{}_{}_".format(category, "img")
                    else:
                        replacement_string = "{}_".format("crop")

                    curr_path = os.path.join(folder_path, directory)
                    counter = recursive_visit(curr_path,
                                                samples, 
                                                filename_pattern,
                                                replacement_string,0)
                    # all items that have been read are in the same folder that represents their category 
                    y_cancer_ids.extend([num_labels for _ in range(counter )])
                    y_labels.extend([category for _ in range(counter)])
                    num_labels += 1
                    


        #sample_iterable = samples
        #if self.verbose:
        #    print("Loading " + partition + " dataset")
        #    sample_iterable = tqdm(samples,total=len(samples), position=0, leave=True)
        #pbar = tqdm(total=100)

        if partition == RCCDatasetManager.X_TRAIN:

              # setup for mapping image paths to the respective patient and respective subgroup of annotations 
              # each file is mapped to a patient with the respective cancer type, membership dataset(train or test)
              # and patientid
              patients = list(map(map_patients_to_ids, samples  )  ) 
              # a REDUCE operation is applied to each emitted key,value pair (patientid,patient)
              # in order to group the samples of the same patient in a single object
              patient_sample_dataset = [reduce(reduce_bykey, group)
                    for _, group in groupby(patients, key=itemgetter(0) )  ]


              count = 0
              img_path_samples = [  item[0] for item in samples ]
              seg_path_samples = [  item[1] for item in samples ]

              found_img_paths = []
              found_seg_paths = []
              for patientid, patient in patient_sample_dataset:
                for p_samplekey, p_sample_list in patient.sample_groups.items():
                  count += len(p_sample_list)
                  found_img_paths.extend( [  item[0] for item in p_sample_list ] )
                  found_seg_paths.extend( [  item[1] for item in p_sample_list ] )

              for curr_img_path, curr_seg_path in zip(img_path_samples, seg_path_samples):
                assert curr_img_path in found_img_paths, f"Error: missing image path {curr_img_path} from the built patient data structure"
                assert curr_seg_path in found_seg_paths, f"Error: missing segmentation path {curr_seg_path} from the built patient data structure"
         

              img_paths = []
              seg_paths = []

              img_to_patient_map = dict()
              seg_to_patient_map = dict()

              img_to_sample_group_statistics = dict()

              if self.verbose:
                  print("Loading " + partition + " dataset")
                  progressbar = tqdm(total=len(samples), position=0, leave=True)
              
              
              train_means = []
              train_variances = []
              for patientid, patient in patient_sample_dataset:
                curr_batch_images = []
                curr_batch_segmentation_masks = []
                tmp_paths_list = []
                for p_samplekey, p_sample_list in patient.sample_groups.items():
                  for img_path, seg_path in p_sample_list:
                    img_image = read_image(img_path, self.resize_dim, cv2.IMREAD_COLOR)/255.  #read_image(img_path, self.resize_dim, self.img_color_mapping)
                    seg_image = read_image(seg_path, self.resize_dim, cv2.IMREAD_GRAYSCALE)

                    curr_batch_images.append(img_image)
                    curr_batch_segmentation_masks.append(seg_image)

                    img_paths.append(img_path)
                    seg_paths.append(seg_path)

                    # for logging batch mean and std after the full patient data has been processed
                    tmp_paths_list.append( (img_path, seg_path) )

                    X_seg.append(seg_image)


                    img_to_patient_map[img_path] = patient
                    seg_to_patient_map[seg_path] = patient
                      
                    if self.verbose:
                      progressbar.update(1)

                    
                batch_images_tensor = np.array(curr_batch_images) 
                if by_patient:
                    batch_mean = batch_images_tensor.mean(axis=(0,1,2))
                    batch_std = batch_images_tensor.std(axis=(0,1,2))
                else:
                    # mean and std such that no standardization happens
                    batch_mean = np.array([0., 0., 0.] )
                    batch_std = np.array([1. , 1., 1.] )

                
                norm_batch_images_tensor = (batch_images_tensor - batch_mean)/ batch_std

                curr_N = batch_images_tensor.shape[0]
                train_means.append((batch_mean,curr_N)  )
                train_variances.append((batch_std**2, curr_N ) )

                for idx in range(norm_batch_images_tensor.shape[0]):  
                    X_img.append(norm_batch_images_tensor[idx,...])
                    

                # register the computed mean and standard deviation statistics
                # that have been used to standardize each image in the current patient batch
                for img_path, seg_path in tmp_paths_list:
                    img_to_sample_group_statistics[img_path] = (batch_mean, batch_std)

              N = sum( [ N_batch for batch_mean, N_batch in train_means ]  )
              K = len(patient_sample_dataset)
              train_avg_mean = sum( [  (N_batch/N)*batch_mean for batch_mean, N_batch in train_means ] )
              train_avg_std = np.sqrt(1./(N-K)*sum(  [  (N_batch-1) *batch_variance   for batch_variance, N_batch in train_variances ]  ))
              self.train_avg_mean = train_avg_mean 
              self.train_avg_std = train_avg_std
              if self.verbose:
                progressbar.close()

              rccStorage = RCCStorage( X_img, X_seg,
                                                  y_labels, y_cancer_ids,
                                                  id_to_labels, labels_dict,
                                                  img_paths, seg_paths,
                                                img_to_patient_map,seg_to_patient_map,
                                            img_to_sample_group_statistics)    
        else:# else partition == X_TEST
                  # TEST partition is not as standard as the train, so a different elaboration is required

                  img_paths = []
                  seg_paths = []

                  img_to_patient_map = dict()
                  seg_to_patient_map = dict()

                  img_to_sample_group_statistics = dict()

                  curr_batch_images = []
                  curr_batch_segmentation_masks = []

                  if self.verbose:
                    print("Loading " + partition + " dataset")
                    progressbar = tqdm(total=len(samples), position=0, leave=True)

                  for img_path, seg_path in samples:
                    img_image = read_image(img_path, self.resize_dim, cv2.IMREAD_COLOR)/255.
                    seg_image = read_image(seg_path, self.resize_dim, cv2.IMREAD_GRAYSCALE)

                    curr_batch_images.append(img_image)
                    curr_batch_segmentation_masks.append(seg_image)

                    img_paths.append(img_path)
                    seg_paths.append(seg_path)

                    X_seg.append(seg_image)




                    train_str = "Train"
                    test_str = "Test"
                    train_information_start = find_position(train_str, img_path)
                    test_information_start = find_position(test_str, img_path)
                    position = max(train_information_start, test_information_start)

                    assert position >= 0 , "Information has not been found in text"
                    img_path_shortened = img_path[position:]
                    seg_path_shortened = seg_path[position:] 

                    img_path_data = img_path_shortened.split("/")
                    seg_path_data = seg_path_shortened.split("/")

                    # train/test
                    sample_split_location = img_path_data[0]
                    # category 
                    sample_label = img_path_data[1]
                    # patientid 
                    import hashlib
                    import random

                    # id is not present in the test set so it is made up from the file name and a random num
                    num = random.randint(0, 10000000000)
                    to_sha_str = f"{img_path_data[2]}_{num}"
                    d = hashlib.sha1(to_sha_str.encode())
           
                    patientid = d.hexdigest()
                    
                    img_filename = img_path_data[-1]
                    seg_filename = seg_path_data[-1] 

                    patient = Patient(patientid, sample_label, sample_split_location)
                    components_of_the_key = img_path_data[2:-1]  # key is composed by patientid + any of the following folders to avoid conflicts due to equal subfolder names
                    sample_key = "/".join( components_of_the_key  )
                    
                    patient.add_sample(img_path, seg_path,  sample_key)





                    img_to_patient_map[img_path] = patient
                    seg_to_patient_map[seg_path] = patient

                    if self.verbose:
                      progressbar.update(1)

                  if self.verbose:
                    progressbar.close()

                  batch_images_tensor = np.array(curr_batch_images) 

                  if by_patient_train_avg_stats_on_test:
                    mean = self.train_avg_mean #batch_images_tensor.mean(axis=(0,1,2))
                    std = self.train_avg_std #batch_images_tensor.std(axis=(0,1,2))
                    norm_batch_images_tensor =  (batch_images_tensor - mean)/ std
                  else:
                    mean = np.array([0., 0. ,0.])
                    std = np.array([1., 1., 1.])
                    norm_batch_images_tensor = batch_images_tensor
                  
                  #if standardize_from train is true, a test image will be standardized according to the statistics of the training set
                  # else if standardize_values is set, it will be standardized according to its own values
                  # else it will be left as it is
                  # An test image will only be standardized either by the average train statistics 
                  # or by its own values, but not both during the loading phase 
                  for idx in range(norm_batch_images_tensor.shape[0]):
                    
                    img_path,seg_path = samples[idx]
                    
                    tensor = norm_batch_images_tensor[idx,...]
                    if by_single_img_stats_on_test:
                      mean = tensor.mean(axis=(0,1))
                      std = tensor.std(axis=(0,1))
                      tensor = (tensor - mean)/(std)
                    
                    X_img.append(tensor)

                    img_to_sample_group_statistics[img_path] = (mean, std) 
                    

                  # register the computed mean and standard deviation statistics
                  # that have been used to standardize each image in the current patient batch


                  rccStorage = RCCStorage( X_img, X_seg,
                                                  y_labels, y_cancer_ids,
                                                  id_to_labels, labels_dict,
                                                  img_paths, seg_paths,
                                                img_to_patient_map,seg_to_patient_map,
                                            img_to_sample_group_statistics)  


        return rccStorage


    
    @staticmethod
    def load_graph_items(X_seg, y_labels,
                                  resize_dim=512,
                                  img_format='RGB',
                                  seg_color_mapping=cv2.IMREAD_GRAYSCALE,
                                  loading_prompt_string=None):
        '''
        returns graph items from segmentation masks
        MUST specify color mapping of the segmentation masks (cv2.IMREAD_COLOR if colored)
        OR cv2.IMREAD_GRAYSCALE if already grayscaled
        Image format specifies what's the sequence of the color channels if any
        (dataset manager convention is RGB)
        '''
        graphs = []
        graph_labels = []
        graph_transformer = ToGraphTransform(SQUARE_IMAGE_SIZE=resize_dim)
        
        
        dataset_iterable = zip( X_seg, y_labels )
        if loading_prompt_string is not None:
            print(loading_prompt_string)
            tot_iters = len(X_seg)
            dataset_iterable = tqdm(dataset_iterable, total=tot_iters,leave=True,position=0  )
        for seg, label in dataset_iterable:
       
            binarized = binarize_to_numpy(seg, seg_color_mapping, img_format)

           
            graph_item = graph_transformer.graph_transform(binarized)
            graphs.append(graph_item)
            graph_labels.append(label)
            
            
      
        return graphs, graph_labels 

    @staticmethod
    def make_stellargraph_dataset(graph_items,
                                  graph_labels,
                                  loading_prompt_string=None):
        sg_graphs = []
        sg_graph_labels = []
    
        dataset_iterable = zip( graph_items, graph_labels )
        if loading_prompt_string is not None:
            print(loading_prompt_string)
            tot_iters = len(graph_items)
            dataset_iterable = tqdm(dataset_iterable, total=tot_iters,leave=True,position=0  )
            
        
        
        for graph_item, label in dataset_iterable:
            stellar_graph = graph_item.stellar_graph
            sg_graphs.append(stellar_graph)
            sg_graph_labels.append(label)
       
            
        sg_graph_labels_series = pd.Series(sg_graph_labels)
        sg_graph_labels_ohe = pd.get_dummies(sg_graph_labels_series)
        return sg_graphs, sg_graph_labels_ohe #sg_graph_labels_dummies 

    @staticmethod
    def make_torch_graph_dataset(graph_items,
                                  graph_labels,
                                  loading_prompt_string=None):
        torch_graphs = []
        torch_graph_labels = []
    
        dataset_iterable = zip( graph_items, graph_labels )
        if loading_prompt_string is not None:
            print(loading_prompt_string)
            tot_iters = len(graph_items)
            dataset_iterable = tqdm(dataset_iterable, total=tot_iters, leave=True,position=0 )
            
        
        
        for graph_item, label in dataset_iterable:
            torch_graph_data = graph_item.torch_geom_data
            
            #torch_graph_data.y = torch.Tensor(label)
            torch_graph_item = Data(x=torch_graph_data.x, edge_index=torch_graph_data.edge_index, y=torch.tensor(label,dtype=torch.long))
            torch_graphs.append(torch_graph_item)
            torch_graph_labels.append(label)

        
        return torch_graphs, torch_graph_labels

    def __download_dataset__(self, drive_file_id, tmp_destination, target_directory):
      #drive_location = "https://drive.google.com/file/d/1jz9lC2j4CfH9oF3CRVHHLs8C5k6yw46i/view?usp=sharing"
      downloader = DriveDownloader()
      downloader.download_file_from_google_drive(drive_file_id, tmp_destination)
      downloader.extract_zip(tmp_destination, target_directory)


"""# RCC Experiment Manager definition"""

class ExperimentManager(object):

  def __init__(self, datasetManager ):
      self.img_means = (0.7297678 , 0.4809845 , 0.65480953)
      self.var = (0.02753073, 0.04772836, 0.02944909)
      self.img_std = np.sqrt(self.var)
      self.gray_mean = (0.5709)
      self.gray_std = np.sqrt(0.0433)

      # for 512x512 images
      self.graph_cc_mean = np.array([ 12.66008916 , 12.10070891, 110.54037857])
      self.graph_cc_std = np.array([ 5.04678162  ,4.65336656 ,72.05804868])

      self.datasetManager = datasetManager


    
  def train_unet(self,model, learning_rate=0.0001, in_channels=3, epochs=20, 
                        train_dataloader=None, val_dataloader=None, test_dataloader=None, validation_split_size=0.1, batch_size=4,
                      img_train_transform=None,
                      seg_train_transform=None,
                      img_test_transform=None,
                      seg_test_transform=None,
                      log_weights_path="./log_dir",
                      weights_filename="fname.pt",
                      augment=False,
                      augment_params_dict={},
                      verbose=True,
                      verbose_loss_acc=True):

      if train_dataloader == None or val_dataloader == None or test_dataloader == None :
        assert val_dataloader == None and val_dataloader == None and test_dataloader == None  , "Error: if any of the dataloaders is empty, all three must be set to None"
        (train_dataset, validation_dataset, test_dataset), _ = self.datasetManager.init_train_val_split(validation_split_size, 
                                        batch_size=batch_size,
                                        img_train_transform = img_train_transform,
                                        seg_train_transform = seg_train_transform,
                                        img_test_transform = img_test_transform,
                                        seg_test_transform = seg_test_transform,
                                        train_augment=augment,
                                            **augment_params_dict)
 
        train_dataloader = DataLoader( train_dataset , batch_size=batch_size, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(validation_dataset , batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)




      loss_train, loss_validation, IOU_train, IOU_validation, model, segmentation_progress, segmentation_progress_pred, segmentation_progress_true_mask  = train_segmentation_model(log_weights_path,
                                                                                        train_dataloader, 
                                                                                        val_dataloader, 
                                                                                        model,
                                                                                        learning_rate=learning_rate,
                                                                                        n_epochs=epochs, 
                                                                                        verbose=verbose, verbose_loss_acc=verbose_loss_acc
                                                                                        , weights_filename=weights_filename)

      IOU_test = validate_segmentation(model, test_dataloader, criterion=nn.CrossEntropyLoss())
      return loss_train, loss_validation, IOU_train, IOU_validation, IOU_test, model, segmentation_progress, segmentation_progress_pred, segmentation_progress_true_mask

      
  def train_convnet(self,model, learning_rate=0.0001, in_channels=3, epochs=20, 
                        train_dataloader=None, val_dataloader=None, test_dataloader=None, validation_split_size=0.1, batch_size=4,
                      img_train_transform=None,
                      seg_train_transform=None,
                      img_test_transform=None,
                      seg_test_transform=None,
                      log_weights_path="./log_dir",
                      weights_filename="fname.pt",
                      augment=False,
                      augment_params_dict={},
                      verbose=True,
                      num_workers=4):

      if train_dataloader == None or val_dataloader == None or test_dataloader == None :
        assert val_dataloader == None and val_dataloader == None and test_dataloader == None  , "Error: if any of the dataloaders is empty, all three must be set to None"
        (train_dataset, validation_dataset, test_dataset), _ = self.datasetManager.init_train_val_split(validation_split_size, 
                                        batch_size=batch_size,
                                        img_train_transform = img_train_transform,
                                        seg_train_transform = seg_train_transform,
                                        img_test_transform = img_test_transform,
                                        seg_test_transform = seg_test_transform,
                                        train_augment=augment,
                                            **augment_params_dict)
 
        train_dataloader = DataLoader( train_dataset , batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(validation_dataset , batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

      loss_train, loss_validation, acc_train, acc_validation, model  = train_classifier(log_weights_path,
                                                                                        train_dataloader, 
                                                                                        val_dataloader, 
                                                                                        model,
                                                                                        in_channels=in_channels, 
                                                                                        learning_rate=learning_rate,
                                                                                        n_epochs=epochs, 
                                                                                        verbose=verbose
                                                                                        , weights_filename=weights_filename)
      test_accuracy = test_classifier(model, test_dataloader)
      return loss_train, loss_validation, acc_train, acc_validation, test_accuracy, model

      
  def train_torch_gcn(self, model, validation_size=0.1,    # all to be supplied as list of torch graphs and list of corresponding graph labels
                      train_torch_graphs=None, train_graphs_labels=None, val_torch_graphs=None, val_graphs_labels=None,  test_torch_graphs=None, test_graphs_labels=None,
                      cross_validation=True,
                      batch_size=32,
                      learning_rate=0.001,
                      epochs=200,
                      folds = None,
                      n_repeats = None,
                      verbose=True,verbose_epochs_accuracy=False):
    
    def torch_geom_train_val_test_split(validation_size, 
                                        train_torch_graphs=None, train_graphs_labels=None,
                                        test_torch_graphs=None, test_graphs_labels=None,
                                        train_indices = None,
                                        validation_indices = None):
                      
                      if train_indices is None:
                          train_indices, validation_indices = train_test_split(range(len(train_graphs_labels)),
                                                                      test_size=validation_size,
                                                                      stratify=train_graphs_labels)
                      test_indices = [i for i in range(len(test_graphs_labels))]

                      X_torch_train = []
                      y_torch_train = []

                      X_torch_validation = []
                      y_torch_validation = []

                      X_torch_test = []
                      y_torch_test = []
                      for idx in train_indices:
                        X_torch_train.append(train_torch_graphs[idx])
                        y_torch_train.append(train_graphs_labels[idx])
                      for idx in validation_indices:
                        X_torch_validation.append(train_torch_graphs[idx])
                        y_torch_validation.append(train_graphs_labels[idx])
                      for idx in test_indices:
                        X_torch_test.append(test_torch_graphs[idx])
                        y_torch_test.append(test_graphs_labels[idx])
                      return (X_torch_train,y_torch_train), (X_torch_validation,y_torch_validation), (X_torch_test,y_torch_test)

    # if cross validation, desired behavior is to not have validation graphs provided, since these will be constructed through cross validation
    if cross_validation:
        assert val_torch_graphs is None and val_graphs_labels is None ,  "Error: during cross validaton, the validation set is taken from a partition of the training data"
        assert folds is not None  and folds >  0 , "Error: number of folds must be greater than 0"
        assert n_repeats is not None and n_repeats > 0 , "Error, number of repeats must be greater than 0"
        
        stratified_folds = model_selection.RepeatedStratifiedKFold(
          n_splits=folds, n_repeats=n_repeats
            ).split(train_graphs_labels, train_graphs_labels)

        train_acc_folds = []
        val_acc_folds = []
        test_acc_folds = []
        best_model = model
        best_acc = 0.
        for i, (train_index, validation_index) in enumerate(stratified_folds):
              print(f"Fold {i+1} of {folds}")
              curr_model = copy.deepcopy(model)
    
              (X_torch_train,y_torch_train), (X_torch_validation,y_torch_validation), (X_torch_test,y_torch_test) = torch_geom_train_val_test_split(validation_size, train_torch_graphs, train_graphs_labels,
                                              test_torch_graphs, test_graphs_labels,
                                              train_index, validation_index)    
              train_acc_epochs, val_acc_epochs, test_acc_epochs = train_test_torch_gcn(curr_model, X_torch_train, X_torch_validation, X_torch_test, batch_size=batch_size,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                verbose=verbose,
                                verbose_epochs_accuracy=verbose_epochs_accuracy)
              train_acc_folds.append(np.mean(train_acc_epochs))
              val_acc_folds.append(np.mean(val_acc_epochs))
              test_acc_folds.append(np.mean(test_acc_epochs))
              print(f"validation accuracy: {val_acc_folds[-1]}")
              if test_acc_folds[-1] > best_acc:
                best_acc = test_acc_folds[-1]
                best_model = curr_model
        return best_model, train_acc_folds, val_acc_folds, test_acc_folds
    else:
      assert not (val_torch_graphs is None ^ val_graphs_labels is None), "Error, val_graphs and val_labels must either be: both None or both allocated"
      
      # if none, define validation set from train set
      if val_torch_graphs is None and val_graphs_labels is None:
        (X_torch_train,y_torch_train), (X_torch_validation,y_torch_validation), (X_torch_test,y_torch_test) = torch_geom_train_val_test_split(validation_size,
                                              train_torch_graphs, train_graphs_labels,
                                              test_torch_graphs, test_graphs_labels,
                                              train_index=None, validation_index=None)    
      # else proceed with the provided train, validation, test sets
      else:
        (X_torch_train,y_torch_train), (X_torch_validation,y_torch_validation), (X_torch_test,y_torch_test) = train_torch_graphs, val_torch_graphs, test_torch_graphs
      curr_model = copy.deepcopy(model)
      train_acc_epochs, val_acc_epochs, test_acc_epochs = train_test_torch_gcn(model, X_torch_train, X_torch_validation, X_torch_test, batch_size=batch_size,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                verbose=verbose,
                                verbose_epochs_accuracy=verbose_epochs_accuracy)
      return curr_model, train_acc_epochs, val_acc_epochs, test_acc_epochs



  def train_sg_gcn(self, validation_size=0.1,    # all to be supplied as list of torch graphs and list of corresponding graph labels
                      train_sg_graphs=None, train_graphs_labels=None,
                      val_sg_graphs=None, val_graphs_labels=None,  
                      test_sg_graphs=None, test_graphs_labels=None,
                      early_stopping = EarlyStopping(
                      monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True),
                      cross_validation=True,
                      batch_size=32,
                      learning_rate=0.001,
                      epochs=200,
                      folds = 10,
                      n_repeats = 1,
                      verbose=True,verbose_epochs_accuracy=False):
    def train_fold(model, train_gen, test_gen, es, epochs):
          history = model.fit(
              train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
          )
          # calculate performance on the test data and return along with history
          test_metrics = model.evaluate(test_gen, verbose=0)
          test_acc = test_metrics[model.metrics_names.index("acc")]

          return history, test_acc
    def test_model(model, test_gen):
          test_metrics = model.evaluate(test_gen, verbose=0)
          test_acc = test_metrics[model.metrics_names.index("acc")]

          return test_acc
    # if cross validation, desired behavior is to not have validation graphs provided, since these will be constructed through cross validation
    if cross_validation:
        assert val_sg_graphs is None and val_graphs_labels is None ,  "Error: during cross validaton, the validation set is taken from a partition of the training data"
        assert folds is not None  and folds >  0 , "Error: number of folds must be greater than 0"
        assert n_repeats is not None and n_repeats > 0 , "Error, number of repeats must be greater than 0"
        
        #''''
        train_ordinal_labels = pd.DataFrame(train_graphs_labels.values.argmax(axis=1))
        test_ordinal_labels = pd.DataFrame(test_graphs_labels.values.argmax(axis=1))
      
        stratified_folds = model_selection.RepeatedStratifiedKFold(
          n_splits=folds, n_repeats=n_repeats
            ).split(train_ordinal_labels, train_ordinal_labels)

        train_histories = []
        train_acc_folds = []
        val_acc_folds = []
        test_acc_folds = []
        
        best_model = None
        best_acc = 0.
        train_padder_gen = PaddedGraphGenerator(train_sg_graphs)
        test_padder_gen = PaddedGraphGenerator(test_sg_graphs)
        for i, (train_index, validation_index) in enumerate(stratified_folds):
              print(f"Fold {i+1} of {folds}")
             
              # from padders obtained keras padded-graph sequences
              train_gen = train_padder_gen.flow(
                        train_index, targets=train_ordinal_labels.iloc[train_index].values, batch_size=batch_size, shuffle=True
                    )
              val_gen = train_padder_gen.flow(
                  validation_index, targets=train_ordinal_labels.iloc[validation_index].values, batch_size=batch_size, shuffle=True
                    )
              test_gen = test_padder_gen.flow(
                  np.array(list(range(len(test_sg_graphs)))), targets=test_ordinal_labels.values, batch_size=batch_size, shuffle=True
                    )
          
       
              #setup model
              model = create_graph_classification_model(train_padder_gen)

              # train and evaluate
              train_history, val1_acc = train_fold(model, train_gen, val_gen, early_stopping, epochs)
              val_acc = test_model(model, val_gen)
              test_acc = test_model(model, test_gen)

              # log results of current fold
              train_histories.append(train_history)
              train_acc_folds.append(val1_acc)
              val_acc_folds.append(val_acc)
              test_acc_folds.append(test_acc)
              plt.plot(train_history.history['acc'])
              plt.plot(train_history.history['val_acc'])
              plt.title('model accuracy')
              plt.ylabel('accuracy')
              plt.xlabel('epoch')
              plt.legend(['train', 'validation'], loc='upper left')
              plt.show()
              

              if test_acc_folds[-1] > best_acc:
                best_acc = test_acc_folds[-1]
                best_model = model
        
        

        return best_model, train_histories, train_acc_folds, val_acc_folds, test_acc_folds
    else:
      assert not (val_sg_graphs is None ^ val_graphs_labels is None), "Error, val_graphs and val_labels must either be: both None or both allocated"
      
      # if none, define validation set from train set
      if val_sg_graphs is None and val_graphs_labels is None:
        train_indices, validation_indices = train_test_split(range(len(train_graphs_labels)),
                                                                      test_size=validation_size,
                                                                      stratify=train_graphs_labels)
        test_indices = [i for i in range(len(test_graphs_labels))]    

        dataset_graphs = train_sg_graphs
        dataset_labels = train_graphs_labels
      # else proceed with the provided train, validation, test sets
      else: 
        N_train = len(train_graphs_labels)
        N_validation = len( val_graphs_labels )

        train_indices = np.array([i for i in range(N_train)])
        validation_indices = np.array([N_train+i for i in range(N_validation) ])
        test_indices = [i for i in range(len(test_graphs_labels))] 

        dataset_graphs = train_sg_graphs + val_sg_graphs
        dataset_labels = pd.concat([train_graphs_labels , val_graphs_labels],  axis=0) #concat labels vertically

        
      

      # graph padders 
      train_padder_gen = PaddedGraphGenerator(graphs=dataset_graphs)
      
      test_padder_gen = PaddedGraphGenerator(graphs=test_sg_graphs)
              
      # from padders obtained keras padded graph sequences
      train_gen = train_padder_gen.flow(
                        train_indices, targets=dataset_labels.iloc[train_indices].values, batch_size=batch_size
                    )
      val_gen = train_padder_gen.flow(
                  validation_indices, targets=dataset_labels.iloc[validation_indices].values, batch_size=batch_size
                    )
      test_gen = test_padder_gen.flow(
                  test_indices, targets=test_graphs_labels.iloc[test_indices], batch_size=batch_size
                    )
      #setup model
      model = create_graph_classification_model(train_padder_gen)

      # train and evaluate
      train_history, train_acc = train_model(model, train_gen, val_gen, early_stopping, epochs)
      val_acc = test_model(model, val_gen)
      test_acc = test_model(model, test_gen)

     
      return model, train_history, train_acc, val_acc, test_acc


  # graph 
  def get_segmented_masks_graph_items(self, segmentation_model,  
                                      train_dataloader=None, val_dataloader=None, test_dataloader=None, 
                                      validation_split_size=0.1, 
                                        batch_size=4,
                                        img_train_transform = None,
                                        seg_train_transform = None,
                                        img_test_transform = None,
                                        seg_test_transform = None):
    
    if train_dataloader == None or val_dataloader == None or test_dataloader == None :
        assert val_dataloader == None and val_dataloader == None and test_dataloader == None  , "Error: if any of the dataloaders is empty, all three must be set to None"
        
        (train_dataset, validation_dataset, test_dataset), _ = self.datasetManager.init_train_val_split(validation_split_size, 
                                        batch_size=batch_size,
                                        img_train_transform = img_train_transform,
                                        seg_train_transform = seg_train_transform,
                                        img_test_transform = img_test_transform,
                                        seg_test_transform = seg_test_transform,
                                        train_augment=False,
                                            **{})
        train_dataloader = DataLoader( train_dataset , batch_size=batch_size, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(validation_dataset , batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    (train_pred_graphs, train_pred_graph_labels), (val_pred_graphs, val_pred_graph_labels), (test_pred_graphs, test_pred_graph_labels) = get_pred_mask_graph_datasets(segmentation_model,
                                 train_dataloader, val_dataloader, test_dataloader, 
                                 self.datasetManager.resize_dim )
    return  (train_pred_graphs, train_pred_graph_labels), (val_pred_graphs, val_pred_graph_labels), (test_pred_graphs, test_pred_graph_labels)





  @staticmethod
  def train_crop_segmentation(dataset_root_path , model, learning_rate=0.00001, epochs=20, crops_per_side = 4,
                         batch_size=4,
                        num_workers = 2,
                        img_train_transform=None,
                        seg_train_transform=None,
                        img_test_transform=None,
                        seg_test_transform=None,
                        log_weights_path="./log_dir",
                        weights_filename="fname.pt",
                        verbose=False,
                        verbose_loss_acc=True):


       
        IMAGE_SIDE_LEN = 2048
        CROPS_PER_SIDE = crops_per_side
        VERBOSE = verbose
        """
        img_train_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])#,transforms.Normalize(gray_mean,gray_std)])
        seg_train_transform = transforms.Compose([transforms.ToTensor()])
        img_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])#,transforms.Normalize(gray_mean, gray_std)])
        seg_test_transform = transforms.Compose([transforms.ToTensor()])
        """
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


        train_crop_dataloader = DataLoader(train,batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
        test_crop_dataloader = DataLoader(test,batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)


        loss_train, loss_validation, IOU_train, IOU_validation, model,segmentation_progress = train_crop_segmentation_model(log_weights_path,
                                                                  train_crop_dataloader, test_crop_dataloader,
                                                                  model,
                                                                  learning_rate=learning_rate,
                                                                n_epochs=10,
                                                                verbose=True, verbose_loss_acc=True,
                                                                weights_filename="torch_weights.pt")

  
        return loss_train, loss_validation, IOU_train, IOU_validation, segmentation_progress, model










def get_predicted_segmentation_masks_dataset(model, dataloader):
  model.eval()
  device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
  
  log_softmax = nn.LogSoftmax(dim=1)
  IoU_test =[]
  predicted_segmentation_masks = []
  y_labels = []
  
  for (path_img, path_seg, img, seg, seg_gt),label in dataloader:
    seg = seg.to(device)
    img = img.to(device)
    seg_gt = seg_gt.long().to(device)
    seg_pred = model(img)
    y_pred_binarized = log_softmax(seg_pred).argmax(dim=1, keepdim=True)

    IoU_test.append(IoU(y_pred_binarized, seg_gt))

    # store each prediction alongside the label
    for idx in range(len(img)):
      # cv2 format for grayscale images requires uint8 data type for numpy arrays
      predicted_segmentation_masks.append( (y_pred_binarized[idx].cpu().detach().numpy().squeeze()*255).astype(np.uint8) )
      y_labels.append(label[idx].numpy().item())
    del seg
    del img
    del seg_gt
    del seg_pred
    
  return predicted_segmentation_masks, y_labels


def get_pred_mask_graph_datasets(segmentation_model, train_dataloader, val_dataloader, test_dataloader, resize_dim=512):
  # inner utility function
  def get_pred_graphs_dataset(X,y,resize_dim, img_format='RGB', seg_color_mapping=cv2.IMREAD_GRAYSCALE, loading_prompt_string=None):
          pred_graphs, pred_graph_labels = RCCDatasetManager.load_graph_items(X, y,
                                                                                                              resize_dim,
                                                                                                              img_format,
                                                                                                            seg_color_mapping, 
                                                                                                            loading_prompt_string= loading_prompt_string)
          return pred_graphs, pred_graph_labels
  X_train, y_train = get_predicted_segmentation_masks_dataset(segmentation_model, train_dataloader)
  X_val, y_val = get_predicted_segmentation_masks_dataset(segmentation_model, val_dataloader)
  X_test, y_test = get_predicted_segmentation_masks_dataset(segmentation_model, test_dataloader)

  training_dataset = X_train 
  training_labels = y_train 

  val_dataset = X_val
  val_labels = y_val

  test_dataset = X_test
  test_labels = y_test

  train_pred_graphs, train_pred_graph_labels = get_pred_graphs_dataset(training_dataset,training_labels,resize_dim, loading_prompt_string="train dataset segmentation_model masks ")
  val_pred_graphs, val_pred_graph_labels = get_pred_graphs_dataset(val_dataset,val_labels,resize_dim, loading_prompt_string="train dataset segmentation_model masks ")
  test_pred_graphs, test_pred_graph_labels = get_pred_graphs_dataset(test_dataset,test_labels,resize_dim, loading_prompt_string="train dataset segmentation_model masks ")
  
  return (train_pred_graphs, train_pred_graph_labels), (val_pred_graphs, val_pred_graph_labels), (test_pred_graphs, test_pred_graph_labels)

