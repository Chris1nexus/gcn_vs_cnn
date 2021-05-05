from torch.utils.data import Dataset
import os
import cv2
import re
from tqdm import tqdm
from torchvision import transforms


from processing_utils import ToGraphTransform
from image_utils import get_segmentation_mask, read_image
from utils import DriveDownloader
from processing_utils import ToGraphTransform
from image_utils import np_make_crops, np_recompose_tensor,  img_paste_fn, seg_paste_fn, read_image, get_segmentation_mask
from utils import recursive_visit, DriveDownloader
from augmentation_utils import AugmentTransform





class RCCStorage(object):
    '''
    Container class that holds all data related to the rcc dataset, maintaining all informations for each sample.
    The dataset is essentially composed of:
            - X_img
            - X_seg
            -y_labels
    These constitute the wsi slide image crop, the segmentation mask and the cancer type label, respectively.
    For each index, the lists contain the data related to a single sample.
    Furthermore, in order to identify samples in any phase of the training, the image path, the segmentation path are maintained.
    Optionally, a data structure that associates each image to its patient is also stored.
    The latter is useful if standardization by patient is requested, and if that is the case, also the statistic used to standardize the data
    related to a specific patient are kept in a dictionary, with key the patient id and value the used statistics
    '''
    
    def __init__(self, X_img,
                 X_seg,
                 y_labels,
                 y_numeric,
                 mapping_id_to_label,
                 mapping_label_to_id,
                 img_paths,
                 seg_paths,
                img_to_patient_map=None,
                seg_to_patient_map=None,
                img_to_sample_group_statistics=None ):
        self.X_img = X_img
        self.X_seg = X_seg
        self.y_labels = y_labels
        self.y_numeric = y_numeric
        self.mapping_id_to_label = mapping_id_to_label
        self.mapping_label_to_id = mapping_label_to_id
        self.img_paths = img_paths
        self.seg_paths = seg_paths

        self.img_to_patient_map = img_to_patient_map 
        self.seg_to_patient_map = seg_to_patient_map 
        self.img_to_sample_group_statistics = img_to_sample_group_statistics


    
class RCCImageDataset(Dataset):
    '''
    Pytorch dataset data structure that holds that exposes the functionalities of a standard torch Dataset:
                -__getitem__ 
                -__len__
    This subclass allows the use of dataloaders for effective parallel processing and also of standardized data splitting in train/validaition/test.
    parameters:
        -rccStorage (container that holds all data related to the current dataset instance)
        -resize_dim (image shape used to resize the original images)
        -img_format (convention used to load images, then maintained everywhere in all dataset structures: cv2 uses BGR but for interfacing purposes between cv2 and pytorch, RGB is used)
        -img_color_mapping (color mapping used to load the images by means of opencv)
        -seg_color_mapping (color mapping used to load the segmentation masks by means of opencv)
    resize_dim is required, since the ToGraphTransform (that extracts a graph data structure from the segmentation mask)
    uses different kernel sizes for preprocessing, depending on the resized image size
    '''

    def __init__(self, 
                 rccStorage,
                resize_dim=512,
                img_format ='RGB', # alternative is BGR, which is the standard format for cv2 but not other libraries
                img_color_mapping=cv2.IMREAD_COLOR,
                seg_color_mapping=cv2.IMREAD_GRAYSCALE):
        #super(RCCDataset, self).__init__(root,
        #                              transform=transform, 
        #                              target_transform=target_transform)
        
        self.rccStorage = rccStorage
      
        self.img_format = img_format
        self.resize_dim = resize_dim
        self.img_color_mapping = img_color_mapping
        self.seg_color_mapping = seg_color_mapping
        
        
    
        self.X_img = self.rccStorage.X_img
        self.X_seg = self.rccStorage.X_seg
        self.y_ids = self.rccStorage.y_numeric
        self.y_labels = self.rccStorage.y_labels
        self.mapping_label_to_id = self.rccStorage.mapping_label_to_id
        self.mapping_id_to_label = self.rccStorage.mapping_id_to_label
        self.img_paths = self.rccStorage.img_paths
        self.seg_paths = self.rccStorage.seg_paths
  
        self.to_graph_transformer = ToGraphTransform(SQUARE_IMAGE_SIZE=self.resize_dim)

    def __getitem__(self, index):
        '''
        __getitem__ access an element through its index.
        Args:
            index (int): Index

        Returns:
            tuple: (image_path, seg_path, image, segmented, segmented_ground_truth), label_id    
            with all the data related to a given sample
        '''

        image_path = self.img_paths[index]
        seg_path = self.seg_paths[index]
        
        image = self.X_img[index]
        segmented = self.X_seg[index]

        label_id = self.y_ids[index]

   
        segmented_ground_truth = get_segmentation_mask(segmented, self.seg_color_mapping, self.img_format ) 
        
      
        return (image_path, seg_path, image, segmented, segmented_ground_truth), label_id

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.y_ids) # Provide a way to get the length (number of elements) of the dataset
        return length





class RCCImageSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The pytorch dataset from which the subset is obtained
        indices (sequence): Indices in the whole set, selected for subset
        img_transform (torch.transforms): image transforms to use on the subset of the original dataset
        seg_transform (torch.transforms): mask transforms to use on the subset of the original dataset
        augment (bool): whether data augmentation has to be applied
        resized_crop   (dictionary)
        rotate=None   (dictionary)
        gauss_blur=None   (dictionary)
        elastic_deform=None   (dictionary)
        convention for the dictionaries is (according to AugmentTransform in the augmentation_utils module)
                     -resize_crop   (default to {'prob':1.0, 'original_kept_crop_percent':(0.75,0.9)})  (dictionary)
                      -rotate   (default to {'prob':1.0})  (dictionary)
                      -gauss_blur   (default{'prob':1.0,'kernel_size':3, 'sigma':(0.1, 2.0)}) (dictionary)
                      -elastic_deform (default {'alpha':(1,10), 'sigma':(0.08, 0.5), 'alpha_affine':(0.01, 0.2), 'random_state':None}) (dictionary)

    """
    
    def __init__(self, dataset, indices, img_transform, seg_transform, 
                 augment=False,
                 resized_crop=None,
                 rotate=None,
                 gauss_blur=None,
                 elastic_deform=None):
        self.dataset = dataset
        self.indices = indices
        self.img_transform = img_transform
        self.seg_transform = seg_transform
        self.augment = augment
        self.augmentTransform = AugmentTransform(resized_crop=resized_crop,
                                                  rotate=rotate,
                                                 gauss_blur=gauss_blur,
                                                 elastic_deform=elastic_deform)
    def __getitem__(self, idx):
        '''
        Args: 
                -index of the sample
        Returns ( sample data, returned after the transformations are applied):
                - img_path (str)
                - seg_path (str)
                - img (torch.Tensor)
                - seg (torch.Tensor)
                - seg_gt (torch.Tensor)
                - label (int)


        '''
        (img_path, seg_path, img, seg, seg_gt), label = self.dataset[self.indices[idx]]
 
        if self.img_transform is not None:

          copied_img_transform = transforms.Compose(self.img_transform.transforms.copy())
          if self.augment:
            # todo augmenttransform for multiple input images instead of just two
            img, seg_gt = self.augmentTransform.transform(img, seg_gt)
            # augmenttransform requires to transform objects to torch tensors, so 
            # its output is already a torch tensor

            #if augmentation happened
            #  it is needed to not consider any ToTensor() transformation in img_transform 
            copied_img_transform.transforms = [ transform for transform in copied_img_transform.transforms \
                                                        if not isinstance(transform, transforms.ToTensor) ]
          else:
            # no normalization has to be applied to the binarized segmentation mask so it is just transformed to torch tensor              
            seg_gt = transforms.ToTensor()(seg_gt)
          
          # in any case this works properly
          img = copied_img_transform(img)
        if self.seg_transform is not None:
          seg = self.seg_transform(seg)
        

        return (img_path, seg_path, img,seg,seg_gt), label

    def __len__(self):
        '''
        Args:

        Returns:
            length of the dataset (int)
        '''
        return len(self.indices)








class CropDataset(Dataset):
  '''
    Torch dataset used to perform which follows the same principles of the other datasets,
    but resizes images to 2048 (from 2000x2000 original size) so that their length can be split in equal length segments that are powers of 2.
    For example, from an image 2048x2048, splitting the side length in 4 pieces yields 16 cropped images from 1 original, and so on.
    

  '''
  def __init__(self,
                  root_path,
                  download_dataset = True,
                 in_memory=True,
                 partition="Train",
                resize_dim=2048,
                 num_crops_per_side =4,

                img_format ='RGB', # alternative is BGR, which is the standard format for cv2 but not other libraries
                img_color_mapping=cv2.IMREAD_COLOR,
                seg_color_mapping=cv2.IMREAD_COLOR,
                 img_transform=None,
                 target_transform=None,
               verbose=True):


        if download_dataset == True:
          drive_file_id = '1jz9lC2j4CfH9oF3CRVHHLs8C5k6yw46i'
          tmp_destination = './dataset.zip'
          target_directory = './rccdataset'
          target_path = os.path.join(target_directory, "vascular_segmentation")
          if (not os.path.exists(target_path) or root_path is None) or \
                              ( root_path is not None and ("vascular_segmentation" not in root_path or not os.path.exists(root_path))  ):
            # if root path is none and the downloaded dataset folder does not exist OR 
            # root path is NOT none and (it does not contain vascular segmentation OR it does not exist)
            # we download the dataset on the folder
            self.__download_dataset__(drive_file_id, tmp_destination, target_directory)
          else:
            target_path = root_path
          self.root_path = target_path
        else:
          assert root_path is not None and "vascular_segmentation" in root_path, "Error: root path must point to the 'vascular_segmentation' folder"
          self.root_path = root_path

        self.in_memory = in_memory
        self.img_format = img_format
        self.resize_dim = resize_dim

        self.num_crops_per_side = num_crops_per_side

        self.img_color_mapping = img_color_mapping
        self.seg_color_mapping = seg_color_mapping
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.verbose = verbose
        
        self.rccStorage = self.__load_rcc_dataset__(
                     partition=partition
                    )        
        
    
        self.X_img = self.rccStorage.X_img
        self.X_seg = self.rccStorage.X_seg
        self.y_ids = self.rccStorage.y_numeric
        self.y_labels = self.rccStorage.y_labels
        self.mapping_label_to_id = self.rccStorage.mapping_label_to_id
        self.mapping_id_to_label = self.rccStorage.mapping_id_to_label
        self.img_paths = self.rccStorage.img_paths
        self.seg_paths = self.rccStorage.seg_paths
  
        self.to_graph_transformer = ToGraphTransform(SQUARE_IMAGE_SIZE=self.resize_dim)

  def __getitem__(self, index):
        image_path = self.img_paths[index]
        seg_path = self.seg_paths[index]
        if self.in_memory:
            image = self.X_img[index]
            segmented = self.X_seg[index]
        else:
            image = read_image(image_path, self.resize_dim, self.img_color_mapping, self.img_format)
            segmented = read_image(seg_path, self.resize_dim, self.seg_color_mapping, self.img_format)

        segmented_ground_truth = get_segmentation_mask(segmented, self.seg_color_mapping, self.img_format ) 
        
        
        if self.img_transform is not None:
          image = self.img_transform(image)
        if self.target_transform is not None:
          segmented_ground_truth = self.target_transform(segmented_ground_truth)


        img_crops = np_make_crops(image,  image_shape_getter_fn=lambda x: x.shape[1:],
                                          crop_fn=lambda tensor_img,row_indices, col_indices: tensor_img[:, row_indices, col_indices]  ,
                                        split_side_in=self.num_crops_per_side)
        seg_gt_crops = np_make_crops(segmented_ground_truth,  image_shape_getter_fn=lambda x: x.shape[1:],
                                          crop_fn=lambda tensor_img,row_indices, col_indices: tensor_img[:, row_indices, col_indices]  ,
                                          split_side_in=self.num_crops_per_side)
        


        return  (img_crops, seg_gt_crops)#list(zip(img_crops, seg_gt_crops))

  def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.y_ids) # Provide a way to get the length (number of elements) of the dataset
        return length
        
  def __load_rcc_dataset__(self,
                     partition="Train",
                    ):
      
        labels_dict = dict()
        id_to_labels = []

        num_labels = 0
        X_img = []
        X_seg = []
        y_cancer_ids = []
        y_labels = []


        assert partition == "Train" or partition == "Test", "Error: dataset split must either be 'Train' or 'Test' "
        
        
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
        if partition == "Test" :
            filename_pattern = test_pattern
        else:
            filename_pattern = train_pattern




        samples = []
        # load all labels first since they are not computationally expensive to process
        # then load all images if in_memory=True with tqdm progress bar
        # 
        
        folder_iterable = os.listdir(folder_path)
        if self.verbose:
            print("Scanning " + partition + " dataset directories")
            folder_iterable = tqdm(folder_iterable, leave=True,position=0 )
        for directory in folder_iterable:

            if directory not in labels_dict:
                    labels_dict[directory] = num_labels
                    id_to_labels.append(directory)
                    # pRCC or cRCC
                    category = directory
                    if partition == "Test":
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
                    
                    
        sample_iterable = samples
        if self.verbose:
            print("Loading " + partition + " dataset")
            sample_iterable = tqdm(samples, position=0, leave=True)
        
        for img_path, seg_path in sample_iterable:
            if self.in_memory:
                # if in memory, read img, and segmented sample and store the pair in the dataset
                img = read_image(img_path, self.resize_dim, self.img_color_mapping, self.img_format)#read_image(img_path, self.resize_dim, self.img_color_mapping)
                X_img.append(img)

                seg = read_image(seg_path, self.resize_dim, self.seg_color_mapping, self.img_format)
                X_seg.append(seg)

            # add paths to path lists
            img_paths.append(img_path)
            seg_paths.append(seg_path)
        rccStorage = RCCStorage( X_img, X_seg,
                                    y_labels, y_cancer_ids,
                                    id_to_labels, labels_dict,
                                    img_paths, seg_paths)      


        return rccStorage
  def __download_dataset__(self, drive_file_id, tmp_destination, target_directory):
                #drive_location = "https://drive.google.com/file/d/1jz9lC2j4CfH9oF3CRVHHLs8C5k6yw46i/view?usp=sharing"
                downloader = DriveDownloader()
                downloader.download_file_from_google_drive(drive_file_id, tmp_destination)
                downloader.extract_zip(tmp_destination, target_directory)




