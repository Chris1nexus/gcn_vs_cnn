# CNN vs GCN - Comparison between CNN and GCN for the classification of Renal Cell Cancer types
## Table of content
1 Usage



2 Requirements

3 Description

- 3.1 Graph data structure creation method

- 3.2 Image processing methods

- 3.3 Data augmentation procedures

- 3.4 Graph features
- 3.5 Experiments
- 3.6 Results

4 Further experiments

- 4.1 Unet segmentation on image sub-patches
- 4.2 Domain adaptation to remove noise due to cells in the images



## 1 Usage
To use the project, run the setup.sh script (preferably in a virtual environment) to set and install all required dependencies.
Ubuntu18.04 is the os in which the code has been tested. 
A user setup version of CUDA >= 10.2 is needed to run the training in moderate time (this is not installed by the script as dependencies might differ depending on the operating system).

The top level folder containts the experiments to run from command line.
Each script can be executed with the default parameters, but for further experiment customization, 
do python experiment_name.py --help to read the available options. 
In the same top level folder there is also the 'lib' folder which contains the source files:
- lib/data contains utilities to manage data and plot graphs.
- lib/models contains the networks used in the experiments
- lib contains all training procedures

Furthermore, a jupyter notebook (gcn_vs_cnn_experiments.ipynb) is available in the top level folder to readily test and run experiments also on colab environments.
Slides of the presentation are also made available in pdf format in the top level folder.

### 1.1 

* CNN experiment 

	usage: cnn_experiment.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
	                         [--lr LR] [--cross-val CROSS_VAL] [--std STD]
	                         [--std-by-patient STD_BY_PATIENT] [--format FORMAT]
	                         [--rand-rot RAND_ROT] [--rand-crop RAND_CROP]
	                         [--rand-elastic-deform RAND_ELASTIC_DEFORM]
	                         [--workers WORKERS] [--weights-dir WEIGHTS_DIR]
	                         [--weights-fname WEIGHTS_FNAME] [--images IMAGES]
	arguments:

	  -h, --help            show this help message and exit

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 4)

	  --epochs EPOCHS       number of epochs to train (default: 100)

	  --lr LR               initial learning rate (default: 0.001)

	  --cross-val CROSS_VAL
	                        Perform 5-fold cross-validation: [True,False] (default
	                        to False)

	  --std STD             standardize images according to the channel statistics
	                        [True, False] (default True)

	  --std-by-patient STD_BY_PATIENT
	                        [True, False] (default to False) compute mean and
	                        variance for each 'train' split patient and
	                        standardize each of their samples by their own
	                        statistics: test samples are standardized according to
	                        the average mean and pooled variance

	  --format FORMAT       image format:['rgb','gray'] (default is rgb)

	  --rand-rot RAND_ROT   random rotations (90,180,270 degrees) data
	                        augmentation, [True, False] (default to False)

	  --rand-crop RAND_CROP
	                        random crop and zoom (keep from 0.7 to 1.0 of the
	                        original image ) data augmentation, [True, False]
	                        (default to False)

	  --rand-elastic-deform RAND_ELASTIC_DEFORM
	                        [True, False] (default to False) elastic deformation:
	                        alpha in [1,4] sigma in [0.07, 0.13] alpha affine in
	                        [0.07, 0.13]

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --weights-dir WEIGHTS_DIR
	                        folder to save weights

	  --weights-fname WEIGHTS_FNAME
	                        weights filename

	  --images IMAGES       root folder with train and test folders

* Unet experiment

	usage: unet_experiment.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
	                          [--lr LR] [--std STD]
	                          [--std-by-patient STD_BY_PATIENT] [--format FORMAT]
	                          [--rand-rot RAND_ROT] [--rand-crop RAND_CROP]
	                          [--rand-elastic-deform RAND_ELASTIC_DEFORM]
	                          [--workers WORKERS] [--weights-dir WEIGHTS_DIR]
	                          [--weights-fname WEIGHTS_FNAME] [--images IMAGES]

	arguments:

	  -h, --help            show this help message and exit

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 4)

	  --epochs EPOCHS       number of epochs to train (default: 10)

	  --lr LR               initial learning rate (default: 0.0001)

	  --std STD             [True, False] (default to True)standardize slide
	                        images according to the channel statistics (default:
	                        True)

	  --std-by-patient STD_BY_PATIENT
	                        [True, False] (default to False) compute mean and
	                        variance for each 'train' split patient and
	                        standardize each of their samples by their own
	                        statistics: test samples are standardized according to
	                        the average mean and pooled variance (default: False)

	  --format FORMAT       slide image format:['rgb','gray'] (default is rgb)

	  --rand-rot RAND_ROT   [True, False] (default to False)random rotations
	                        (90,180,270 degrees) data augmentation (default:
	                        False)

	  --rand-crop RAND_CROP
	                        [True, False] (default to False)random crop and zoom
	                        (keep from 0.7 to 1.0 of the original image ) data
	                        augmentation (default: False)

	  --rand-elastic-deform RAND_ELASTIC_DEFORM
	                        [True, False] (default to False)elastic deformation
	                        (default: False): alpha in [1,4] sigma in [0.07, 0.13]
	                        alpha affine in [0.07, 0.13]

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --weights-dir WEIGHTS_DIR
	                        folder to save weights

	  --weights-fname WEIGHTS_FNAME
	                        weights filename

	  --images IMAGES       root folder with train and test folders


* GCN experiment 

	usage: gcn_experiment.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
	                         [--lr LR] [--gcn-type GCN_TYPE] [--dataset DATASET]
	                         [--state-dict-path STATE_DICT_PATH]
	                         [--unet-lr UNET_LR] [--unet-epochs UNET_EPOCHS]
	                         [--std STD] [--std-by-patient STD_BY_PATIENT]
	                         [--format FORMAT] [--workers WORKERS]
	                         [--weights-dir WEIGHTS_DIR]
	                         [--weights-fname WEIGHTS_FNAME] [--images IMAGES]



	arguments:

	  -h, --help            show this help message and exit

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 32)

	  --epochs EPOCHS       number of epochs to train (default: 200)

	  --lr LR               initial learning rate (default: 0.001)

	  --gcn-type GCN_TYPE   gcn library that implements the GCN(default torch):
	                        ['torch', 'sg'] 'torch' is the torch geometric library
	                        'sg' is the stellargraph library

	  --dataset DATASET     dataset on which the GCN is trained (default unet):
	                        ['unet', 'gt'] 'unet' is the graph dataset created
	                        from predicted segmentation masks 'gt' is the graph
	                        dataset created from ground truth masks

	  --state-dict-path STATE_DICT_PATH
	                        (ONLY VALID IF unet dataset) Path to the weights of
	                        the trained unet model (if this is valid, --unet-lr,
	                        --unet-epochs are ignored), else a unet will be
	                        trained with: lr=unet-lr argument, batch_size=4,
	                        epochs=unet-epochs argument all remaining parameters
	                        are configurable by means of the available command
	                        line arguments

	  --unet-lr UNET_LR     (ONLY VALID IF unet dataset) unet learning rate
	                        (default: 0.00005)

	  --unet-epochs UNET_EPOCHS
	                        (ONLY VALID IF unet dataset) unet epochs (default: 40)

	  --std STD             (ONLY VALID IF unet dataset) [True, False] (default to
	                        True) standardize slide images according to the
	                        channel statistics

	  --std-by-patient STD_BY_PATIENT
	                        (ONLY VALID IF unet dataset) [True, False] (default to
	                        False) compute mean and variance for each 'train'
	                        split patient and standardize each of their samples by
	                        their own statistics: test samples are standardized
	                        according to the average mean and pooled variance

	  --format FORMAT       (ONLY VALID IF unet dataset) slide image
	                        format:['rgb','gray'] (default is rgb)

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --weights-dir WEIGHTS_DIR
	                        folder to save weights

	  --weights-fname WEIGHTS_FNAME
	                        weights filename

	  --images IMAGES       root folder with train and test folders



* Train cell segmentation domain adaptation network 

	usage: adaptation_experiment.py [-h] [--batch-size BATCH_SIZE]
	                                [--epochs EPOCHS] [--lr LR]
	                                [--cross-val CROSS_VAL] [--workers WORKERS]
	                                [--weights-dir WEIGHTS_DIR]
	                                [--weights-fname WEIGHTS_FNAME]
	                                [--images IMAGES]


	arguments:

	  -h, --help            show this help message and exit

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 4)

	  --epochs EPOCHS       number of epochs to train (default: 100)

	  --lr LR               initial learning rate (default: 0.001)

	  --cross-val CROSS_VAL
	                        Perform 5-fold cross-validation: [True, False]
	                        (default to False)

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --weights-dir WEIGHTS_DIR
	                        folder to save weights

	  --weights-fname WEIGHTS_FNAME
	                        weights filename

	  --images IMAGES       root folder with train and test folders


* Train unet to segment on domain adapted images 

	usage: cell_segmentation_experiment.py [-h] [--batch-size BATCH_SIZE]
	                                       [--epochs EPOCHS] [--lr LR]
	                                       [--workers WORKERS]
	                                       [--weights-dir WEIGHTS_DIR]
	                                       [--weights-fname WEIGHTS_FNAME]
	                                       [--images IMAGES]

	Unet on adapted images experiment

	optional arguments:

	  -h, --help            show this help message and exit

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 4)

	  --epochs EPOCHS       number of epochs to train (default: 40)

	  --lr LR               initial learning rate (default: 0.0001)

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --weights-dir WEIGHTS_DIR
	                        folder to save weights

	  --weights-fname WEIGHTS_FNAME
	                        weights filename

	  --images IMAGES       root folder with train and test folders


* Plot gradcam results of a trained CNN (requires weights of an already trained vgg16 net from cnn_experiment.py )

	usage: eval_gradcam.py [-h] [--vgg-weights-path VGG_WEIGHTS_PATH]
	                       [--batch-size BATCH_SIZE] [--std STD] [--format FORMAT]
	                       [--workers WORKERS] [--logs-dir LOGS_DIR]
	                       [--images IMAGES]

	CNN gradcam plotting (de-standardization by-patient not supported)

	optional arguments:

	  -h, --help            show this help message and exit

	  --vgg-weights-path VGG_WEIGHTS_PATH
	                        path to the weights of the vgg16 classifier

	  --batch-size BATCH_SIZE
	                        input batch size for training (default: 4)

	  --std STD             standardize images according to the channel statistics
	                        [True, False] (default to True)

	  --format FORMAT       image format:['rgb','gray'] (default is rgb)

	  --workers WORKERS     number of workers for data loading (default: 4)

	  --logs-dir LOGS_DIR   folder to save weights

	  --images IMAGES       root folder with train and test folders

## 2 Requirements



## 3 Description
The aim of this project is to compare the capability of convolutional and graph deep learning methods to discriminate between papillary and clear cell renal cancer.
For these purposes, convolutional models are trained on raw image data coming from WSI slides, while graph convolutional models are trained on segmentation masks generated by a U-net architecture. 
Since graph neural models require graph samples, such data structure has been extracted from the predicted segmentation masks.
In order to achieve this, the predicted masks have to be processed so that a graph data structure can be extracted from them.

### 3.1 Graph data structure creation method

This is done by means of (http://dx.doi.org/10.1007/978-3-642-02345-3_45). In order to obtain a faster and more efficient implementation, few improvements have been implemented on the proposed algorithm in their paper.
More specifically, the algorithm in section 3.2 of the paper (Adjacency matrix computation) has been improved from their O(N_j*V^2) complexity, to linear complexity
To this regard, the original paper proposes three for loops, of which the first and the third are the computationally expensive ones: the second is just the
allocation of the adjacency matrix so it is trivial.
The first, of complexity O(N_j*P) (N_j number of joints between edges and nodes, and P number of pixels in the image) has been reduced to O(P).
This has been done by observing that the identification of a joint and the associated node and edge can be done by simply taking anyone of the pixels in the area of a given joint: through a lookup of its coordinates, also the labels of the edge and the node can be obtained.
Furthermore, the 'third' for loop of complexity O(N_j*N_j) has been reduced to linear complexity O(N_j) (with N_j number of joints), by employing the explained 
data structure construction which allows faster lookups.
More details are provided in the code of the ToGraphTransform class, whose function graph_transform implements this procedure.

### 3.2 Image processing methods
Images have been resized to shape 512x512. 
For colored images, the adopted channel ordering convention is 'RGB' (alternative is 'BGR').
Experiments have been performed both with and without standardization by mean of the image channels. In any case, the pixel values have been translated in the range 0...1 for pytorch compatibility before standardizing. 
Furthermore in a test experiment, since different patient annotations show different color intensities independently on the cancer class, 
the samples of any given patient annotation have been preprocessed by standardizing by the patient statistics to remove any dependence on the patient that is
unrelated to the cancer classification.
However no improvement has been observed over simple standardization by the whole train sample statistics.

### 3.3 Data augmentation procedures

Due to the few samples available in the renal cell cancer dataset, data augmentation procedure should positively improve the classifier accuracy.
In the case of segmentation model training, the same data augmentation is applied to both the image and the mask.
To this end, the following augmentation techniques have been implemented:
- random rotation of 0, 90, 180, 270 degrees
- random crop and resize of the image (from 0 to 25% of the image is randomly cropped and resized to the original 512x512)
- gaussian blur (standard deviation from 0.1 to 2 and kernel size 3)
- elastic deformation as described in [Simard2003]_Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis". 
        values of the parameters of the deformation are:
        	alpha=(1,10)
        	sigma=(0.08, 0.5)
        	alpha_affine=(0.01, 0.2)

From the experiment initialization, any of the transformation can be activated comtemporaneously to any other.
In practice, random rotation alone has yielded a 4% improvement on average in the IoU metric on the validation set (unet segmentation task).

### 3.4 Graph features
In order to provide descriptive features for the graph nodes, the following features are considered:
-vertical size of the bounding box of the node (can be treated also as major-minor axis of a bounding ellipse, which is more descriptive of the shape of the found nodes)
-horizontal size of the bounding box of the node (can be treated also as major-minor axis of a bounding ellipse, which is more descriptive of the shape of the found nodes)
-area of the found node
These are obtained by means of the cv2 function connectedComponentsWithStats.


### 3.5 Experiments
The objective of the project is the comparison between CNNs trained on raw data and GCNs trained on unet segmentations of the raw data's vascular networks.
To achieve this, cnn_experiment.py and gcn_experiment.py provide the command line program that allows to customize the experiments with respect to:
-preprocessing
-processing
-data augmentation
-hyper parameter configurations
-logging directories and files
By calling either the python script with the --help command line arguments, every detail of the command line arguments can be read.
The cnn_experiment.py script allows to train a VGG 16 with batchnorm with the provided configurations.
The gcn_experiment.py script allows to train a GCN either with:
- ground truth dataset 'gt' (graphs are generated from ground truth masks without any segmentation)
- unet dataset 'unet' (graphs are generated from unet segmentations of the raw data)
This helps in performing a simple ablation study that assesses the contribution brought by segmenting true data by means of a Unet instead of simply generating it from ground truths.
From this, it has been observed that training a gcn on graphs generated from unet segmented data improves a GCN classifiation accuracy of about 20% (validation accuracy of 85%) over training them on graphs generated from ground truth masks (65-70%).
Furthermore, the script unet_experiment.py has been provided in order to separately evaluate the unet segmentation performance over the epochs and to also train separately a unet model with more customization possibilities. The saved best model weights can then be used in the GCN script by specifying the path to those weights and also the used preprocessing procedures.


### 3.6 Results
The CNN easily reaches 95% accuracy. The learning curves are generated after its training is completed, in the log folder.
The GCN instead has been evaluated by splitting the training experiment in three parts:
- evaluation of the unet segmentation performance by means of the intersection over union (IoU) metric:
	the best unet model reaches 40% validation IoU after 30-40 epochs of training. Stopping the training sooner results in more randomness in the predicted segmentation masks(this result can be observed in the log folder of the gcn_experiment.py or unet_experiment.py, where a gif of the result over the training epochs is plotted after learning is done).
- evaluation of a GCN trained on graphs generated from ground truth segmentation masks: 
	with 5 fold cross validation, the model reaches around 70% accuracy on average of the folds. Learning curves about train and validation performance are, also in this case, provided after training in the log folder
- evaluation of a GCN trained on graphs generated by means of a trained unet predicted segmentation masks from the raw data. Learning curves about train and validation performance are provided after training in the log folder, both for the GCN and the UNet(if it has been trained during the call to the script).
Through this set of experiments, it has been observed that segmenting raw data from unet (and then creating graphs from these), allows to improve accuracy of the GCN classifier of up to 25%: the best obtained GCN model trained this way, achieved around 95% validation accuracy, reaching comparable performance to the CNN.




### 4.1 Further experiments (Unet segmentation on image crops)
Unet segmentation on square crops of the raw data: the goal of this experiment is to simplify the area that has to be segmented, by splitting each image in a 4x4 grid of patches that now become the samples that have to be segmented. This has been done since i have observed that many samples show very complex vascular network patterns that appeared to be difficult to segment for the network. 
However in this case the IoU on validation crops stayed around 40%, similar to segmentation over the uncropped image sample.
Note that splitting the images into 4x4 grids provides a way to do intrinsic data augmentation since now each sample is divided into 16 crops: the original dataset of 175 images becomes 
of 2800 images.

### 4.2 Further experiments (Domain adaptation to remove noise due to cells in the images):
By observing the dataset sample, it is easy to see that cells very frequently overlap the vascular network: this could be a source of noise that can be eliminated.
In order to achieve this, a secondary unet is trained to segment cell nuclei. 
From the segmentation masks of the cell nuclei it is possible to delete pixels of the original image where the cells were located.
Then, the color of the surroundings of the eliminated pixels is propagated into to the removed pixels in order to smooth out the color of the vascular network.
However, a proper dataset has not been found for this purpose, so degraded performance were obtained through the dataset provided here https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet.
The cell dataset provided in his repository slightly differs from our RCC dataset, so a simple unet trained on his dataset can't properly do inference on our samples.
To overcome this issue, domain adaptation is used as a means to minimize the difference between our dataset and his:
a domain classifier equal to the one provided in (https://arxiv.org/abs/1505.07818) is attached to the latent representation of the UNet (the layer in between the downsampling and upsampling paths) in order to perform domain adversarial training. The novelty compared to the approach provided in the cited paper is that this approach has been used for a main task of image segmentation task rather
than image classification. This allowed to shift the learned distribution of the latent space representing just (https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet)
to one that also considers our dataset distribution.
This has been necessary since our dataset provides segmentation ground truths for the vascular network but not for the cell segmentation, which are instead provided in the mentioned dataset in the github repository.
In the end, the original unet segmentation performance has not improved when applied to the simplified images, and performed on par to the one trained on the non simplified images. 






