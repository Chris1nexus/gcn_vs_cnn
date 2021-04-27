import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

from log_utils import GradcamLogger
from processing_utils import UnNormalize





#gradcam/gradcam++/scorecam
def test_classifier_gradcam(model, target_layer, dataloader, 
                             standardization_img_means, 
                             standardization_img_std,
                             gradcam_method= 'gradcam++'):
  # non interactive backend for plotting images
  #curr_backend =  plt.rcParams['backend']
  matplotlib.use('agg')
  mapper_to_categorical = dataloader.dataset.dataset.mapping_id_to_label

  device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
  use_cuda = False

  if 'cuda' in str(device):
    use_cuda = True

  model.eval()

  cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
  
  softmax = nn.Softmax(dim=1)
  figures = []
  gradcam_logs = []
  for (path_img, path_seg, img, seg, seg_gt), label in dataloader:
    img = img.float().to(device)
    label = label.long().to(device)
    y_preds = model(img)
   
    probabilities = softmax(y_preds)
    
    y_pred_binarized = probabilities.argmax(dim=1)
    
    corrects = torch.sum(y_pred_binarized == label).data.item() 


    
    for idx in range(len(img)):

      cuda_img = img[idx].unsqueeze(0)
 
      confidence_proba = probabilities[idx]

      ground_truth = label[idx]
      predicted =  confidence_proba.argmax().data.item()
      not_predicted = 1 - predicted

      curr_mask = seg_gt[idx].squeeze()
  
      correct = (label[idx] ==predicted)
 
      unnorm = UnNormalize(standardization_img_means, standardization_img_std)
      
      method = gradcam_method # Can be gradcam/gradcam++/scorecam

      input_tensor = img[idx].unsqueeze(0)# Create an input tensor image for the model
      rgb_img = (unnorm(img[idx].cpu()).permute(1,2,0).numpy()).clip(0.,1.)#rgb_img = unnorm(img[idx].cpu()).permute(1,2,0).numpy()
      
      grayscale_cam_predicted = cam(input_tensor=input_tensor, target_category=predicted)
      visualization_predicted = show_cam_on_image(rgb_img, grayscale_cam_predicted)

      grayscale_cam_not_predicted = cam(input_tensor=input_tensor, target_category=not_predicted)
      visualization_not_predicted = show_cam_on_image(rgb_img, grayscale_cam_not_predicted)

      if correct:
        correct_prediction_str = "correct"
      else:
        correct_prediction_str = "wrong"



      fig = plt.figure(figsize=(14,10))
      gradcam_predicted = fig.add_subplot(131)
      gradcam_predicted.imshow(visualization_predicted)
      gradcam_predicted.title.set_text('ROI of the prediction: ' + str(mapper_to_categorical[predicted]) + " probability=%.1f"%(confidence_proba[predicted]*100) \
                                       + "%\n" + correct_prediction_str + " prediction")
      raw = fig.add_subplot(132)

      if 1 in rgb_img.shape:
        raw.imshow(rgb_img.squeeze(), cmap='gray')
      else:
        raw.imshow(rgb_img.squeeze())
      raw.title.set_text('Raw slide patch: '+ str(mapper_to_categorical[ground_truth]))
      #gradcam_not_predicted = fig.add_subplot(133)
      #gradcam_not_predicted.imshow(visualization_not_predicted)
      #gradcam_not_predicted.title.set_text('ROI of the excluded prediction: '+ str(mapper_to_categorical[not_predicted]) )
      #fig.canvas.draw()
      #pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
      figures.append(fig)
      
      gradlog = GradcamLogger(fig, mapper_to_categorical.copy(), confidence_proba.cpu().detach().numpy().squeeze()
                                  ,label[idx] )
      gradcam_logs.append(gradlog)
      
    del img
    del label

  
  return gradcam_logs, figures