

import torch.nn as nn

class ExperimentLogger(object):

  def __init__(self):
    self.log_dictionary = {}
  
  def log(self, key_name, value):
    assert self.log_dictionary.get(key_name, None) is None, "Error: key has already been inserted"
    self.log_dictionary[key_name] = value
  
  def keys(self):
    return self.log_dictionary.keys()





class GradcamLogger(object):
  def __init__(self, figure, mapping, probabilities, ground_truth, img_path, seg_path):
    self.figure = figure
    self.mapping = mapping
    self.probabilities = probabilities
    self.ground_truth = ground_truth
    self.img_path = img_path
    self.seg_path = seg_path


class OverlaySegmentationLogger(object):
  def __init__(self, figure, mapping, IOU):
    self.figure = figure
    self.mapping = mapping
    self.IOU = IOU
  


















class Logger(object):

    def __init__(self, log_dir): 
        self.writer = tf.summary.create_file_writer(log_dir)#tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step ) 
            
            self.writer.flush()

    
def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)
def log_images(x, y_true, y_pred, channel=1):
    images = []
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(x_np.shape[0]):
        image = gray2rgb(np.squeeze(x_np[i]))
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc