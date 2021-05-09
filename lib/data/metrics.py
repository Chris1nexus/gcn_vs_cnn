from sklearn.metrics import jaccard_score as jsc


def IoU(y_pred, y_true):
  '''
  Args:
  		-y_pred (torch.Tensor) compute iou from given argument
  		-y_true (torch.Tensor)
  Returns:
  		-iou (float)
  '''
  labels = y_true.cpu().detach().numpy().reshape(-1)
  targets = y_pred.cpu().detach().numpy().reshape(-1)
  return jsc(targets, labels)
