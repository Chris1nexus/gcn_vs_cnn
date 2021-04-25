import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

from log_utils import ExperimentLogger


def train_classifier(log_weights_path,  train_dataloader, val_dataloader, model, learning_rate=0.0001,n_epochs=10, verbose=True,weights_filename="torch_weights.pt"):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    torch.cuda.empty_cache() 
    
    if not os.path.exists(log_weights_path):
      os.makedirs(log_weights_path)
    
   
    loaders = {"train": train_dataloader, "valid": val_dataloader}
  
    criterion = nn.CrossEntropyLoss()


    model.to(device)

    acc_train = []
    acc_validation = []
    best_validaton_BCE = sys.float_info.max
    #0.0001 goood
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    loss_train = []
    loss_validation = []
    
    logger = ExperimentLogger()

    step = 0
    epochs = n_epochs
    for epoch in tqdm(range(epochs), total=epochs,position=0, leave=True):
        loss_train_epoch = []
        loss_validation_epoch = []
        acc_train_epoch = []
        acc_validation_epoch = []
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                (path_img, path_seg, img, seg, seg_gt),label  = data

                img, labels  = img.float().to(device), label.long().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(img)
                   
                    loss = criterion(y_pred,labels)
                    
                    logsoftmax = nn.LogSoftmax(dim=1)
                    y_pred_binarized = logsoftmax(y_pred).argmax(dim=1)

                    corrects = torch.sum(y_pred_binarized == labels).data.item() 
                    accuracy = corrects/len(labels)
                    if phase == "valid":
                        loss_validation_epoch.append(loss.item())
                        acc_validation_epoch.append(accuracy)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        loss_train_epoch.append(loss.item())
                        acc_train_epoch.append(accuracy)
                    del img
                    del seg_gt
                    del labels
                    del y_pred
            if phase == "train":
                    loss_train.append(np.mean(loss_train_epoch))
                    acc_train.append(np.mean(acc_train_epoch))
                   
                    loss_train_epoch = []
                    acc_train_epoch = []
            
            if phase == "valid":
               
                mean_validation_loss = np.mean(loss_validation_epoch)
                if mean_validation_loss < best_validaton_BCE:
                  best_validaton_BCE = mean_validation_loss
                  torch.save(model.state_dict(), os.path.join(log_weights_path, weights_filename))
                if verbose:
                	print("valid loss: ", mean_validation_loss)
                	print("acc_valid: ", np.mean(acc_validation_epoch))
                
                loss_validation.append(mean_validation_loss)
                acc_validation.append(np.mean(acc_validation_epoch))
                
                loss_valid_epoch = []
                acc_validation_epoch = []
    logger.log('train_accuracy', acc_train)
    logger.log('validation_accuracy', acc_validation)
    logger.log('train_loss', loss_train)
    logger.log('validation_loss', loss_validation)
    logger.log('model', model)

    return loss_train, loss_validation, acc_train, acc_validation, model

def test_classifier(model, test_dataloader):
  model.eval()
  device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
  acc_test =[]
  
  for (path_img, path_seg, img, seg, seg_gt), labels in test_dataloader:

    img = img.float().to(device)
    # long() is important as transformation may have distorted some pixels
    labels = labels.long().to(device)
    out = model(img)

    logsoftmax = nn.LogSoftmax(dim=1)
    preds = logsoftmax(out).argmax(dim=1)

    accuracy = (preds == labels).sum()/len(img)
    acc_test.append( accuracy  )
    del seg
    del img
    del seg_gt
    del out
    
  return acc_test



