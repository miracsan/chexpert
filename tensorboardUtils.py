# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:09:42 2019

@author: Mirac
"""
from torchvision import transforms
from makePredictions import *
from torch.utils.tensorboard import SummaryWriter
import torch
import os

def tensorboard_writer(last_epoch=11, uncertainty='weighted_multiclass'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
              'train': transforms.Compose([
                  transforms.RandomHorizontalFlip(),
                  transforms.Scale(224),
                  # because scale doesn't always give 224 x 224, this ensures 224 x
                  # 224
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
              ]),
              'valid': transforms.Compose([
                  transforms.Scale(224),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
              ]),
          }
    
    PATH_TO_MAIN_FOLDER = "/content"
    UNCERTAINTY = 'weighted_multiclass'
    N_LABELS = 14
    
    writer = SummaryWriter()
    
    for i in range(last_epoch):
      checkpoint = torch.load(os.path.join('results',uncertainty,'checkpoint'+str(i+1)))
      model = checkpoint['model']
      model = model.cuda()
      writer.add_scalar('Val Loss', checkpoint['epoch_loss'], i+1)
      
      preds, aucs = make_pred_multilabel(data_transforms,
                                             model,
                                             PATH_TO_MAIN_FOLDER,
                                             UNCERTAINTY,
                                             N_LABELS,
                                             epoch=i+1)
      
      aucs_numpy = aucs['auc'].to_numpy()
      writer.add_scalar('Atelectasis', aucs_numpy[0], i+1)
      writer.add_scalar('Cardiomegaly', aucs_numpy[1], i+1)
      writer.add_scalar('Consolidation', aucs_numpy[2], i+1)
      writer.add_scalar('Edema', aucs_numpy[3], i+1)
      writer.add_scalar('Enl Cardio', aucs_numpy[4], i+1)
      writer.add_scalar('Lung Lesion', aucs_numpy[6], i+1)
      writer.add_scalar('Lung Opacity', aucs_numpy[7], i+1)
      writer.add_scalar('No Finding', aucs_numpy[8], i+1)
      writer.add_scalar('Pleural Effusion', aucs_numpy[9], i+1)
      writer.add_scalar('Pleural Other', aucs_numpy[10], i+1)
      writer.add_scalar('Pneumonia', aucs_numpy[11], i+1)
      writer.add_scalar('Pneumothorax', aucs_numpy[12], i+1)
      writer.add_scalar('Support Devices', aucs_numpy[13], i+1)
      writer.add_scalar('Total AUC', np.sum(aucs_numpy[[0,1,2,3,4,6,7,8,9,10,11,12,13]])/13, i+1)
      
    writer.close()