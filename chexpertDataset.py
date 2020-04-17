# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:10:18 2019

@author: Mirac
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split

class CheXpertDataset(Dataset):
    """
    Dataset class for the chexpert dataset
    
    Args:
        path_to_main_folder: path where the extracted chexpert data is located
        fold: choose 'train', 'valid' or 'test'
        transform: torchvision transforms to be applied to raw images
        uncertainty: the uncertainty method to be used in training
        
    """

    def __init__(
            self,
            path_to_main_folder,
            fold,
            transform=None,
            uncertainty="zeros"):

        self.transform = transform
        self.path_to_main_folder = path_to_main_folder
        
        if fold == 'test':   #Use the validation set in the chexpert dataset as the test set
            self.df = pd.read_csv(os.path.join(path_to_main_folder,
                                               'CheXpert-v1.0-small',
                                               'valid.csv'))
        elif fold == 'train': #Use 80% of the train set in the chexpert dataset as the train set
            self.df = pd.read_csv(os.path.join(path_to_main_folder,
                                           'CheXpert-v1.0-small',
                                           'train.csv'))
            self.df, _ = train_test_split(self.df, test_size=0.2, random_state=42)
        elif fold == 'val':  #Use 20% of the train set in the chexpert dataset as the validation set
            self.df = pd.read_csv(os.path.join(path_to_main_folder,
                                           'CheXpert-v1.0-small',
                                           'train.csv'))
            _, self.df = train_test_split(self.df, test_size=0.2, random_state=42)
                
                
        
        self.df = self.df.set_index("Path") #Use the path of the image directory as the index
        self.df = self.df.drop(['Sex', 'Age', 'Frontal/Lateral','AP/PA'], axis=1) #Drop these columns because we won't be needing them
        self.df = self.df.fillna(value=0) #Fill in the missing values with zeros (negative)
        
        if "zeros" in uncertainty: #If the zeros uncertainty method is used, treat all uncertain labels as zeros
            self.df = self.df.replace(-1, 0)
        elif "ones" in uncertainty: #If the ones uncertainty method is used, treat all uncertain labels as ones
            self.df = self.df.replace(-1, 1)
        
        
        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']  #These are the 14 labels we try to predict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_main_folder, 
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') in set([-1,1])):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label,self.df.index[idx])
      
    def getWeights(self, uncertainty='weighted_zeros'): #The weight array for weighted cross entropy methods
        
        if uncertainty == 'weighted_zeros':
          positives = self.df.sum(axis=0)[-14:] / self.df.shape[0]
          negatives = 1 - positives
          weights = 2 * negatives
          
        elif uncertainty == 'weighted_multiclass':  
          counts = self.df.count()[-14:]
          uncertains = self.df.isin([-1]).sum(axis=0)[-14:]
          positives = self.df.isin([1]).sum(axis=0)[-14:]
          negatives = self.df.isin([0]).sum(axis=0)[-14:]
          weights = [(counts - uncertains) / counts * 3/2, \
                     (counts - negatives)  / counts * 3/2, \
                     (counts - positives) / counts  * 3/2] 
          weights = np.transpose(np.array(weights))
        
        return weights
        
    def effective_num_weights(self, beta=0.9999, mode='intra'):
        num_pos_samples = []
        num_neg_samples = []


        for i in range(0, len(self.PRED_LABEL)):
            num_pos_samples.append(self.df[self.df[self.PRED_LABEL[i].strip()] == 1].shape[0])
            num_neg_samples.append(self.df[self.df[self.PRED_LABEL[i].strip()] == 0].shape[0])
            
        pos_denominator_term = 1.0 - np.power(beta, num_pos_samples);
        neg_denominator_term = 1.0 - np.power(beta, num_neg_samples);
        pos_weights = (1.0 - beta) / np.array(pos_denominator_term);
        neg_weights = (1.0 - beta) / np.array(neg_denominator_term);
        
        if mode == 'inter':
            weights_interclass_normalized = pos_weights / np.sum(pos_weights) * len(pos_weights)
            weights = weights_interclass_normalized
        
        else:
        
            #weights_intraclass_normalized = pos_weights / (pos_weights + neg_weights) * 2; Bence bu yanlış
            weights_intraclass_normalized = pos_weights / (neg_weights + 0.001)
            weights = weights_intraclass_normalized
            #weights = torch.Tensor(weights_intraclass_normalized)
            #weights = weights.cuda()
            #weights = weights.type(torch.cuda.FloatTensor)
        return weights
    
    def pos_neg_sample_nums(self):
        df = self.df[self.PRED_LABEL]
        N, C = df.shape
        pos_neg_sample_nums = np.zeros((2,C))
        pos_neg_sample_nums[0, :] = df.sum(axis=0).values
        pos_neg_sample_nums[1, :] = N - pos_neg_sample_nums[0, :]
        del df
        return pos_neg_sample_nums


def createDatasets(PATH_TO_MAIN_FOLDER, data_transforms, UNCERTAINTY):
    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold='train',
        transform=data_transforms['train'],
        uncertainty=UNCERTAINTY)
    transformed_datasets['val'] = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold='val',
        transform=data_transforms['val'],
        uncertainty=UNCERTAINTY)
    transformed_datasets['test'] = CheXpertDataset(
        path_to_main_folder=PATH_TO_MAIN_FOLDER,
        fold='test',
        transform=data_transforms['val'],
        uncertainty=UNCERTAINTY)

    return transformed_datasets