# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:02 2019

@author: Mirac
"""
from imports import *
from chexpertDataset import createDatasets
from lossFunctions import *
import os
from shutil import rmtree
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from trainModel import train_model
from makePredictions import make_pred_multilabel
import torch.optim as optim

def train_cnn(PATH_TO_MAIN_FOLDER, LR, WEIGHT_DECAY, USE_MODEL=0,UNCERTAINTY="zeros", use_gpu=1):
    """
    Train a model with chexpert data using the given hyperparameters

    Args:
        PATH_TO_MAIN_FOLDER: path where the extracted chexpert data is located
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD
        UNCERTAINTY: the uncertainty method to be used in training
        USE_MODEL: specify the checkpoint object if you want to continue 
            training  

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    N_LABELS = 14  # we are predicting 14 labels
    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    
    
    if USE_MODEL == 0:
        try:
            rmtree(os.path.join('results',UNCERTAINTY))
        except BaseException:
            pass  # directory doesn't yet exist, no need to clear it
        os.makedirs(os.path.join('results',UNCERTAINTY))

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")


    if not USE_MODEL == 0:
        model = USE_MODEL['model']
        starting_epoch = USE_MODEL['epoch']
        UNCERTAINTY = USE_MODEL['uncertainty']
    else:
        starting_epoch = 0
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        # add final layer with # outputs in same dimension of labels with sigmoid
        # activation
        if UNCERTAINTY in ["multiclass", 'weighted_multiclass']:
            model.classifier = nn.Linear(num_ftrs, 3 * N_LABELS)
        else:
            model.classifier = nn.Linear(num_ftrs, N_LABELS)

    # put model on GPU
    if not USE_MODEL:
        model = nn.DataParallel(model)
    model = model.cuda()


    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(320),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }


    transformed_datasets = createDatasets(PATH_TO_MAIN_FOLDER, data_transforms, UNCERTAINTY)
    
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)


    # define criterion, optimizer for training
    #TODO: Reimplement the loss functions (include sigmoid or BCEWithLogitsLoss)
    if UNCERTAINTY == "ignore":
        criterion = BCEwithIgnore()
    elif UNCERTAINTY == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif UNCERTAINTY == 'weighted_multiclass':
        label_weights = torch.tensor(transformed_datasets['train'].getWeights(uncertainty='weighted_multiclass'))
        label_weights = label_weights.to(torch.device("cuda"))
        criterion = WeightedCrossEntropy(label_weights)
    elif UNCERTAINTY == "weighted_zeros":
        label_weights = torch.tensor(transformed_datasets['train'].getWeights())
        label_weights = label_weights.to(torch.device("cuda"))
        criterion = WeightedBCE(label_weights)
    elif UNCERTAINTY == 'effective_num_zeros':
        intra_class_weights=transformed_datasets['train'].effective_num_weights()
        print("intra_class_weights: {0}".format(intra_class_weights))
        criterion = nn.BCEWithLogitsLoss(pos_weight=intra_class_weights)
    elif UNCERTAINTY == 'effective_num_gradnorm_zeros':
        intra_class_weights=transformed_datasets['train'].effective_num_weights()
        print("intra_class_weights: {0}".format(intra_class_weights))
        criterion = EffectiveNumGradNormLoss(model=model, pos_weight=intra_class_weights)
    elif UNCERTAINTY == 'focal_zeros':
        criterion = WeightedFocalLoss(inter_class_weights=np.ones(N_LABELS), intra_class_weights=np.ones(N_LABELS))
    elif UNCERTAINTY == 'effective_num_focal_zeros':
        intra_class_weights=transformed_datasets['train'].effective_num_weights()
        criterion = WeightedFocalLoss(inter_class_weights=np.ones(N_LABELS), intra_class_weights=intra_class_weights)     
    elif UNCERTAINTY == 'focal_gradnorm_zeros':
        criterion = FocalGradNormLoss(model=model, N_LABELS=N_LABELS)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    #optimizer = optim.SGD(
    #    filter(
    #        lambda p: p.requires_grad,
    #        model.parameters()),
    #    lr=LR,
    #    momentum=0.9,
    #    weight_decay=WEIGHT_DECAY)
    
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY)


    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR,
                                    num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders,
                                    weight_decay=WEIGHT_DECAY,
                                    batch_size=BATCH_SIZE,
                                    uncertainty=UNCERTAINTY,
                                    starting_epoch=starting_epoch)

    # get preds and AUCs on test fold
    #preds, aucs = make_pred_multilabel(dataset=transformed_datasets["test"],
    #                                   model=model,
    #                                   UNCERTAINTY=UNCERTAINTY,
    #                                   epoch=starting_epoch+NUM_EPOCHS)

    #return preds, aucs
