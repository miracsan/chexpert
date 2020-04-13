# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:15:47 2019

@author: Mirac
"""
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
import csv
from createCheckpoint import create_checkpoint
from makePredictions import make_pred_multilabel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pos_neg_weights_in_batch(labels_batch):

    num_total = labels_batch.shape[0] * labels_batch.shape[1]
    num_positives = labels_batch.sum()
    num_negatives = num_total - num_positives

    if not num_positives == 0:
        beta_p = num_negatives / num_positives
    else:
        beta_p = num_negatives
    # beta_p = torch.tensor(beta_p)
    beta_p = beta_p.to(device)
    beta_p = beta_p.type(torch.cuda.FloatTensor)

    return beta_p

def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        weight_decay,
        batch_size,
        uncertainty,
        starting_epoch=0):
    """
    Performs the actual training

    Args:
        model: model to be trained
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
        uncertainty: the uncertainty method to be used in training
        starting_epoch: specify the number of previous epochs if continuing 
            training from a checkpoint
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1 + starting_epoch
    num_epochs = num_epochs + starting_epoch
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1
    
    dataset_sizes = {x: len(dataloaders[x]) * dataloaders[x].batch_size for x in ['train', 'val']}
    print("Dataset sizes:", dataset_sizes)

    

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)

                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    if uncertainty == "multiclass":
                        outputs_reshaped = outputs.view(-1, 3)
                        labels_reshaped = (labels.view(-1) + 1).type(torch.long)
                        loss = criterion(outputs_reshaped, labels_reshaped)
                    elif uncertainty == 'weighted_multiclass':
                        outputs_reshaped = outputs.view(-1, 3)
                        labels_reshaped = (labels.view(-1) + 1).type(torch.long)
                        loss = criterion(outputs_reshaped, labels_reshaped)
                    elif uncertainty == 'batchwise_zeros':
                        weights = pos_neg_weights_in_batch(labels)
                        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
                        loss = criterion(outputs, labels)
                    elif uncertainty == 'effective_num_gradnorm_zeros':
                        #import pdb
                        #pdb.set_trace()
                        loss = criterion(outputs, labels, model)
                        if loss!=loss:
                            print("nan VAAAAR")
                            print(bobo)
                        #pdb.set_trace()
                    else:
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * batch_size 
                    
                sys.stdout.write("\r Progress in the epoch:     %.3f" % (i * batch_size / dataset_sizes[phase] * 100)) #keep track of the progress
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            '''
            if phase == 'valid' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))
            '''
            # checkpoint model if has best val loss yet
            if phase == 'val':   
                create_checkpoint(model, epoch_loss, epoch, LR, uncertainty, criterion)
                #preds, aucs = make_pred_multilabel(dataloader=dataloaders["val"],
                #                       model=model,
                #                       UNCERTAINTY=uncertainty,
                #                       epoch=epoch)
                #aucs.set_index('label', inplace=True)
                #print(aucs)
                if epoch_loss < best_loss:
                  best_loss = epoch_loss
                  best_epoch = epoch

            # log training and validation loss over each epoch
            if phase == 'val':
                with open(os.path.join('results',uncertainty,'log_train'), 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        

        # break if no val loss improvement in 3 epochs
        '''
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break
        '''
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(os.path.join('results',uncertainty,'checkpoint'+str(best_epoch)))
    model = checkpoint_best['model']

    return model, best_epoch