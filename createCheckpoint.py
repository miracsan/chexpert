# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:11:18 2019

@author: Mirac
"""
import torch
import os

def create_checkpoint(model, epoch_loss, epoch, LR, uncertainty):
    """
    Saves a checkpoint of the model

    Args:
        model: model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
        uncertainty: the uncertainty method used in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'epoch_loss': epoch_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'uncertainty': uncertainty 
    }

    torch.save(state, os.path.join('results',uncertainty,'checkpoint'+str(epoch)))