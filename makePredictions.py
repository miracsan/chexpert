# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:51 2019

@author: Mirac
"""


import torch
import pandas as pd
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import rmtree



def make_pred_multilabel(dataloader, model, UNCERTAINTY="zeros", epoch=0, save_as_csv=False):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # calc preds in batches of 16, can reduce if your GPU has less RAM
    batch_size = dataloader.batch_size
    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)

        true_labels = labels.cpu().data.numpy()
        # batch_size = true_labels.shape

        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, true_labels.shape[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]
            truerow["Image Index"] = dataloader.dataset.df.index[batch_size * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataloader.dataset.PRED_LABEL)):
                thisrow["prob_" + dataloader.dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataloader.dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        # if(i % 10 == 0):
        #     print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
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
            'Support Devices']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        thisrow['AP'] = np.nan
        
        try:
            thisrow['auc'] = sklm.roc_auc_score(actual.values.astype(int), pred.values)
            thisrow['AP'] = sklm.average_precision_score(actual.values.astype(int), pred.values)
            
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    if save_as_csv:
        pred_df.to_csv(os.path.join("results", UNCERTAINTY, "preds{0}.csv".format(epoch)), index=False)
        auc_df.to_csv(os.path.join("results", UNCERTAINTY, "aucs{0}.csv".format(epoch)), index=False)
        true_df.to_csv(os.path.join("results", UNCERTAINTY, "true.csv"), index=False)
        

    return pred_df, auc_df



def extract_prec_recall_curves(UNCERTAINTY="zeros", epoch=0):
    try:
        rmtree(os.path.join('results', UNCERTAINTY, "precision recall"))
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs(os.path.join('results', UNCERTAINTY, "precision recall"))
    
    true_df = pd.read_csv(os.path.join("results", UNCERTAINTY, "true.csv"))
    pred_df = pd.read_csv(os.path.join("results", UNCERTAINTY, "preds{0}.csv".format(epoch)))

    try:
        true_df = true_df.drop(['Image Index'], axis=1)
        pred_df = pred_df.drop(['Image Index'], axis=1)
    except:
        pass

    true_df_columns = true_df.keys()
    pred_df_columns = pred_df.keys()

    if len(true_df_columns) != len(pred_df_columns):
        raise KeyError("Number of columns in pred and ground-truth do not match ")

    for pathology_num in range(len(true_df_columns)):
        pathology = true_df_columns[pathology_num]
        pred_key = pred_df_columns[pathology_num]

        true_array = true_df[pathology].to_numpy()
        pred_array = pred_df[pred_key].to_numpy()

        precision, recall, _ = sklm.precision_recall_curve(true_array, pred_array)
        
        plt.figure(dpi=500)
        plt.plot(recall, precision, label=pathology)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        plt.savefig(os.path.join("results", UNCERTAINTY, "precision recall", "{0}.png".format(pathology)))
        plt.clf()
        
        
def extract_separation_curves(UNCERTAINTY="zeros", epoch=0):
    try:
        rmtree(os.path.join('results',UNCERTAINTY,'separation'))
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs(os.path.join('results',UNCERTAINTY,'separation'))
    
    true_df = pd.read_csv(os.path.join("results", UNCERTAINTY, "true.csv"))
    pred_df = pd.read_csv(os.path.join("results", UNCERTAINTY, "preds{0}.csv".format(epoch)))

    try:
        true_df = true_df.drop(['Image Index'], axis=1)
        pred_df = pred_df.drop(['Image Index'], axis=1)
    except:
        pass

    true_df_columns = true_df.keys()
    pred_df_columns = pred_df.keys()

    if len(true_df_columns) != len(pred_df_columns):
        raise KeyError("Number of columns in pred and ground-truth do not match ")

    for pathology_num in range(len(true_df_columns)):
        pathology = true_df_columns[pathology_num]
        pred_key = pred_df_columns[pathology_num]

        true_array = true_df[pathology].to_numpy()
        pred_array = pred_df[pred_key].to_numpy()

        positives = pred_array[true_array==1]
        negatives = pred_array[true_array==0]

        
        plt.figure(dpi=500)
        sns.distplot(positives, kde=True, hist=True, norm_hist=True, label="positives")
        sns.distplot(negatives, kde=True, hist=True, norm_hist=True, label="negatives")
        plt.xlabel("values")
        plt.ylabel("occurrence")
        plt.legend(loc="best")
        plt.title("Separation for {0}".format(pathology))
        plt.savefig(os.path.join("results", UNCERTAINTY, "separation", "{0}.png".format(pathology)))
        plt.clf()
