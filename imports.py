# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:09:30 2019

@author: Mirac
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree
import sys

# data science imports
import pandas as pd
import numpy as np
import csv

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt