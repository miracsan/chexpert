#imports
from imports import *
from chexpertDataset import *
from createCheckpoint import create_checkpoint
from lossFunctions import *
from makePredictions import make_pred_multilabel
from trainCnn import train_cnn
from trainModel import train_model
from camUtils import *
from tensorboardUtils import *

USE_MODEL = torch.load(os.path.join('results','effective_num_focal_zeros','checkpoint'+'5'))
#USE_MODEL = 0
UNCERTAINTY = "effective_num_focal_zeros"

PATH_TO_IMAGES = "../"
WEIGHT_DECAY = 1e-4

#LEARNING_RATE = 0.001 
#USE_MODEL = checkpoint_last
#preds, aucs = train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, USE_MODEL, UNCERTAINTY="zeros")

LEARNING_RATE = 0.001

train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, USE_MODEL=USE_MODEL, UNCERTAINTY=UNCERTAINTY)
