from chexpertDataset import *
from makePredictions import *
from torchvision import transforms
import torch

PATH_TO_MAIN_FOLDER = "../"
UNCERTAINTY = "effective_num_focal_zeros"
epoch = 6
MODEL_PATH = os.path.join('results', UNCERTAINTY, 'checkpoint' + str(epoch))

BATCH_SIZE = 32
# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# define torchvision transforms
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
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


transformed_datasets = createDatasets(PATH_TO_MAIN_FOLDER, data_transforms, UNCERTAINTY)

dataloader = torch.utils.data.DataLoader(transformed_datasets['test'],
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
saved_model = torch.load(MODEL_PATH)
model = saved_model['model']
model.to(device)
del saved_model

_, auc = make_pred_multilabel(dataloader, model, UNCERTAINTY, epoch, save_as_csv=True)
extract_prec_recall_curves(UNCERTAINTY, epoch)
extract_separation_curves(UNCERTAINTY, epoch)
print(auc)
