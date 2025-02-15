from torch.utils.data import Dataset 
import torch.nn as nn
import torch
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
from utils.processing import RandomSizeCrop, rand_jpeg_compression
import os
from PIL import Image


class TrainValDataset(Dataset):
    def __init__(self, img_path_table, transforms_dict, modelname, data_dir):
        self.img_path_table = img_path_table
        self.transforms_dict = transforms_dict
        self.modelname = modelname
        self.data_dir = data_dir

    def __len__(self):
        return len(self.img_path_table)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.img_path_table.iloc[index]['path'])
        label = self.img_path_table.iloc[index]['label']
        image = Image.open(filepath)
        transformed_image = self.transforms_dict[self.modelname](image)
        return transformed_image, label

class InferDataset(Dataset):
    def __init__(self, img_path_table, data_dir, transform):
        self.img_path_table = img_path_table
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_path_table)

    def __getitem__(self, idx):
        index = self.img_path_table.index[idx]
        filepath = os.path.join(self.data_dir, self.img_path_table.loc[index, 'path'])
        label = self.img_path_table.iloc[index]['label']
        image = Image.open(filepath)
        image = self.transform(image)
        return image, index,label

class LinearSVM(nn.Module):
    def __init__(self, in_features):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(in_features, 1)  # Single output for binary classification

    def forward(self, x):
        return self.fc(x)

    def hinge_loss(self, outputs, labels):
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))

def initialize_models(model_list, post_process, next_to_last):
    print("INITIALISING")
    models_dict = {}
    transforms_dict = {}
    for modelname, dataset in model_list:
        transform = []
        if post_process:
            transform += [RandomSizeCrop(min_scale=0.625, max_scale=1.0), Resize((200, 200), interpolation=InterpolationMode.BICUBIC), rand_jpeg_compression]
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
        if next_to_last:
            model.visual.proj = None
            #model.visual.head = None
        models_dict[modelname] = model
        transforms_dict[modelname] = Compose(transform + [preprocess])
    return models_dict, transforms_dict  