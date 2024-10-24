import argparse
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
from torchvision.transforms import Compose, Resize, InterpolationMode
from PIL import Image
import os
import numpy as np
import open_clip
from utils.processing import RandomSizeCrop, rand_jpeg_compression, set_random_seed, prepare_data
from torch.utils.data import DataLoader, Dataset

class TrainDataset(Dataset):
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune entire model')
    parser.add_argument('--data_dir', type=str, default='data/trainsvm', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='finetune_weights', help='directory to store weights of trained models.')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

def initialize_models(model_list, device, post_process, next_to_last):
    models_dict = {}
    transforms_dict = {}
    for modelname, dataset in model_list:
        transform = []
        if post_process:
            transform += [RandomSizeCrop(min_scale=0.625, max_scale=1.0), Resize((200, 200), interpolation=InterpolationMode.BICUBIC), rand_jpeg_compression]
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
        model.to(device)
        if next_to_last:
            model.visual.proj = None
        models_dict[modelname] = model
        transforms_dict[modelname] = Compose(transform + [preprocess])
    return models_dict, transforms_dict

# Custom Linear SVM Layer (using hinge loss)
class LinearSVM(nn.Module):
    def __init__(self, in_features):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(in_features, 1)  # Single output for binary classification

    def forward(self, x):
        return self.fc(x)

    def hinge_loss(self, outputs, labels):
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))


# Training loop for joint fine-tuning CLIP and SVM
def train_clip_svm(model, svm, dataloader, device, optimizer, epochs=10):
    model.train()  # Set CLIP to train mode
    svm.train()    # Set SVM to train mode
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()

            # Forward pass through CLIP backbone
            features = model.visual.forward(images)
            # Forward pass through SVM
            outputs = svm(features).squeeze()

            # Compute hinge loss
            loss = svm.hinge_loss(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")


# Modify SVM training to jointly train CLIP backbone and SVM
def joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs=10, lr=1e-4):

    wandb.init(project="clip-svm-finetuning", config={"batch_size": batch_size, "epochs": epochs,"learning_rate":lr})
    for modelname in models_dict.keys():
        dataset = TrainDataset(img_path_table, transforms_dict, modelname, data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Initialize Linear SVM (instead of scikit-learn)
        
        svm = LinearSVM(in_features=1280).to(device)

        # Use optimizer to update both CLIP and SVM
        optimizer = optim.Adam(list(models_dict[modelname].parameters()) + list(svm.parameters()), lr)

        # Jointly train CLIP and SVM
        train_clip_svm(models_dict[modelname], svm, dataloader, device, optimizer, epochs=epochs)

        # Save model and optimizer state
        torch.save({
            'clip_model': models_dict[modelname].state_dict(),
            'svm_model': svm.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(output_dir, f"joint_model_{modelname}.pth"))
    wandb.finish()


# Modified main function for joint fine-tuning
def main():
    args = parse_arguments()
    set_random_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df = prepare_data(args.data_dir)
    data_df['label'] = np.where(data_df['type'] == 'real', -1, 1)
    img_path_table = data_df[['path', 'label']]
    model_list = [
        # ('ViT-SO400M-14-SigLIP', 'webli'),
        # ('ViT-SO400M-14-SigLIP-384', 'webli'),
        # ('ViT-H-14-quickgelu', 'dfn5b'),
        ('ViT-H-14-378-quickgelu', 'dfn5b'),
    ]

    models_dict, transforms_dict = initialize_models(model_list, device, args.postprocess, args.next_to_last)
    print(models_dict.keys())
    print(args.train)
    if args.train:
        joint_train_clip_svm(models_dict, img_path_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir, args.epochs, args.lr)
    else:
        print("Evaluation mode not implemented for joint fine-tuning yet.")
        # Implement evaluation if needed

if __name__ == "__main__":
    main()
