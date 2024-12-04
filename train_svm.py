import torch
import random
import os
import argparse
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import open_clip
from PIL import Image
from torchvision.transforms import Resize, Compose, InterpolationMode
from utils.processing import prepare_data, RandomSizeCrop, rand_jpeg_compression, set_random_seed
from torch.utils.data import DataLoader, Dataset


# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SVM on top of other clip encoders')
    parser.add_argument('--data_dir', type=str, default='data/trainsvm', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='svm_weights', help='directory to store weights of trained models.')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--infer', action='store_true', help='Used for inference, csv output of logits')
    return parser.parse_args()


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
# Custom Linear SVM Layer (using hinge loss)
class LinearSVM(nn.Module):
    def __init__(self, in_features):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(in_features, 1)  # Single output for binary classification

    def forward(self, x):
        return self.fc(x)

    def hinge_loss(self, outputs, labels):
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))
# Initialize models and transforms
def initialize_models(model_list, device, post_process, next_to_last):
    models_dict = {}
    transforms_dict = {}
    for modelname, dataset in model_list:
        transform = []
        if post_process:
            transform += [RandomSizeCrop(min_scale=0.625, max_scale=1.0), Resize((200, 200), interpolation=InterpolationMode.BICUBIC), rand_jpeg_compression]
   
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
        model.eval()
        model.to(device)
        if next_to_last:
            model.visual.proj = None
            model.visual.head = nn.Identity()
        models_dict[modelname] = model
        transforms_dict[modelname] = Compose(transform + [preprocess])
    return models_dict, transforms_dict



# Feature extraction and SVM training
def train_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs=20,lr=1e-5):
    
    
    for modelname in models_dict.keys():
        wandb.init(project="just-svm-finetuning", config={"batch_size": batch_size, "epochs": epochs,"learning_rate":lr})
        traindataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
        traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
        
        if modelname.startswith('ViT-SO400M'):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith('ViT-H'):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print('Model Not Supported')

        # Use optimizer to update both CLIP and SVM
        optimizer = optim.Adam(list(svm.parameters()), lr)
        checkpoint_base_path = os.path.join(output_dir, f"svm_{modelname}")
        # Jointly train CLIP and SVM
        # train_clip_svm(models_dict[modelname], svm, traindataloader, device, optimizer,checkpoint_path, epochs=epochs)
        model = models_dict[modelname]
        model.to(device)
        model.eval()
        svm.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_idx=0
            for images, labels in tqdm(traindataloader):
                images, labels = images.to(device), labels.to(device).float()

                optimizer.zero_grad()

                with torch.no_grad():
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
                batch_idx += 1
                wandb.log({"train_batch_loss": loss.item(), "epoch": epoch + 1, "batch": batch_idx + 1})
            epoch_loss = total_loss / len(traindataloader)
            wandb.log({"epoch_loss":epoch_loss})
            checkpoint_path =  f"{checkpoint_base_path}_epoch{epoch + 1}.pth"
            torch.save({
                'svm_model': svm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        wandb.finish()

def validate(models_dict,img_path_table, transforms_dict, data_dir, batch_size, device,checkpoint_dir,epochs=5):
    for modelname in models_dict.keys():
        wandb.init(project="just-svm-validation", config={"batch_size": batch_size, "epochs": epochs,},name=modelname)
        valdataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
        valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
        model = models_dict[modelname]
        model.to(device)
        checkpoint_base_path = os.path.join(checkpoint_dir, f"svm_{modelname}")
        if modelname.startswith('ViT-SO400M'):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith('ViT-H'):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print('Model Not Supported')

        for epoch in range(epochs):
            checkpoint_path =f'{checkpoint_base_path}_epoch{epoch+1}.pth'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            svm.load_state_dict(checkpoint['svm_model'])
            model.eval()
            svm.eval()
            total_loss =0
            for images, labels in tqdm(valdataloader):
                images, labels = images.to(device), labels.to(device).float()
                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    features = model.visual.forward(images)
                    outputs = svm(features).squeeze()
                    # Compute hinge loss
                    loss = svm.hinge_loss(outputs, labels)
                    wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})
                total_loss += loss.item()
            epoch_loss = total_loss / len(valdataloader)
            wandb.log({"epoch_loss":epoch_loss})
        wandb.finish()
# Load SVM model and perform evaluation
def infer(models_dict, final_table, transforms_dict, data_dir, batch_size, device, checkpoint_dir, ep):
    for modelname in models_dict.keys():
        model = models_dict[modelname]
        model.to(device)
        if modelname.startswith('ViT-SO400M'):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith('ViT-H'):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print('Model Not Supported')

        checkpoint_name =f'svm_{modelname}_epoch{ep}.pth'
        checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_name), map_location=device)
        svm.load_state_dict(checkpoint['svm_model'])
        model.eval()
        svm.eval()
        all_outputs, all_ids = [], []
        batch, batch_id = [], []
        last_index = final_table.index[-1]
        for index in tqdm(final_table.index, total=len(final_table)):
            filepath = os.path.join(data_dir, final_table.loc[index, 'path'])
            image = Image.open(filepath)
            transformed_image = transforms_dict[modelname](image).to(device)
            batch.append(transformed_image)
            batch_id.append(index)

            if len(batch) >= batch_size or index == last_index:
                batch = torch.stack(batch, dim=0)
                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    features = model.visual.forward(batch)
                    outputs = svm(features).squeeze().cpu().numpy()
                all_outputs.extend(outputs)
                all_ids.extend(batch_id)
                batch, batch_id = [], []

        modelname_column = f'svm_{modelname}_epoch{ep}'
        for ii, logit in zip(all_ids, all_outputs):
            final_table.loc[ii, modelname_column] = logit
        torch.cuda.empty_cache()
    final_table.to_csv(f'csvs_post/just_svm_post.csv', index=False)


def main():
    args = parse_arguments()
    set_random_seed()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df = prepare_data(args.data_dir)
    data_df['label'] = np.where(data_df['type'] == 'real', -1, 1)
    img_path_table = data_df[['path', 'label']]
    model_list = [
        ('ViT-SO400M-14-SigLIP', 'webli'),
        ('ViT-SO400M-14-SigLIP-384', 'webli'),
        ('ViT-H-14-quickgelu', 'dfn5b'),
        ('ViT-H-14-378-quickgelu', 'dfn5b'),
    ]
    
    models_dict, transforms_dict = initialize_models(model_list, device, args.postprocess, args.next_to_last)
    
    if args.train:
        train_svm(models_dict, img_path_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir,args.epochs,args.lr)
    else:
        if not args.infer:
            validate(models_dict,img_path_table, transforms_dict,  args.data_dir, args.batch_size,  device, args.weight_dir, args.epochs)
        else:
            final_table = data_df[['path']]
            infer(models_dict, final_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir, args.epochs)

if __name__ == "__main__":
    main()
