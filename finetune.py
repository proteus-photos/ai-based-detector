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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune entire model')
    parser.add_argument('--data_dir', type=str, default='data/trainclip', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='finetune_weights', help='directory to store weights of trained models.')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--modelname', type=str, required=True, help='name of model')
    return parser.parse_args()

def initialize_models(model_list, device, post_process, next_to_last, is_train):
    models_dict = {}
    transforms_dict = {}
    for modelname, dataset in model_list:
        transform = []
        if post_process:
            transform += [RandomSizeCrop(min_scale=0.625, max_scale=1.0), Resize((200, 200), interpolation=InterpolationMode.BICUBIC), rand_jpeg_compression]
        if is_train:#Load pretrained model for finetuning !    
            model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
        else:#Load random weights initially;later load checkpoints
            model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=None)
        model.to(device)
        if next_to_last:
            model.visual.proj = None
            #model.visual.head = None
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
def train_clip_svm(model, svm, dataloader, device, optimizer,checkpoint_base_path, epochs=10, checkpoint_interval =1):
    model.train()  # Set CLIP to train mode
    svm.train()    # Set SVM to train mode
    for epoch in range(epochs):
        total_loss = 0
        batch_idx=0
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
            batch_idx += 1
            wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1, "batch": batch_idx + 1})

        epoch_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss":epoch_loss})
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path =  f"{checkpoint_base_path}_epoch{epoch + 1}.pth"
            torch.save({
                'clip_model': model.state_dict(),
                'svm_model': svm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")


# Modify SVM training to jointly train CLIP backbone and SVM
def joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs=10, lr=1e-4):

    wandb.init(project="clip-svm-finetuning", config={"batch_size": batch_size, "epochs": epochs,"learning_rate":lr})
    for modelname in models_dict.keys():
        traindataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
        traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
        # Initialize Linear SVM (instead of scikit-learn)
        
        svm = LinearSVM(in_features=1280).to(device)

        # Use optimizer to update both CLIP and SVM
        optimizer = optim.Adam(list(models_dict[modelname].parameters()) + list(svm.parameters()), lr)
        checkpoint_path = os.path.join(output_dir, f"joint_model_{modelname}")
        # Jointly train CLIP and SVM
        train_clip_svm(models_dict[modelname], svm, traindataloader, device, optimizer,checkpoint_path, epochs=epochs)

        # Save model and optimizer state
        torch.save({
            'clip_model': models_dict[modelname].state_dict(),
            'svm_model': svm.state_dict(),
            'optimizer': optimizer.state_dict(),
        },  f"{checkpoint_path}.pth")
    wandb.finish()

def validate_checkpoints(models_dict,img_path_table, transforms_dict, data_dir, batch_size, device,checkpoint_dir,epochs=5):
    for modelname in models_dict.keys():
        wandb.init(project="clip-svm-validation", config={"batch_size": batch_size, "epochs": epochs,},name=modelname)
        model = models_dict[modelname]
        valdataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
        valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
        # in_features =  model.visual.ln_post.normalized_shape[0]
        svm = LinearSVM(in_features=1152).to(device)
        for epoch in range(epochs):
            checkpoint_name =f'joint_model_{modelname}_epoch{epoch+1}.pth'
            checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_name), map_location=device)
            model.load_state_dict(checkpoint['clip_model'])
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
            wandb.log({"epoch_loss":total_loss})
        wandb.finish()

def infer(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, final_table,alpha=0):
    for modelname in models_dict.keys():
        model = models_dict[modelname]
        svm = LinearSVM(in_features=1280).to(device)
        checkpoint_name =f'joint_model_{modelname}_epoch4.pth'
        checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_name), map_location=device)
        model.load_state_dict(checkpoint['clip_model'])
        svm.load_state_dict(checkpoint['svm_model'])
        model.eval()
        svm.eval()
        all_image_features, all_ids = [], []
        batch, batch_id = [], []
        last_index = img_path_table.index[-1]
        for index in tqdm(img_path_table.index, total=len(img_path_table)):
            filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
            image = Image.open(filepath)
            transformed_image = transforms_dict[modelname](image).to(device)
            batch.append(transformed_image)
            batch_id.append(index)

            if len(batch) >= batch_size or index == last_index:
                batch = torch.stack(batch, dim=0)
                features = model.visual.forward(images)
                outputs = svm(features).squeeze()
                all_image_features.append(features)
                all_ids.extend(batch_id)
                batch, batch_id = [], []

        all_image_features = np.vstack(all_image_features)
        
        modelname_column = f'joint_model_{modelname}_epoch4'
        for ii, logit in zip(all_ids, all_image_features):
            final_table.loc[ii, modelname_column] = logit
        final_table.to_csv(args.out_csv, index=False)

def main():
    args = parse_arguments()
    set_random_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df = prepare_data(args.data_dir)
    data_df['label'] = np.where(data_df['type'] == 'real', -1, 1)
    img_path_table = data_df[['path', 'label']]
    model_list = [
        # ('ViT-SO400M-14-SigLIP', 'webli'),#
        # ('ViT-SO400M-14-SigLIP-384', 'webli'),
        ('ViT-H-14-quickgelu', 'dfn5b'),
        ('ViT-H-14-378-quickgelu', 'dfn5b'),#
    ]
    filtered_list = [item for item in model_list if item[0]==args.modelname]
    models_dict, transforms_dict = initialize_models(filtered_list, device, args.postprocess, args.next_to_last,args.train)
    print(models_dict.keys())
    print(args.train)
    if args.train:
        joint_train_clip_svm(models_dict, img_path_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir, args.epochs, args.lr)
    else:
        if not args.infer:
            validate_checkpoints(models_dict,img_path_table, transforms_dict,  args.data_dir, args.batch_size,  device, args.weight_dir, args.epochs)
        else:
            final_table = data_df[['path']]
            infer(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, final_table)

if __name__ == "__main__":
    main()
