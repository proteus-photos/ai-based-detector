import modal
from utils.processing import RandomSizeCrop, rand_jpeg_compression, set_random_seed, prepare_data
from pathlib import Path
# from torch.utils.data import DataLoader, Dataset  

app = modal.App('proteus')

# Define the Modal image
finetune_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("wandb")
)

wandb_secret = modal.Secret.from_name('wandb-secret')

# Imports specific to the image environment
with finetune_image.imports():
    import torch
    import argparse
    import wandb
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    from torchvision.transforms import Compose, Resize, InterpolationMode
    import numpy as np
    import open_clip
    from torch.utils.data import DataLoader, Dataset  
    from PIL import Image
    import os

volume = modal.Volume.from_name("finetune-volume", create_if_missing=True)
VOL_PATH = Path("/finetune_volume")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune entire model')
    parser.add_argument('--data_dir', type=str, default='data/trainclip', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='finetune_weights64', help='directory to store weights of trained models.')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--modelname', type=str, required=True, help='name of model')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')  # Add this
    return parser.parse_args()



@app.function(image=finetune_image,secrets=[wandb_secret], volumes={VOL_PATH: volume}, gpu = "a100-80gb",timeout=24*60*60 )
def finetune(model_list,img_path_table,data_dir,batch_size,device,output_dir,post_process,next_to_last,is_train,epochs=10, lr=1e-5):

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

    class LinearSVM(nn.Module):
        def __init__(self, in_features):
            super(LinearSVM, self).__init__()
            self.fc = nn.Linear(in_features, 1)  # Single output for binary classification

        def forward(self, x):
            return self.fc(x)

        def hinge_loss(self, outputs, labels):
            return torch.mean(torch.clamp(1 - outputs * labels, min=0))

    def initialize_models(model_list, post_process, next_to_last, is_train):
        print("INITIALISING")
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
            if next_to_last:
                model.visual.proj = None
                #model.visual.head = None
            models_dict[modelname] = model
            transforms_dict[modelname] = Compose(transform + [preprocess])
        return models_dict, transforms_dict        

    def joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs=10, lr=1e-5):
        print("COOOOL")

        data_dir = VOL_PATH / data_dir
        output_dir = VOL_PATH / output_dir
        output_dir.mkdir(exist_ok=True)
        wandb.login(key=os.environ["WANDB_API_KEY"])
        for modelname in models_dict.keys():

            wandb.init(project="clip-svm-finetuning-modal", config={"batch_size": batch_size, "epochs": epochs,"learning_rate":lr},name=modelname, id='79uavdmf', resume='must')
            # wandb.init(project="clip-svm-finetuning-modal", config={"batch_size": batch_size, "epochs": epochs,"learning_rate":lr},name=modelname)


            traindataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
            traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
            # Initialize Linear SVM (instead of scikit-learn)
            svm = LinearSVM(in_features=1280)

            # Use optimizer to update both CLIP and SVM
            optimizer = optim.Adam(list(models_dict[modelname].parameters()) + list(svm.parameters()), lr)
            checkpoint_base_path = output_dir / f"joint_model_{modelname}"
            # Jointly train CLIP and SVM

            model = models_dict[modelname]
            model.to(device)
            svm.to(device)
            ##########
            checkpoint = torch.load(output_dir /"joint_model_ViT-H-14-quickgelu_epoch24.pth")
            model.load_state_dict(checkpoint['clip_model'])
            svm.load_state_dict(checkpoint['svm_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #########
            
            model.train()  # Set CLIP to train mode
            svm.train()    # Set SVM to train mode
            for epoch in range(24,epochs):
                total_loss = 0
                batch_idx=0
                print("HELLO")
                for images, labels in tqdm(traindataloader):
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

                epoch_loss = total_loss / len(traindataloader)
                wandb.log({"epoch_loss":epoch_loss})
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")
                if (epoch + 1) % 1 == 0:
                    checkpoint_path =  f"{checkpoint_base_path}_epoch{epoch + 1}.pth"
                    torch.save({
                        'clip_model': model.state_dict(),
                        'svm_model': svm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

            wandb.finish()
        
    models_dict, transforms_dict = initialize_models(model_list,  post_process,next_to_last, is_train)
    print(models_dict.keys())
    joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs, lr)




   
# def train_ddp(rank, world_size, models_dict, img_path_table, transforms_dict, data_dir, batch_size, output_dir, epochs, lr):
#     from torch.nn.parallel import DistributedDataParallel as DDP
#     import torch.distributed as dist
#     import wandb
#     import os
#     from tqdm import tqdm
#     import torch.optim as optim
#     print("AAAAAAAA")

#     def setup_ddp(rank, world_size):
#         dist.init_process_group("nccl", rank=rank, world_size=world_size)
#         torch.cuda.set_device(rank)

#     def cleanup_ddp():
#         dist.destroy_process_group()

#     class TrainValDataset(Dataset):
#         def __init__(self, img_path_table, transforms_dict, modelname, data_dir):
#             self.img_path_table = img_path_table
#             self.transforms_dict = transforms_dict
#             self.modelname = modelname
#             self.data_dir = data_dir

#         def __len__(self):
#             return len(self.img_path_table)

#         def __getitem__(self, index):
#             filepath = os.path.join(self.data_dir, self.img_path_table.iloc[index]['path'])
#             label = self.img_path_table.iloc[index]['label']
#             image = Image.open(filepath)
#             transformed_image = self.transforms_dict[self.modelname](image)
#             return transformed_image, label
#     class LinearSVM(nn.Module):
#         def __init__(self, in_features):
#             super(LinearSVM, self).__init__()
#             self.fc = nn.Linear(in_features, 1)  # Single output for binary classification

#         def forward(self, x):
#             return self.fc(x)

#         def hinge_loss(self, outputs, labels):
#             return torch.mean(torch.clamp(1 - outputs * labels, min=0))

#     setup_ddp(rank, world_size)
#     print("BBBBBBBBB")

#     # Keep everything else as it was in the original train_ddp
#     data_dir = VOL_PATH / data_dir
#     output_dir = VOL_PATH / output_dir
#     output_dir.mkdir(exist_ok=True)
#     wandb.login(key=os.environ["WANDB_API_KEY"])

#     # models_dict, transforms_dict = initialize_models(model_list, transforms_dict, data_dir)

#     for modelname, model in models_dict.items():
#         wandb.init(project="clip-svm-finetuning-modal", config={"batch_size": batch_size, "epochs": epochs, "learning_rate": lr}, name=modelname)

#         dataset = TrainValDataset(img_path_table, transforms_dict, modelname, data_dir)
#         sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#         dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

#         svm = LinearSVM(in_features=1280)
#         optimizer = optim.Adam(list(model.parameters()) + list(svm.parameters()), lr)

#         model = model.to(rank)
#         svm = svm.to(rank)
#         model = DDP(model, device_ids=[rank])
#         svm = DDP(svm, device_ids=[rank])

#         model.train()
#         svm.train()
#         accumulation_steps = 8
#         for epoch in range(epochs):
#             sampler.set_epoch(epoch)
#             total_loss = 0
#             optimizer.zero_grad()
#             for i,(images, labels) in tqdm(enumerate(dataloader)):
#                 images, labels = images.to(rank), labels.to(rank).float()

#                 features = model.module.visual(images)
#                 outputs = svm(features).squeeze()

#                 loss = svm.hinge_loss(outputs, labels)
#                 loss = loss / accumulation_steps  

#                 loss.backward()

#                 if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
#                     optimizer.step()
#                     optimizer.zero_grad()

#                 total_loss += loss.item() * accumulation_steps  

#             epoch_loss = total_loss / len(dataloader)
#             wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})
#             print(f"[Rank {rank}] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")

#             if rank == 0 and (epoch + 1) % 1 == 0:
#                 checkpoint_path = f"{output_dir}/joint_model_{modelname}_epoch{epoch + 1}.pth"
#                 torch.save({'clip_model': model.module.state_dict(), 'svm_model': svm.module.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
#                 print(f"Checkpoint saved at {checkpoint_path}")

#         wandb.finish()

#     cleanup_ddp()

# @app.function(image=finetune_image, secrets=[wandb_secret], volumes={VOL_PATH: volume}, gpu="a100-80gb:3", timeout=24*60*60)
# def finetune_ddp(rank, world_size, model_list, img_path_table, data_dir, batch_size, output_dir, post_process, next_to_last, is_train, epochs=10, lr=1e-5):
#     import torch.multiprocessing as mp
#     import socket

#     # Set MASTER_ADDR to the hostname of the current container (rank 0)
#     hostname = socket.gethostname()
#     master_addr = socket.gethostbyname(hostname)
#     os.environ["MASTER_ADDR"] = master_addr
#     os.environ["MASTER_PORT"] = "12355"

#     def initialize_models(model_list, post_process, next_to_last, is_train):

#         print("INITIALISING")
#         models_dict = {}
#         transforms_dict = {}
#         for modelname, dataset in model_list:
#             transform = []
#             if post_process:
#                 transform += [RandomSizeCrop(min_scale=0.625, max_scale=1.0), Resize((200, 200), interpolation=InterpolationMode.BICUBIC), rand_jpeg_compression]
#             if is_train:
#                 model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
#             else:
#                 model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=None)
#             if next_to_last:
#                 model.visual.proj = None
#             models_dict[modelname] = model
#             transforms_dict[modelname] = Compose(transform + [preprocess])
#         return models_dict, transforms_dict  

#     # Initialize the necessary components for training
#     models_dict, transforms_dict = initialize_models(model_list, post_process, next_to_last, is_train)
#     print("INITIALIZED")
#     # Spawn processes for DDP training
#     mp.spawn(train_ddp, args=(world_size, models_dict, img_path_table, transforms_dict, data_dir, batch_size, output_dir, epochs, lr), nprocs=world_size)


@app.local_entrypoint()
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
    print(args.train)
    testl=[]
    if args.train:
        with modal.enable_output():
            with app.run():
                finetune.remote(filtered_list, img_path_table, args.data_dir, args.batch_size, device, args.weight_dir,args.postprocess,args.next_to_last,args.train, args.epochs, args.lr)
                # finetune_ddp.remote(0,3,filtered_list, img_path_table, args.data_dir, args.batch_size, args.weight_dir,args.postprocess,args.next_to_last,args.train, args.epochs, args.lr)

    
if __name__ == "__main__":
    main()