import modal
from utils.processing import set_random_seed, prepare_data
from pathlib import Path

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
    import torch.optim as optim
    import torch.nn.functional as F
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    import numpy as np
    from torch.utils.data import DataLoader 
    from PIL import Image
    import os
    from utils.helper import TrainValDataset, LinearSVM, initialize_models, InferDataset
    from adversarial.attacks import apgd_train as apgd

volume = modal.Volume.from_name("finetune-volume", create_if_missing=True)
VOL_PATH = Path("/finetune_volume")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune entire model')
    parser.add_argument('--data_dir', type=str, default='data/trainclip', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='finetune_weights64', help='directory to store weights of trained models.')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--infer', action='store_true', help='Used to infer from the model')
    parser.add_argument('--attack', type=str, default=None, help='Used to evaluate on adversarial examples')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--modelname', type=str, required=True, help='name of model')
    parser.add_argument('--inner_loss',type=str, default='ce',help='loss for generating adversarial examples')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training') 
    return parser.parse_args()



@app.function(image=finetune_image,secrets=[wandb_secret], volumes={VOL_PATH: volume}, gpu = "a100-80gb",timeout=24*60*60 )
def finetune(model_list,img_path_table,data_dir,batch_size,output_dir,post_process,next_to_last,is_train,epochs=10, lr=1e-5):

    def joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs=10, lr=1e-5):
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(model_list,  post_process,next_to_last, is_train)
    print(models_dict.keys())
    joint_train_clip_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, epochs, lr)



@app.function(image=finetune_image,secrets=[wandb_secret], volumes={VOL_PATH: volume}, gpu = "a100-40gb",timeout=24*60*60 )
def infer(model_list,img_path_table,data_dir,batch_size,output_dir,post_process,next_to_last,is_train,attack=None,inner_loss):

    def ce(out, targets, reduction='mean'):
        # out = logits
        assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
        assert out.shape[0] > 1

        return F.cross_entropy(out, targets, reduction=reduction)

    def hinge_loss(outputs, labels):
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))

    # def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
    #                 embedding_text_labels_norm=None, reduction='mean'):
    #     if loss_str == 'l2':
    #         loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    #     elif loss_str == 'ce':
    #         loss = ce(
    #             out=embedding @ (logit_scale * embedding_text_labels_norm),
    #             targets=targets,
    #             reduction=reduction
    #         )
    #     elif loss_str == 'hinge':
    #         loss = hinge_loss(embedding,targets)
    #     else:
    #         raise ValueError(f'loss {loss_str} not supported')
    #     return loss


    # class ComputeLossWrapper:
    #     def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', loss=None,
    #                 logit_scale=100.):
    #         self.embedding_orig = embedding_orig
    #         self.embedding_text_labels_norm = embedding_text_labels_norm
    #         self.reduction = reduction
    #         self.loss_str = loss
    #         self.logit_scale = logit_scale

    #     def __call__(self, embedding, targets):
    #         return compute_loss(
    #             loss_str=self.loss_str, embedding=embedding, targets=targets,
    #             embedding_orig=self.embedding_orig, logit_scale=self.logit_scale,
    #             embedding_text_labels_norm=self.embedding_text_labels_norm, reduction=self.reduction
    #             )

    # loss_inner_wrapper = ComputeLossWrapper(
    #     embedding_orig, embedding_text_labels_norm=None,
    #     reduction='none' if attack == 'apgd' else 'mean', loss=inner_loss,
    #     logit_scale=100.
    #     )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(model_list,  post_process,next_to_last, is_train)
    for modelname in models_dict.keys():
        model = models_dict[modelname]
        svm = LinearSVM(in_features=1280).to(device)
        checkpoint_name =f'joint_model_{modelname}_epoch3.pth'
        checkpoint = torch.load(os.path.join(output_dir,checkpoint_name), map_location=device)
        model.load_state_dict(checkpoint['clip_model'])
        svm.load_state_dict(checkpoint['svm_model'])
        model.eval()
        svm.eval()
        all_image_features, all_ids = [], []
        # batch, batch_id = [], []
        last_index = img_path_table.index[-1]
        # for index in tqdm(img_path_table.index, total=len(img_path_table)):
        #     filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
        #     image = Image.open(filepath)
        #     transformed_image = transforms_dict[modelname](image).to(device)
        #     batch.append(transformed_image)
        #     batch_id.append(index)

        #     if len(batch) >= batch_size or index == last_index:
        #         batch = torch.stack(batch, dim=0)
        #         with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        #             features = model.visual.forward(batch)
        #             outputs = svm(features).squeeze().cpu()
        #         all_image_features.extend(outputs.flatten())
        #         all_ids.extend(batch_id)
        dataset = ImageDataset(img_path_table, data_dir, transforms_dict[modelname])
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        # Initialize storage
        all_image_features, all_ids = [], []

        # Process batches
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for batch, batch_id, targets in tqdm(dataloader, total=len(dataloader)):
                if attack=='apgd':
                    batch = apgd(
                    model=model,
                    loss_fn=hinge_loss if inner_loss=='hinge' else raise NotImplemented,
                    x=batch,
                    y=targets,
                    norm='linf',
                    eps=4,
                    n_iter=args.iterations_adv,#FIX THIS
                    verbose=True

                batch = batch.to(device)  # Move batch to GPU
                features = model.visual.forward(batch)
                outputs = svm(features).squeeze().cpu()
                all_image_features.extend(outputs.flatten())
                all_ids.extend(batch_id.tolist())
                # batch, batch_id = [], []

        all_image_features = np.array(all_image_features)
        
        modelname_column = f'joint_model_{modelname}_epoch3'
        for ii, logit in zip(all_ids, all_image_features):
            final_table.loc[ii, modelname_column] = logit
        final_table.to_csv('csvs_adv/nonrobust_hinge.csv', index=False)





@app.local_entrypoint()
def main():
    args = parse_arguments()
    set_random_seed()
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
                finetune.remote(filtered_list, img_path_table, args.data_dir, args.batch_size, args.weight_dir,args.postprocess,args.next_to_last,args.train, args.epochs, args.lr)
                # finetune_ddp.remote(0,3,filtered_list, img_path_table, args.data_dir, args.batch_size, args.weight_dir,args.postprocess,args.next_to_last,args.train, args.epochs, args.lr)

    elif args.infer:
        with modal.enable_output():
            with app.run():
                infer.remote(filtered_list, img_path_table, args.data_dir, args.batch_size, args.weight_dir,args.postprocess,args.next_to_last,args.train, args.attack, args.inner_loss)

    
if __name__ == "__main__":
    main()