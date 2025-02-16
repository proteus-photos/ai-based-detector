import argparse
import torch
import wandb
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from utils.processing import set_random_seed, prepare_data
from torch.utils.data import DataLoader
from pathlib import Path
from utils.helper import TrainValDataset, LinearSVM, initialize_models, InferDataset
from adversarial.attacks import apgd_train as apgd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune entire model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/trainclip",
        help="path of directory that contains data in two folders i.e. real and ai-gen",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--weight_dir",
        type=str,
        default="weights",
        help="directory to store weights of trained models.",
    )
    parser.add_argument("--train", action="store_true", help="Used to train the model")
    parser.add_argument("--infer", action="store_true", help="Used to run inference")
    parser.add_argument("--val", action="store_true", help="Used to run validation")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Whether to postprocess images or not",
    )
    parser.add_argument(
        "--next_to_last",
        action="store_true",
        help="Whether to take features from next to last layer or not.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--modelname", type=str, required=True, help="name of model")
    parser.add_argument(
        "--inner_loss",
        type=str,
        default="ce",
        help="loss for generating adversarial examples",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Used to evaluate on adversarial examples",
    )
    return parser.parse_args()


def train(
    models_dict,
    img_path_table,
    transforms_dict,
    data_dir,
    batch_size,
    output_dir,
    epochs=10,
    lr=1e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    wandb.login(key=os.environ["WANDB_API_KEY"])
    for modelname in models_dict.keys():
        wandb.init(
            project="clip-svm-finetuning",
            config={"batch_size": batch_size, "epochs": epochs, "learning_rate": lr},
            name=modelname,
        )
        traindataset = TrainValDataset(
            img_path_table, transforms_dict, modelname, data_dir
        )
        traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
        # Initialize Linear SVM (instead of scikit-learn)
        # Custom logic for only models listed in model_list. Change to add more models
        if modelname.startswith("ViT-SO400M"):
            svm = LinearSVM(in_features=1152)
        elif modelname.startswith("ViT-H"):
            svm = LinearSVM(in_features=1280)
        else:
            print("Model Not Supported")

        # Use optimizer to update both CLIP and SVM
        optimizer = optim.Adam(
            list(models_dict[modelname].parameters()) + list(svm.parameters()), lr
        )

        checkpoint_base_path = output_dir / f"joint_model_{modelname}_adv15000"

        model = models_dict[modelname]
        model.to(device)
        svm.to(device)
        checkpoint = torch.load(
            output_dir
            / "ViT-H-14-quickgelu_dfn5b_imagenet_l2_imagenet_FARE4_ZlXOM/checkpoints/step_15000.pt"
        )

        # Uncooment to load checkpoint in the format saved in this file
        # model.load_state_dict(checkpoint['clip_model'])
        # svm.load_state_dict(checkpoint['svm_model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        # Uncomment for loading visual backbone checkpoint from adversarial training
        checkpoint.pop("proj", None)
        model.visual.load_state_dict(checkpoint)

        model.train()  # Set CLIP to train mode
        svm.train()  # Set SVM to train mode
        for epoch in range(epochs):
            total_loss = 0
            batch_idx = 0
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
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                    }
                )

            epoch_loss = total_loss / len(traindataloader)
            wandb.log({"epoch_loss": epoch_loss})
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")
            if (epoch + 1) % 1 == 0:
                checkpoint_path = f"{checkpoint_base_path}_epoch{epoch + 1}.pth"
                torch.save(
                    {
                        "clip_model": model.state_dict(),
                        "svm_model": svm.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at {checkpoint_path}")

        wandb.finish()


def validate_checkpoints(
    models_dict,
    img_path_table,
    transforms_dict,
    data_dir,
    batch_size,
    device,
    checkpoint_dir,
    epochs=5,
):
    for modelname in models_dict.keys():
        wandb.init(
            project="clip-svm-validation-adv",
            config={
                "batch_size": batch_size,
                "epochs": epochs,
            },
            name=modelname + "_15000",
        )
        model = models_dict[modelname]
        valdataset = TrainValDataset(
            img_path_table, transforms_dict, modelname, data_dir
        )
        valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
        svm = LinearSVM(in_features=1280).to(device)
        for epoch in range(epochs):
            checkpoint_name = f"joint_model_{modelname}_adv15000_epoch{epoch+1}.pth"
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, checkpoint_name), map_location=device
            )
            model.load_state_dict(checkpoint["clip_model"])
            svm.load_state_dict(checkpoint["svm_model"])
            model.eval()
            svm.eval()
            total_loss = 0
            for images, labels in tqdm(valdataloader):
                images, labels = images.to(device), labels.to(device).float()
                with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                    features = model.visual.forward(images)
                    outputs = svm(features).squeeze()
                    # Compute hinge loss
                    loss = svm.hinge_loss(outputs, labels)
                    wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})
                total_loss += loss.item()
            wandb.log({"epoch_loss": total_loss})
        wandb.finish()


def infer(
    model_list,
    img_path_table,
    data_dir,
    batch_size,
    output_dir,
    post_process,
    next_to_last,
    is_train,
    inner_loss,
    attack=None,
):

    def ce(out, targets, reduction="mean"):
        # out = logits
        assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
        assert out.shape[0] > 1

        return F.cross_entropy(out, targets, reduction=reduction)

    def hinge_loss(outputs, labels):
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(
        model_list, post_process, next_to_last
    )
    for modelname in models_dict.keys():
        model = models_dict[modelname].to(device)
        svm = LinearSVM(in_features=1280).to(device)
        checkpoint_name = f"joint_model_{modelname}_epoch3.pth"
        checkpoint = torch.load(
            os.path.join(output_dir, checkpoint_name), map_location=device
        )
        model.load_state_dict(checkpoint["clip_model"])
        svm.load_state_dict(checkpoint["svm_model"])
        model.eval()
        svm.eval()
        all_image_features, all_ids = [], []
        # batch, batch_id = [], []
        last_index = img_path_table.index[-1]
        dataset = InferDataset(img_path_table, data_dir, transforms_dict[modelname])
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

        # Initialize storage
        all_image_features, all_ids = [], []

        class FullModel(nn.Module):
            def __init__(self, feature_extractor, classifier):
                super().__init__()
                self.feature_extractor = feature_extractor  # model.visual
                self.classifier = classifier  # svm

            def forward(self, x):
                features = self.feature_extractor(x)  # Extract features
                outputs = self.classifier(features).squeeze().cpu()  # Apply SVM
                return outputs

        full_model = FullModel(model.visual, svm).eval()
        # Process batches

        for batch, batch_id, targets in tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device)
            print("BATCH")
            print(batch.shape)
            print(targets.shape)
            if attack == "apgd":
                batch = apgd(
                    model=full_model,
                    loss_fn=hinge_loss,
                    x=batch,
                    y=targets,
                    norm="linf",
                    eps=4,
                    n_iter=100,
                    verbose=True,
                )

            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                features = model.visual.forward(batch)
                outputs = svm(features).squeeze().cpu()
                all_image_features.extend(outputs.flatten())
                all_ids.extend(batch_id.tolist())
            # batch, batch_id = [], []

        all_image_features = np.array(all_image_features)

        modelname_column = f"joint_model_{modelname}_epoch3"
        final_table = img_path_table[["path"]]
        for ii, logit in zip(all_ids, all_image_features):
            final_table.loc[ii, modelname_column] = logit
        final_table.to_csv("csvs_adv/nonrobust_hinge.csv", index=False)


def main():
    args = parse_arguments()
    set_random_seed()

    # Prepare data
    data_df = prepare_data(args.data_dir)
    data_df["label"] = np.where(data_df["type"] == "real", -1, 1)
    img_path_table = data_df[["path", "label"]]

    model_list = [
        ("ViT-SO400M-14-SigLIP", "webli"),  #
        ("ViT-SO400M-14-SigLIP-384", "webli"),
        ("ViT-H-14-quickgelu", "dfn5b"),
        ("ViT-H-14-378-quickgelu", "dfn5b"),  #
    ]
    filtered_list = [item for item in model_list if item[0] == args.modelname]
    models_dict, transforms_dict = initialize_models(
        model_list, post_process, next_to_last
    )
    if args.train:
        train(
            models_dict,
            img_path_table,
            transforms_dict,
            args.data_dir,
            args.batch_size,
            args.weight_dir,
            args.epochs,
            args.lr,
        )
    elif args.infer:
        infer(
            filtered_list,
            img_path_table,
            args.data_dir,
            args.batch_size,
            args.weight_dir,
            args.postprocess,
            args.next_to_last,
            args.train,
            args.inner_loss,
            args.attack,
        )
    elif args.val:
        validate_checkpoints(
            models_dict,
            img_path_table,
            transforms_dict,
            args.data_dir,
            args.batch_size,
            device,
            args.weight_dir,
            args.epochs,
        )


if __name__ == "__main__":
    main()
