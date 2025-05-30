import modal
from utils.processing import set_random_seed, prepare_data
from pathlib import Path

app = modal.App("proteus")

# Define the Modal image
finetune_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("wandb")
)

wandb_secret = modal.Secret.from_name("wandb-secret")

# Imports specific to the image environment
with finetune_image.imports():
    import torch
    import argparse
    import wandb
    import torch.optim as optim
    import torch.nn.functional as F
    import torch.nn as nn
    from tqdm import tqdm
    import numpy as np
    from torch.utils.data import DataLoader
    from PIL import Image
    import os
    from utils.helper import (
        TrainValDataset,
        LinearSVM,
        initialize_models,
        InferDataset,
        FullModel,
    )
    from adversarial.attacks import apgd_train as apgd
    from utils.processing import set_random_seed, prepare_data
    from utils.losses import ce, hinge_loss

# Setup volumes for MODAL
volume = modal.Volume.from_name("finetune-volume", create_if_missing=True)
VOL_PATH = Path("/finetune_volume")
out_volume = modal.Volume.from_name("out-volume", create_if_missing=True)
OUT_PATH = Path("/out_volume")


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
        default="finetune_weights64",
        help="directory to store weights of trained models.",
    )
    parser.add_argument("--train", action="store_true", help="Used to train the model")
    parser.add_argument(
        "--infer", action="store_true", help="Used to infer from the model"
    )
    parser.add_argument("--val", action="store_true", help="Used to validate the model")
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Used to evaluate on adversarial examples",
    )
    parser.add_argument(
        "--iterations_adv",
        type=int,
        default=10,
        help="Iterations for adversarial attack",
    )
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
        "--csv_path",
        type=str,
        default="out.csv",
        help="Path for output logits during inference",
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=4,
        help="Eps",
    )

    return parser.parse_args()


@app.function(
    image=finetune_image,
    secrets=[wandb_secret],
    volumes={VOL_PATH: volume, OUT_PATH: out_volume},
    gpu="a100-40gb",
    timeout=24 * 60 * 60,
)
def train(
    model_list,
    data_dir,
    batch_size,
    output_dir,
    post_process,
    next_to_last,
    epochs=10,
    lr=1e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(
        model_list, post_process, next_to_last
    )
    data_dir = VOL_PATH / data_dir
    data_df = prepare_data(data_dir)
    data_df["label"] = np.where(data_df["type"] == "real", -1, 1)
    img_path_table = data_df[["path", "label"]]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    wandb.login(key=os.environ["WANDB_API_KEY"])
    for modelname in models_dict.keys():
        wandb.init(
            project="clip-svm-finetuning-modal-adv",
            config={"batch_size": batch_size, "epochs": epochs, "learning_rate": lr},
            name=modelname + "_20000",
        )
        traindataset = TrainValDataset(
            img_path_table, transforms_dict, modelname, data_dir
        )
        traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

        # Initialize Linear SVM (instead of scikit-learn)
        # Custom logic for only models listed in model_list. Change to add more models
        if modelname.startswith("ViT-SO400M"):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith("ViT-H"):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print("Model Not Supported")

        # Use optimizer to update both CLIP and SVM
        optimizer = optim.Adam(
            list(models_dict[modelname].parameters()) + list(svm.parameters()), lr
        )
        ckp_output = output_dir / "full_model"
        ckp_output.mkdir(exist_ok=True)
        checkpoint_base_path = ckp_output / f"joint_model_{modelname}_adv20000"
        # Jointly train CLIP and SVM

        model = models_dict[modelname].to(device)
        checkpoint = torch.load(
            output_dir
            / "ViT-H-14-quickgelu_dfn5b_imagenet_l2_imagenet_FARE4_ZlXOM/ViT-H-14-quickgelu_dfn5b_imagenet_l2_imagenet_FARE4_ZlXOM/checkpoints/step_20000.pt"
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


@app.function(
    image=finetune_image,
    secrets=[wandb_secret],
    volumes={VOL_PATH: volume, OUT_PATH: out_volume},
    gpu="a100-80gb",
    timeout=24 * 60 * 60,
)
def infer(
    model_list,
    data_dir,
    batch_size,
    output_dir,
    post_process,
    next_to_last,
    inner_loss,
    csv_path,
    attack=None,
    iterations_adv=10,
    eps=4,
):
    data_dir = VOL_PATH / data_dir
    data_df = prepare_data(data_dir)
    data_df["label"] = np.where(data_df["type"] == "real", -1, 1)
    img_path_table = data_df[["path", "label"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(
        model_list, post_process, next_to_last
    )
    for modelname in models_dict.keys():
        model = models_dict[modelname].to(device)
        # Initialize Linear SVM (instead of scikit-learn)
        # Custom logic for only models listed in model_list. Change to add more models
        if modelname.startswith("ViT-SO400M"):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith("ViT-H"):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print("Model Not Supported")
        checkpoint_name = f"joint_model_{modelname}_epoch3.pth"
        checkpoint = torch.load(
            os.path.join(output_dir, checkpoint_name), map_location=device
        )
        model.load_state_dict(checkpoint["clip_model"])
        svm.load_state_dict(checkpoint["svm_model"])

        # # -------------------------------------------------------------------------------------------------------------------------
        # # just for interpolation
        # alpha = 0.5
        # # Load first checkpoint
        # checkpoint_name1 = f"weights/joint_model_{modelname}_epoch3.pth"
        # checkpoint1 = torch.load(
        #     os.path.join(output_dir, checkpoint_name1), map_location=device
        # )

        # # Load second checkpoint
        # checkpoint_name2 = (
        #     f"output_adv/full_model/joint_model_{modelname}_adv15000_epoch4.pth"
        # )
        # checkpoint2 = torch.load(
        #     os.path.join(output_dir, checkpoint_name2), map_location=device
        # )

        # # Interpolate model weights
        # interpolated_clip_state_dict = {
        #     key: (1 - alpha) * checkpoint1["clip_model"][key]
        #     + alpha * checkpoint2["clip_model"][key]
        #     for key in checkpoint1["clip_model"]
        # }

        # # Interpolate SVM weights (if applicable)
        # interpolated_svm_state_dict = {
        #     key: (1 - alpha) * checkpoint1["svm_model"][key]
        #     + alpha * checkpoint2["svm_model"][key]
        #     for key in checkpoint1["svm_model"]
        # }

        # # Load interpolated weights into the model
        # model.load_state_dict(interpolated_clip_state_dict)
        # svm.load_state_dict(interpolated_svm_state_dict)
        # # ---end---
        # # ---end-------------------------------------------------------------------------------------------------------------------------------------

        model.eval()
        svm.eval()
        all_image_features, all_ids = [], []
        # batch, batch_id = [], []
        last_index = img_path_table.index[-1]
        dataset = InferDataset(img_path_table, data_dir, transforms_dict[modelname])
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

        # Initialize storage
        all_image_features, all_ids = [], []
        full_model = FullModel(model.visual, svm).eval()

        # Process batches
        for batch, batch_id, targets in tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device)
            target_for_attack = targets.clone()
            if inner_loss == "hinge":
                loss_fn = hinge_loss
            elif inner_loss == "ce":
                target_for_attack = (target_for_attack + 1) // 2
                loss_fn = ce
            if attack == "apgd":
                batch = apgd(
                    model=full_model,
                    loss_fn=loss_fn,
                    x=batch,
                    y=target_for_attack,
                    norm="linf",
                    eps=eps,
                    n_iter=iterations_adv,
                    verbose=True,
                )

            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                features = model.visual.forward(batch)
                outputs = svm(features).squeeze().cpu()
                all_image_features.extend(outputs.flatten())
                all_ids.extend(batch_id.tolist())

        all_image_features = np.array(all_image_features)

        modelname_column = f"joint_model_{modelname}_epoch3"
        final_table = img_path_table[["path"]]
        for ii, logit in zip(all_ids, all_image_features):
            final_table.loc[ii, modelname_column] = logit
        output_path = OUT_PATH / csv_path
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        final_table.to_csv(output_path, index=False)


@app.function(
    image=finetune_image,
    secrets=[wandb_secret],
    volumes={OUT_PATH: out_volume},
    gpu="a100-40gb",
    timeout=24 * 60 * 60,
)
def validate_checkpoints(
    model_list,
    data_dir,
    batch_size,
    checkpoint_dir,
    post_process,
    next_to_last,
    epochs=5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict, transforms_dict = initialize_models(
        model_list, post_process, next_to_last
    )
    data_dir = OUT_PATH / data_dir
    data_df = prepare_data(data_dir)
    data_df["label"] = np.where(data_df["type"] == "real", -1, 1)
    img_path_table = data_df[["path", "label"]]
    wandb.login(key=os.environ["WANDB_API_KEY"])
    for modelname in models_dict.keys():
        wandb.init(
            project="clip-svm-validation-modal-adv",
            config={
                "batch_size": batch_size,
                "epochs": epochs,
            },
            name=modelname + "_20000",
        )
        model = models_dict[modelname].to(device)
        valdataset = TrainValDataset(
            img_path_table, transforms_dict, modelname, data_dir
        )
        valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
        if modelname.startswith("ViT-SO400M"):
            svm = LinearSVM(in_features=1152).to(device)
        elif modelname.startswith("ViT-H"):
            svm = LinearSVM(in_features=1280).to(device)
        else:
            print("Model Not Supported")
        for epoch in range(epochs):
            checkpoint_name = f"joint_model_{modelname}_adv20000_epoch{epoch+1}.pth"
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


@app.local_entrypoint()
def main():
    args = parse_arguments()
    set_random_seed()
    model_list = [
        ("ViT-SO400M-14-SigLIP", "webli"),
        ("ViT-SO400M-14-SigLIP-384", "webli"),
        ("ViT-H-14-quickgelu", "dfn5b"),
        ("ViT-H-14-378-quickgelu", "dfn5b"),
    ]
    filtered_list = [item for item in model_list if item[0] == args.modelname]
    if args.train:
        with modal.enable_output():
            with app.run(detach=True):
                train.remote(
                    filtered_list,
                    args.data_dir,
                    args.batch_size,
                    args.weight_dir,
                    args.postprocess,
                    args.next_to_last,
                    args.epochs,
                    args.lr,
                )

    elif args.infer:
        with modal.enable_output():
            with app.run(detach=False):
                infer.remote(
                    filtered_list,
                    args.data_dir,
                    args.batch_size,
                    args.weight_dir,
                    args.postprocess,
                    args.next_to_last,
                    args.inner_loss,
                    args.csv_path,
                    args.attack,
                    args.iterations_adv,
                    args.eps,
                )

    elif args.val:
        with modal.enable_output():
            with app.run(detach=True):
                validate_checkpoints.remote(
                    filtered_list,
                    args.data_dir,
                    args.batch_size,
                    args.weight_dir,
                    args.postprocess,
                    args.next_to_last,
                    args.epochs,
                )


if __name__ == "__main__":
    main()
