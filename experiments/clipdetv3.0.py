import torch
import os
from tqdm import tqdm
import open_clip
from PIL import Image
from utils.processing import prepare_data
import requests

data_dir = "data"
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
data_df = prepare_data(data_dir)
img_path_table = data_df[["path"]]
model_list = [
    ("ViT-SO400M-14-SigLIP", "webli"),
    ("ViT-SO400M-14-SigLIP-384", "webli"),
    ("ViT-H-14-quickgelu", "dfn5b"),
    ("ViT-H-14-378-quickgelu", "dfn5b"),
]
models_dict = dict()
transforms_dict = dict()
for modelname, dataset in model_list:
    model, _, preprocess = open_clip.create_model_and_transforms(
        modelname, pretrained=dataset
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer(modelname)
    models_dict[modelname] = (model, tokenizer)
    transforms_dict[modelname] = preprocess
with torch.no_grad(), torch.amp.autocast("cuda"):
    batch = {k: list() for k in transforms_dict}
    batch_id = list()
    last_index = img_path_table.index[-1]
    for index in tqdm(img_path_table.index, total=len(img_path_table)):
        filepath = os.path.join(data_dir, img_path_table.loc[index, "path"])
        image = Image.open(filepath)
        for k in transforms_dict:
            batch[k].append(transforms_dict[k](Image.open(filepath)).to(device))
        # image = preprocess(Image.open(filepath)).to(device)
        # batch.append(image)
        batch_id.append(index)
        if (len(batch_id) >= batch_size) or (index == last_index):
            for k in transforms_dict:
                batch[k] = torch.stack(batch[k], 0)
            for modelname in models_dict.keys():
                text = models_dict[modelname][1](["Real", "AI generated"]).to(device)
                image_features = models_dict[modelname][0].encode_image(
                    batch[modelname]
                )
                text_features = models_dict[modelname][0].encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                for ind, prob in zip(batch_id, text_probs):
                    img_path_table.loc[ind, modelname] = prob[1].item() - 0.5

            batch = {k: list() for k in transforms_dict}
            batch_id = list()
img_path_table.to_csv("../csvs/clipdetv3.0.csv", index=False)
