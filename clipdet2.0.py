import torch
import os
from tqdm import tqdm
import clip
from PIL import Image
from utils.processing import prepare_data
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

data_dir ='data'
batch_size =1
device = "cuda" if torch.cuda.is_available() else "cpu"
data_df = prepare_data(data_dir)
img_path_table = data_df[['path']]
model_name='clipdetv2.5'
# print(clip.available_models())
model, preprocess= clip.load('ViT-L/14', device)
# image = preprocess(Image.open('data/ai-gen/dalle2/r0a2e85f0t.png')).unsqueeze(0).to(device)

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  

with torch.no_grad():
    batch =list()
    batch_id=list()
    last_index = img_path_table.index[-1]
    for index in tqdm(img_path_table.index, total=len(img_path_table)):
        filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
        image = Image.open(filepath)
        blip_inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**blip_inputs)
        
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        text = clip.tokenize([generated_text+"This image is Real", generated_text+"This image is AI generated"]).to(device)
        image = preprocess(Image.open(filepath)).to(device)
        batch.append(image)
        batch_id.append(index)
        if (len(batch)>=batch_size) or (index == last_index):
            batch = torch.stack(batch,0)
            logits_per_image, logits_per_text = model(batch, text)
            probs = logits_per_image.softmax(dim =-1).cpu().numpy()
            for ind, prob in zip(batch_id, probs):
                img_path_table.loc[ind, model_name] = prob[1] - 0.5
            # for ind,logit in zip(batch_id, logits_per_image):
            #     img_path_table.loc[ind, model_name] = logit[1].item()
            batch =list()
            batch_id=list()
            break

img_path_table.to_csv('csvs/clipdetv2.5.csv',index=False)