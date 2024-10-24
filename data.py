import os
import random 
import shutil
import argparse
from PIL import Image
from diffusers import AutoPipelineForText2Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

parser = argparse.ArgumentParser(description="Generate real-aigen pair from real data")

parser.add_argument('--source_dir', type=str, required=True, help='Path to the source directory containing images.')
parser.add_argument('--real_dir', type=str, required=True, help='Directory to save real images.')
parser.add_argument('--aigen_dir', type=str, required=True, help='Directory to save AI-generated images.')
parser.add_argument('--num_img', type=int, required=True, help='Number of image pairs required')

args=parser.parse_args()

source_dir = args.source_dir
real_dir = args.real_dir
aigen_dir = args.aigen_dir
num_pairs = args.num_img

os.makedirs(real_dir, exist_ok=True)
os.makedirs(aigen_dir, exist_ok=True)
random.seed(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

list_img = os.listdir(source_dir)
random_list = random.sample(list_img,num_pairs)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)

for filename in random_list:
    source_path =os.path.join(source_dir,filename)
    shutil.copy(source_path,real_dir)
    image = Image.open(source_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    generated_image = pipeline_text2image(prompt=prompt).images[0]
    generated_image.save(os.path.join(aigen_dir, filename))


