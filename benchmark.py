# ------------------------------------------------------------------------------
# Reference: https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/main.py
# Modified by Aayan Yadav (https://github.com/ydvaayan)
# ------------------------------------------------------------------------------
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
import glob
import sys
import yaml
from PIL import Image
import random

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode, Lambda
from utils.processing import make_normalize, prepare_data
from utils.fusion import apply_fusion
from utils.patch import patch_img
from networks import create_architecture, load_weights



def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir,model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def run_tests(data_dir,weights_dir, models_list, device, batch_size=1):
    data_df = prepare_data(data_dir)
    img_path_table = data_df[['path']]

    models_dict = dict()
    transform_dict = dict()
    print('Models:')
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = create_architecture(arch)
        # # Load the weights
        model = load_weights(model, model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size=='Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif patch_size == 'ssp16':
            transform.append(Lambda(lambda img: patch_img(img, 16, 256)))
            transform_key = 'ssp16_%s' % norm_type
        elif patch_size == 'laanet':
            transform.append(Resize((384,384), interpolation=InterpolationMode.BICUBIC))
            transform_key = 'laanet_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = img_path_table.index[-1]
        for index in tqdm(img_path_table.index, total=len(img_path_table)):
            filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
            for k in transform_dict:
                batch_img[k].append(transform_dict[k](Image.open(filepath).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index==last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)# convert the list of tensors to a single tensor along axis 0

                for model_name in do_models:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()
                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        img_path_table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

        
    return img_path_table




if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir"     , '-i', type=str, help="The path of the directory that contains data", default='./data')
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--batch_size"     , '-b', type=int, help="No. of images in a batch", default=1)
    args = vars(parser.parse_args())

    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')   

    seed =317 #Put any random no.
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) 
    
    table = run_tests(args['data_dir'], args['weights_dir'], args['models'], args['device'])
    if args['fusion']=='None':
        args['fusion']=None
    if args['fusion'] is not None:
        table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    

