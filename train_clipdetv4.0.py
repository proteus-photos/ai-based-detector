import torch
import random
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import open_clip
from PIL import Image
from torchvision.transforms import Resize, Compose, InterpolationMode
from utils.processing import prepare_data, RandomSizeCrop, rand_jpeg_compression, set_random_seed

# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SVM on top of other clip encoders')
    parser.add_argument('--data_dir', type=str, default='data/trainsvm', help='path of directory that contains data in two folders i.e. real and ai-gen')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--weight_dir', type=str, default='svm_weights', help='directory to store weights of trained models.')
    parser.add_argument('--alpha', type=float, default=0, help='alpha for interpolation')
    parser.add_argument('--train', action='store_true', help='Used to train the model')
    parser.add_argument('--postprocess', action='store_true', help='Whether to postprocess images or not')
    parser.add_argument('--next_to_last', action='store_true', help='Whether to take features from next to last layer or not.')
    parser.add_argument('--out_csv', type=str, help='Output csv path in eval mode')
    return parser.parse_args()

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
        models_dict[modelname] = model
        transforms_dict[modelname] = Compose(transform + [preprocess])
    return models_dict, transforms_dict

# Extract features for a batch of images
def extract_features(model, images, device):
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        features = model.visual.forward(images)
    return features.cpu().numpy()

# Feature extraction and SVM training
def train_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir):
    for modelname in models_dict.keys():
        all_image_features, all_labels = [], []
        batch, batch_labels = [], []
        last_index = img_path_table.index[-1]
        
        for index in tqdm(img_path_table.index, total=len(img_path_table)):
            filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
            label = img_path_table.loc[index, 'label']
            image = Image.open(filepath)
            transformed_image = transforms_dict[modelname](image).to(device)
            batch.append(transformed_image)
            batch_labels.append(label)

            if len(batch) >= batch_size or index == last_index:
                batch = torch.stack(batch, dim=0)
                features = extract_features(models_dict[modelname], batch, device)
                all_image_features.append(features)
                all_labels.extend(batch_labels)
                batch, batch_labels = [], []
        
        # Train SVM on the extracted features
        all_image_features = np.vstack(all_image_features)
        all_labels = np.array(all_labels)
        X_train = StandardScaler().fit_transform(all_image_features)
        y_train = all_labels
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        
        # Save model and scaler
        save_svm_model(clf, output_dir, modelname)

def save_svm_model(clf, output_dir, modelname):
    os.makedirs(output_dir, exist_ok=True)
    svm_model_filename = os.path.join(output_dir, f"svm_{modelname}_10k.pkl")
    scaler_filename = os.path.join(output_dir, f"scaler_{modelname}_10k.pkl")
    joblib.dump(clf, svm_model_filename)
    print(f"SVM for {modelname} saved to {svm_model_filename}")

def interpolate_svm(modelname,output_dir,alpha=0):
    clf1 = joblib.load(os.path.join(output_dir, f"svm_{modelname}_10k.pkl"))
    scaler1 = joblib.load(os.path.join(output_dir, f"scaler_{modelname}_10k.pkl")) 
    clf2 = joblib.load(os.path.join(output_dir, f"svm_{modelname}_10kplus.pkl"))
    scaler2 = joblib.load(os.path.join(output_dir, f"scaler_{modelname}_10kplus.pkl")) 
    # clf_interpolated = svm.SVC(kernel='linear')
    clf_interpolated = svm.LinearSVC()
    clf_interpolated.classes_ = clf1.classes_
    clf_interpolated.coef_ = (1 - alpha) * clf1.coef_ + alpha * clf2.coef_
    clf_interpolated.intercept_ = (1 - alpha) * clf1.intercept_ + alpha * clf2.intercept_
    #interpolate scaler
    mean_interpolated = (1 - alpha) * scaler1.mean_ + alpha * scaler2.mean_
    var_interpolated = (1 - alpha) * scaler1.var_ + alpha * scaler2.var_ + alpha*(1-alpha)*((scaler1.mean_- scaler2.mean_)**2)
    # Create a new StandardScaler and set its mean and variance
    scaler_interpolated = StandardScaler()
    scaler_interpolated.mean_ = mean_interpolated
    scaler_interpolated.scale_ = np.sqrt(var_interpolated)
    scaler_interpolated.var_ = var_interpolated
    print(type(clf_interpolated))
    return clf_interpolated, scaler_interpolated

# Load SVM model and perform evaluation
def evaluate_svm(models_dict, img_path_table, transforms_dict, data_dir, batch_size, device, output_dir, final_table,alpha=0):
    for modelname in models_dict.keys():
        # svm_model_filename = os.path.join(output_dir, f"svm_{modelname}_10k.pkl")
        # scaler_filename = os.path.join(output_dir, f"scaler_{modelname}_10k.pkl")
        # if os.path.exists(svm_model_filename) and os.path.exists(scaler_filename):
        #     clf = joblib.load(svm_model_filename)
        #     scaler = joblib.load(scaler_filename)
        clf, scaler = interpolate_svm(modelname, output_dir, alpha)
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
                features = extract_features(models_dict[modelname], batch, device)
                all_image_features.append(features)
                all_ids.extend(batch_id)
                batch, batch_id = [], []

        all_image_features = np.vstack(all_image_features)
        X_test = scaler.transform(all_image_features)
        y_pred = clf.predict(X_test)
        modelname_column = 'svm_' + modelname + '_10k'
        for ii, logit in zip(all_ids, y_pred):
            final_table.loc[ii, modelname_column] = logit
        # else:
        #     print(f"Trained SVM or scaler not found for {modelname}. Train the model first.")


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
        train_svm(models_dict, img_path_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir)
    else:
        final_table = data_df[['path']]
        evaluate_svm(models_dict, img_path_table, transforms_dict, args.data_dir, args.batch_size, device, args.weight_dir, final_table,args.alpha)
        final_table.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
