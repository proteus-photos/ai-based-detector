import torch
import random
import os
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib 
import open_clip
from PIL import Image
from torchvision.transforms  import Resize, Compose, InterpolationMode, Lambda
from utils.processing import prepare_data, RandomSizeCrop, rand_jpeg_compression

is_train = False

seed =317 #Put any random no.
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed) 

# Directories and configurations
data_dir = 'data/evaldata'
batch_size = 512
next_to_last = True
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = 'svm_weights'  # Directory to save trained SVMs
post_process =True
os.makedirs(output_dir, exist_ok=True)

# Load data and initialize models
data_df = prepare_data(data_dir)
data_df['label'] = np.where(data_df['type'] == 'real', -1, 1)
img_path_table = data_df[['path', 'label']]  
model_list = [
    ('ViT-SO400M-14-SigLIP', 'webli'),
    ('ViT-SO400M-14-SigLIP-384', 'webli'),
    ('ViT-H-14-quickgelu', 'dfn5b'),
    ('ViT-H-14-378-quickgelu', 'dfn5b'),
]

# Initialize dictionaries to store models and preprocessing transforms
models_dict = dict()
transforms_dict = dict()
# Prepare models and transforms
for modelname, dataset in model_list:
    transform = list()
    if post_process:
        transform.append(RandomSizeCrop(min_scale=0.625, max_scale=1.0))
        transform.append(Resize((200,200), interpolation=InterpolationMode.BICUBIC))
        transform.append(rand_jpeg_compression)
    model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=dataset)
    model.eval()
    model.to(device)
    models_dict[modelname] = model
    transform.append(preprocess)
    transform =Compose(transform)
    transforms_dict[modelname] = transform
    if next_to_last == True:
        model.visual.proj = None

# Function to extract features for a batch of images
def extract_features(model, images):
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        features = model.visual.forward(images) 
    return features.cpu().numpy()

if not is_train:
    final_table = data_df[['path']]

# Feature extraction and SVM training
for modelname in models_dict.keys():

    if is_train:
        all_image_features = []
        all_labels = []
        
        batch = []
        batch_labels = []
        last_index = img_path_table.index[-1]
        
        # Process images in batches
        for index in tqdm(img_path_table.index, total=len(img_path_table)):
            filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
            label = img_path_table.loc[index, 'label']
            image = Image.open(filepath)
            transformed_image = transforms_dict[modelname](image).to(device)
            
            batch.append(transformed_image)
            batch_labels.append(label)
            
            if len(batch) >= batch_size or index == last_index:
                # Stack and extract features for the batch
                batch = torch.stack(batch, dim=0)
                features = extract_features(models_dict[modelname], batch)
                
                # Append features and labels to full dataset
                all_image_features.append(features)
                all_labels.extend(batch_labels)
                
                # Reset batch
                batch = []
                batch_labels = []
        
        # Convert the list of arrays into a single NumPy array
        all_image_features = np.vstack(all_image_features)
        all_labels = np.array(all_labels)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(all_image_features)
        y_train = all_labels
        
        # Train the SVM model
        clf = svm.SVC(kernel='linear')  # Use a linear kernel or 'rbf' if you want more complexity
        clf.fit(X_train, y_train)
        

        # Save the trained SVM and the scaler for later use
        svm_model_filename = os.path.join(output_dir, f"svm_{modelname}_10k.pkl")
        scaler_filename = os.path.join(output_dir, f"scaler_{modelname}_10k.pkl")
        
        joblib.dump(clf, svm_model_filename)
        joblib.dump(scaler, scaler_filename)

        print(f"SVM for {modelname} saved to {svm_model_filename}")
    else:
        # Load the trained SVM model and scaler
        svm_model_filename = os.path.join(output_dir, f"svm_{modelname}_10k.pkl")
        scaler_filename = os.path.join(output_dir, f"scaler_{modelname}_10k.pkl")

        if os.path.exists(svm_model_filename) and os.path.exists(scaler_filename):
            clf = joblib.load(svm_model_filename)
            scaler = joblib.load(scaler_filename)
            print(f"Loaded SVM model and scaler for {modelname}.")

            all_image_features = []
            all_ids = []

            batch = []
            batch_id = []

            last_index = img_path_table.index[-1]

            # Process images in batches for evaluation
            for index in tqdm(img_path_table.index, total=len(img_path_table)):
                filepath = os.path.join(data_dir, img_path_table.loc[index, 'path'])
                label = img_path_table.loc[index, 'label']
                image = Image.open(filepath)
                transformed_image = transforms_dict[modelname](image).to(device)

                batch.append(transformed_image)
                batch_id.append(index)


                if len(batch) >= batch_size or index == last_index:
                    # Stack and extract features for the batch
                    batch = torch.stack(batch, dim=0)
                    features = extract_features(models_dict[modelname], batch)

                    # Append features and labels to full dataset
                    all_image_features.append(features)
                    all_ids.extend(batch_id)

                    # Reset batch
                    batch = []
                    batch_id = []

            # Convert the list of arrays into a single NumPy array
            all_image_features = np.vstack(all_image_features)
            all_ids = np.array(all_ids)

            # Standardize features using the loaded scaler
            X_test = scaler.transform(all_image_features)

            y_pred = clf.predict(X_test)
            modelname = 'svm_'+modelname+'_10k'
            for ii, logit in zip(all_ids, y_pred):
                final_table.loc[ii, modelname] = logit

        else:
            print(f"Trained SVM or scaler not found for {modelname}. Train the model first.")

if not is_train:
    final_table.to_csv('csvs_post/clipdetv4.0_10k_post.csv', index=False)
