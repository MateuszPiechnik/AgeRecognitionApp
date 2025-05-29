import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
try:
    from torchvision.transforms.functional import InterpolationMode
    resize_interpolation = InterpolationMode.BICUBIC 
except ImportError:
    resize_interpolation = transforms.InterpolationMode.BICUBIC

import timm
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import time

TEST_DIR = '/kaggle/input/test-images/test'

MODEL_PATHS_TO_EVALUATE = {
    'Hybrid_EfficientNet_ViT': '/kaggle/input/efficientnet_vit_model/pytorch/default/1/best_hybrid_efficientnet_vit_age_model.pt',
    'Hybrid_ResNet50_ViT': '/kaggle/input/resnet_vit_hybrid/pytorch/default/1/best_hybrid_resnet_vit_age_model.pt',
    'Standalone_EfficientNetB4': '/kaggle/input/efficientnet_model/pytorch/default/1/best_efficientnet_b4_age_model.pt',
    'Standalone_ResNet50': '/kaggle/input/resnet50_model/pytorch/default/1/best_resnet50_age_model.pt',
    'Standalone_ViT_B16': '/kaggle/input/vit_model/pytorch/default/1/best_vit_b16_age_model.pt',
}

RESULTS_CSV_PATH = '/kaggle/working/final_evaluation_results.csv'

IMG_SIZE = 224
BATCH_SIZE = 16 
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFFICIENTNET_MODEL_NAME = 'efficientnet_b4'
RESNET_MODEL_NAME = 'resnet50'
VIT_MODEL_NAME = 'vit_base_patch16_224'

def parse_age_from_filename(filename):
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return None
    
class UTKFaceDataset(Dataset):
    """Custom Dataset dla UTKFace."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.ages = []

        print(f"Loading data from: {image_dir}")
        for filename in tqdm(os.listdir(image_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                age = parse_age_from_filename(filename)
                if age is not None:
                    self.image_paths.append(os.path.join(image_dir, filename))
                    self.ages.append(float(age))
        print(f"Found {len(self.image_paths)} valid images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\nError loading image {img_path}: {e}")
            placeholder_path = self.image_paths[0]
            image = Image.open(placeholder_path).convert('RGB')
            age = torch.tensor(self.ages[0], dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, age 

        age = torch.tensor(self.ages[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, age
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=resize_interpolation),
    transforms.ToTensor(),
    normalize,
])

class EfficientNetViTHybrid(nn.Module):
    def __init__(self, efficientnet_model_name, vit_model_name, num_reg_outputs=1, pretrained_weights=True, dropout_rate=0.2):
        super().__init__()
        self.efficientnet_backbone = timm.create_model(efficientnet_model_name, pretrained=pretrained_weights, num_classes=0)
        num_efficientnet_features = self.efficientnet_backbone.num_features

        self.vit_backbone = timm.create_model(vit_model_name, pretrained=pretrained_weights, num_classes=0)
        num_vit_features = self.vit_backbone.num_features

        self.fusion_fc1 = nn.Linear(num_efficientnet_features + num_vit_features, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fusion_fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(1024, num_reg_outputs)

    def forward(self, x):
        features_efficientnet = self.efficientnet_backbone(x)
        features_vit = self.vit_backbone(x)
        combined_features = torch.cat((features_efficientnet, features_vit), dim=1)
        x = self.fusion_fc1(combined_features)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fusion_fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        output = self.regressor(x)
        return output
    
class ResNetViTHybrid(nn.Module):
    def __init__(self, resnet_model_name, vit_model_name, num_reg_outputs=1, pretrained=True, dropout_rate=0.2): 
        super().__init__()
        self.resnet_backbone = timm.create_model(resnet_model_name, pretrained=pretrained, num_classes=0) # num_classes=0 usuwa głowę
        num_resnet_features = self.resnet_backbone.num_features

        self.vit_backbone = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        num_vit_features = self.vit_backbone.num_features 

        # na podstawie rysunku z artykulu Naznin & Islam
        self.fusion_fc1 = nn.Linear(num_resnet_features + num_vit_features, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fusion_fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(1024, num_reg_outputs) # Dla regresji wieku

    def forward(self, x):
        features_resnet = self.resnet_backbone(x) 

        features_vit = self.vit_backbone(x) 

        # Konkatenacja cech
        combined_features = torch.cat((features_resnet, features_vit), dim=1)

        x = self.fusion_fc1(combined_features)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fusion_fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        output = self.regressor(x)
        return output
    
def create_standalone_model(base_model_name, pretrained_weights=True, num_reg_outputs=1):
    model = timm.create_model(base_model_name, pretrained=pretrained_weights, num_classes=num_reg_outputs)
    return model

print("\n--- Starting Final Evaluation on Test Set ---")

results_data = [] 

test_dataset = UTKFaceDataset(image_dir=TEST_DIR, transform=test_transforms)
if len(test_dataset) == 0:
    print(f"Error: Test dataset at {TEST_DIR} is empty or could not be loaded. Exiting.")
    exit()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

for model_key, model_path in MODEL_PATHS_TO_EVALUATE.items():
    print(f"\n---> Evaluating model: {model_key} <---")

    # pretrained_weights=False, ponieważ ładujemy nasze wytrenowane wagi, a nie z ImageNet
    if model_key == 'Hybrid_EfficientNet_ViT':
        model = EfficientNetViTHybrid(EFFICIENTNET_MODEL_NAME, VIT_MODEL_NAME, pretrained_weights=False)
    elif model_key == 'Hybrid_ResNet50_ViT':
        model = ResNetViTHybrid(RESNET_MODEL_NAME, VIT_MODEL_NAME, pretrained_weights=False)
    elif model_key == 'Standalone_EfficientNetB4':
        model = create_standalone_model(EFFICIENTNET_MODEL_NAME, pretrained_weights=False)
    elif model_key == 'Standalone_ResNet50':
        model = create_standalone_model(RESNET_MODEL_NAME, pretrained_weights=False)
    elif model_key == 'Standalone_ViT_B16':
        model = create_standalone_model(VIT_MODEL_NAME, pretrained_weights=False)
    else:
        print(f"Warning: Architecture for {model_key} not defined explicitly. Skipping or trying generic load.")
        try:
            model = timm.create_model(model_key.lower().replace('standalone_', ''), pretrained=False, num_classes=1)
        except Exception as e_timm:
            print(f"Could not create model {model_key} with timm: {e_timm}. Skipping.")
            results_data.append({'Model': model_key, 'MAE': float('nan'), 'Error': f"Arch not defined/timm error: {e_timm}"})
            continue

    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path} for {model_key}. Skipping.")
        results_data.append({'Model': model_key, 'MAE': float('nan'), 'Error': f"Weights file missing: {model_path}"})
        continue
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded weights from {model_path}")
    except Exception as e:
        print(f"Error loading weights for {model_key} from {model_path}: {e}")
        results_data.append({'Model': model_key, 'MAE': float('nan'), 'Error': f"Error loading weights: {e}"})
        continue

    model = model.to(DEVICE)
    model.eval() 

    all_outputs_list = []
    all_labels_list = []
    inference_times = []

    with torch.no_grad(): # Wyłączamy obliczanie gradientów
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_key}", leave=False):
            inputs = inputs.to(DEVICE)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            all_outputs_list.extend(outputs.cpu().numpy().flatten())
            all_labels_list.extend(labels.numpy().flatten()) 

    if not all_labels_list:
        print(f"No data was processed for {model_key} from the test set. Skipping MAE calculation.")
        results_data.append({'Model': model_key, 'MAE': float('nan'), 'Error': "No test data processed"})
        continue

    all_outputs_np = np.array(all_outputs_list)
    all_labels_np = np.array(all_labels_list)

    mae = np.mean(np.abs(all_outputs_np - all_labels_np))
    avg_inference_time_batch = np.mean(inference_times) if inference_times else 0
    
    print(f"MAE for {model_key} on test set: {mae:.4f}")
    print(f"Avg. inference time per batch for {model_key}: {avg_inference_time_batch:.4f}s")
    results_data.append({'Model': model_key, 'MAE': mae, 'Avg_Inference_Time_Batch': avg_inference_time_batch, 'Error': None})

print("\n--- Final Evaluation Results (Test Set) ---")
results_df = pd.DataFrame(results_data)
results_df = results_df.set_index('Model') 

if 'MAE' in results_df.columns:
    results_df = results_df.sort_values(by='MAE', ascending=True)

print(results_df)

try:
    results_df.to_csv(RESULTS_CSV_PATH)
    print(f"\nSuccessfully saved final results to {RESULTS_CSV_PATH}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

print("\n--- Evaluation Script Finished ---")