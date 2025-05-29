import torch
import torch.nn as nn
import torch.optim as optim
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
from tqdm.notebook import tqdm 
import time

TRAIN_DIR = '/kaggle/input/train-images/train'
VAL_DIR = '/kaggle/input/val-images/val'
MODEL_SAVE_PATH = '/kaggle/working/best_hybrid_efficientnet_vit_age_model_v2.pt'

EFFICIENTNET_MODEL_NAME = 'efficientnet_b4'
VIT_MODEL_NAME = 'vit_base_patch16_224'

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_EPOCHS_HEAD_ONLY = 5
INITIAL_LEARNING_RATE = 0.0001
FINETUNE_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.01
DROPOUT_RATE_HEAD = 0.4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def parse_age_from_filename(filename):
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return None

class UTKFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.ages = []
        corrupted_files = 0
        print(f"Loading data from: {image_dir}")
        if not os.path.isdir(image_dir):
             print(f"Error: Directory not found - {image_dir}")
             return
        for filename in tqdm(os.listdir(image_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                age = parse_age_from_filename(filename)
                if age is not None:
                    full_path = os.path.join(image_dir, filename)
                    try:
                        with Image.open(full_path) as img:
                           img.verify()
                        self.image_paths.append(full_path)
                        self.ages.append(float(age))
                    except Exception as e:
                        corrupted_files += 1
        if corrupted_files > 0:
            print(f"Skipped {corrupted_files} potentially corrupted files during init.")
        print(f"Found {len(self.image_paths)} valid images.")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            age = torch.tensor(self.ages[idx], dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, age
        except Exception as e:
            print(f"\nError loading/processing image {img_path} in __getitem__: {e}")
            print("Returning placeholder data (image 0).")
            placeholder_path = self.image_paths[0]
            image = Image.open(placeholder_path).convert('RGB')
            age = torch.tensor(self.ages[0], dtype=torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, age


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=resize_interpolation),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    normalize,
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=resize_interpolation),
    transforms.ToTensor(),
    normalize,
])

print("\nCreating Datasets and DataLoaders...")
train_dataset = UTKFaceDataset(image_dir=TRAIN_DIR, transform=train_transforms)
val_dataset = UTKFaceDataset(image_dir=VAL_DIR, transform=val_transforms)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("Error: Training or validation dataset is empty. Please check the paths and data.")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print("DataLoaders created.")


class EfficientNetViTHybrid(nn.Module):
    def __init__(self, efficientnet_model_name, vit_model_name, num_reg_outputs=1, pretrained=True,
                 dropout_rate_head=DROPOUT_RATE_HEAD, freeze_backbones_initially=True):
        super().__init__()
        print(f"Initializing Hybrid model: EffNet='{efficientnet_model_name}', ViT='{vit_model_name}'")
        print(
            f"Pretrained: {pretrained}, Dropout in head: {dropout_rate_head}, Freeze backbones initially: {freeze_backbones_initially}")

        self.efficientnet_backbone = timm.create_model(efficientnet_model_name, pretrained=pretrained, num_classes=0)
        num_efficientnet_features = self.efficientnet_backbone.num_features

        self.vit_backbone = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        num_vit_features = self.vit_backbone.num_features

        if freeze_backbones_initially:
            for param in self.efficientnet_backbone.parameters():
                param.requires_grad = False
            for param in self.vit_backbone.parameters():
                param.requires_grad = False
            print("Backbones are initially frozen.")

        # HEAD
        self.head_fc1 = nn.Linear(num_efficientnet_features + num_vit_features, 1024) 
        self.head_act1 = nn.GELU() 
        self.head_dropout1 = nn.Dropout(dropout_rate_head)
        self.regressor = nn.Linear(1024, num_reg_outputs)


    def forward(self, x):
        features_efficientnet = self.efficientnet_backbone(x)
        features_vit = self.vit_backbone(x)
        combined_features = torch.cat((features_efficientnet, features_vit), dim=1)

        x = self.head_fc1(combined_features)
        x = self.head_act1(x)
        x = self.head_dropout1(x)
        output = self.regressor(x)
        return output

    def unfreeze_backbones(self, unfreeze_effnet_blocks=2, unfreeze_vit_blocks=2):
        """Odmraża ostatnie X bloków w każdym backbone."""
        print(
            f"Unfreezing last {unfreeze_effnet_blocks} blocks of EfficientNet and last {unfreeze_vit_blocks} blocks of ViT.")

        # Odmrażanie backbone'ow
        if hasattr(self.efficientnet_backbone, 'blocks'):
            total_effnet_blocks = len(self.efficientnet_backbone.blocks)
            for i, block in enumerate(self.efficientnet_backbone.blocks):
                if i >= total_effnet_blocks - unfreeze_effnet_blocks:
                    for param in block.parameters():
                        param.requires_grad = True
                    print(f"  Unfroze EfficientNet block {i}")
        else:
            print("  Could not find 'blocks' in EfficientNet backbone, unfreezing all.")
            for param in self.efficientnet_backbone.parameters():
                param.requires_grad = True

        if hasattr(self.vit_backbone, 'blocks'):
            total_vit_blocks = len(self.vit_backbone.blocks)
            for i, block in enumerate(self.vit_backbone.blocks):
                if i >= total_vit_blocks - unfreeze_vit_blocks:
                    for param in block.parameters():
                        param.requires_grad = True
                    print(f"  Unfroze ViT block {i}")
        else:
            print("  Could not find 'blocks' in ViT backbone, unfreezing all.")
            for param in self.vit_backbone.parameters():
                param.requires_grad = True

        for param in self.head_fc1.parameters(): param.requires_grad = True
        for param in self.regressor.parameters(): param.requires_grad = True

print(f"\nCreating Hybrid model: {EFFICIENTNET_MODEL_NAME} + {VIT_MODEL_NAME}")
model = EfficientNetViTHybrid(
    EFFICIENTNET_MODEL_NAME,
    VIT_MODEL_NAME,
    freeze_backbones_initially=True # Trenowanie poczatkowo samego HEAD'A
)
model = model.to(DEVICE)
print("Hybrid model created and moved to device.")

criterion = nn.L1Loss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)


def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS,
                num_epochs_head_only=NUM_EPOCHS_HEAD_ONLY):
    since = time.time()
    best_mae = float('inf')
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    backbones_unfrozen = False

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Odmrażanie backbone'ów 
        if epoch == num_epochs_head_only and not backbones_unfrozen:
            print(f"\n--- Epoch {epoch + 1}: Unfreezing backbone layers and adjusting optimizer ---")
            model.unfreeze_backbones(unfreeze_effnet_blocks=3, unfreeze_vit_blocks=3) 


            # nowy optymalizator z nowymi parametrami do optymalizacji
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
                                                             verbose=True) 
            print(f"Optimizer updated. New learning rate: {FINETUNE_LEARNING_RATE:.1e}")
            backbones_unfrozen = True

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(); dataloader = train_loader
            else:
                model.eval(); dataloader = val_loader
            running_loss = 0.0;
            running_mae = 0.0
            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch + 1}", leave=False)
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE);
                labels = labels.to(DEVICE).view(-1, 1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if outputs.shape != labels.shape: print(
                        f"Shape mismatch! O:{outputs.shape}, L:{labels.shape}"); continue
                    loss = criterion(outputs, labels);
                    mae = torch.abs(outputs - labels).mean()
                    if phase == 'train': loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0);
                running_mae += mae.item() * inputs.size(0)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae.item():.4f}")

            if len(dataloader.dataset) == 0:
                epoch_loss, epoch_mae = 0, 0
            else:
                epoch_loss = running_loss / len(dataloader.dataset); epoch_mae = running_mae / len(dataloader.dataset)
            print(f'{phase.capitalize():<5} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss);
                history['train_mae'].append(epoch_mae)
            else:
                history['val_loss'].append(epoch_loss);
                history['val_mae'].append(epoch_mae)
                scheduler.step(epoch_mae) 
                if epoch_mae < best_mae:
                    best_mae = epoch_mae
                    print(f"*** New best validation MAE: {best_mae:.4f}. Saving model to {MODEL_SAVE_PATH} ***")
                    try:
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    except Exception as e:
                        print(f"Error saving model: {e}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s');
    print(f'Best validation MAE: {best_mae:4f}')
    print(f"Loading best model weights from {MODEL_SAVE_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    except Exception as e:
        print(f"Error loading model weights: {e}"); print("Continuing with last state.")
    return model, history

print("\nStarting training...")
model_ft, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, num_epochs_head_only=NUM_EPOCHS_HEAD_ONLY)
print("Training finished.")

def plot_history(history):
    num_epochs = len(history['val_loss'])
    epochs = list(range(1, num_epochs + 1))
    ticks = list(range(1, num_epochs + 1, 2))
    if epochs[-1] not in ticks:
        ticks.append(epochs[-1])  

    # Wykres MAE jako metryka
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_mae'], label='MAE treningowy')
    plt.plot(epochs, history['val_mae'], label='MAE walidacyjny')
    plt.title('Średni błąd bezwzględny (MAE) przy wykorzystaniu hybrydy Vision Transformer i EfficientNet')
    plt.xlabel('Epoka')
    plt.ylabel('MAE')
    plt.xticks(ticks)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/kaggle/working/mae_uczacy_walidacyjny.png')
    print("Zapisano wykres: mae_uczacy_walidacyjny.png")
    plt.close()

print("\nPlotting training history...")
plot_history(history)
print("Script finished.")