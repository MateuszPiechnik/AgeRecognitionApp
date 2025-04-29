import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import timm # Biblioteka z modelami
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
import copy

TRAIN_DIR = '/kaggle/input/train-images/train'
VAL_DIR = '/kaggle/input/val-images/val'
MODEL_SAVE_PATH = '/kaggle/working/best_efficientnet_b4_age_model.pt'

# Parametry modelu i treningu
MODEL_NAME = 'efficientnet_b4'
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_age_from_filename(filename):
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse age from {filename}")
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC), # Użyj BICUBIC dla EfficientNet
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    normalize,
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

train_dataset = UTKFaceDataset(image_dir=TRAIN_DIR, transform=train_transforms)
val_dataset = UTKFaceDataset(image_dir=VAL_DIR, transform=val_transforms)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("Error: One or both datasets are empty. Check image paths and parsing.")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print("DataLoaders created.")

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes = 1)

# Zamień ostatnią warstwę klasyfikacyjną na warstwę regresyjną (1 wyjście - wiek)
num_ftrs = model.get_classifier().in_features
model.reset_classifier(1)

model = model.to(DEVICE)

criterion = nn.L1Loss() # MAE Loss - odpowiednik dla regresji MAE
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf') # Chcemy minimalizować MAE

    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Ustaw model w tryb treningowy
                dataloader = train_loader
            else:
                model.eval()   # Ustaw model w tryb ewaluacyjny
                dataloader = val_loader

            running_loss = 0.0
            running_mae = 0.0

            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()

                # Śledź historię tylko podczas treningu
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    mae = torch.abs(outputs - labels).mean()

                    # Backward + optymalizacja tylko w fazie treningowej
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)
                progress_bar.set_postfix(loss=loss.item(), mae=mae.item())


            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_mae = running_mae / len(dataloader.dataset)


            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_mae'].append(epoch_mae)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_mae'].append(epoch_mae)

                if epoch_mae < best_mae:
                    best_mae = epoch_mae
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"*** New best validation MAE: {best_mae:.4f}. Saving model... ***")
                    try:
                        torch.save(best_model_wts, MODEL_SAVE_PATH)
                    except Exception as e:
                         print(f"Error saving model: {e}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val MAE: {best_mae:4f}')

    try:
        model.load_state_dict(best_model_wts)
    except Exception as e:
         print(f"Error loading best model weights: {e}")

    return model, history

# --- Uruchomienie Treningu ---
print("\nStarting training...")
model_ft, history = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
print("Training finished.")

def plot_history(history):
    num_epochs = len(history['val_loss'])
    epochs = list(range(1, num_epochs + 1))
    ticks = list(range(1, num_epochs + 1, 2))
    if epochs[-1] not in ticks:
        ticks.append(epochs[-1])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Błąd treningowy (L1Loss)')
    plt.plot(epochs, history['val_loss'], label='Błąd walidacyjny (L1Loss)')
    plt.title('Strata ucząca i walidacyjna (MAE Loss)')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.xticks(ticks)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/kaggle/working/strata_uczaca_walidacyjna.png')
    print("Zapisano wykres: strata_uczaca_walidacyjna.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_mae'], label='MAE treningowy')
    plt.plot(epochs, history['val_mae'], label='MAE walidacyjny')
    plt.title('Średni błąd bezwzględny (MAE) przy wykorzystaniu EfficientNet')
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