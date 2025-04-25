import os
import shutil
import random


source_dir = r'path'
destination_root = r'path'

# Podział zbiorów
train_split = 0.8
val_split = 0.1
test_split = 0.1

image_extensions = ('.jpg', '.jpeg', '.png')

all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]
random.shuffle(all_images)

total = len(all_images)
train_end = int(train_split * total)
val_end = train_end + int(val_split * total)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

def copy_images(image_list, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for image in image_list:
        shutil.copy(os.path.join(source_dir, image), os.path.join(target_folder, image))

copy_images(train_images, os.path.join(destination_root, 'train'))
copy_images(val_images, os.path.join(destination_root, 'val'))
copy_images(test_images, os.path.join(destination_root, 'test'))
