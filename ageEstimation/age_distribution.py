import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13

base_path = r'path'
subfolders = ['train', 'val', 'test']

ages = []

age_pattern = re.compile(r'^(\d+)_')

for folder in subfolders:
    folder_path = os.path.join(base_path, folder)
    for filename in os.listdir(folder_path):
        match = age_pattern.match(filename)
        if match:
            age = int(match.group(1))
            ages.append(age)

plt.figure(figsize=(14, 6))
sns.histplot(ages, bins=range(0, 117), kde=False, color='cornflowerblue', edgecolor='black')

plt.title("Rozkład wieku w zbiorze UTKFace", pad=15)
plt.xlabel("Wiek")
plt.ylabel("Liczba zdjęć")
plt.xticks(range(0, 117, 5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()