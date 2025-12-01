import os
import random
import zipfile
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

dataset_slug = "razaimam45/spacenet-an-optimally-distributed-astronomy-data"
sampled_path = "data/sampled"
num_samples_per_class = 10

# Authenticate
api = KaggleApi()
api.authenticate()

os.makedirs("data", exist_ok=True)
os.makedirs(sampled_path, exist_ok=True)

# Download dataset zip (compressed)
print("Downloading dataset zip (compressed)...")
api.dataset_download_files(dataset_slug, path="data", unzip=False, quiet=False)
print("Download complete!")

# Find the downloaded zip file
zip_files = glob.glob("data/*.zip")
if not zip_files:
    raise FileNotFoundError("No zip file found in data/")
zip_path = zip_files[0]

# Extract only a sample of images
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    all_files = zip_ref.namelist()
    classes = sorted({f.split("/")[0] for f in all_files if "/" in f})
    for cls in classes:
        cls_files = [f for f in all_files if f.startswith(cls + "/") and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        sampled_files = random.sample(cls_files, min(num_samples_per_class, len(cls_files)))

        cls_dir = os.path.join(sampled_path, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for f in sampled_files:
            zip_ref.extract(f, sampled_path)
            os.rename(os.path.join(sampled_path, f), os.path.join(cls_dir, os.path.basename(f)))

print(f"Sampled {num_samples_per_class} images per class in '{sampled_path}'")
