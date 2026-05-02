import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class ToadDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.files = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            return self.transform(image)
        return image


def get_data_splits(data_dir, val_split=0.2, seed=42):
    """
    Splits data by folder (Toad ID) to ensure validation is done on unseen individuals.
    """
    all_folders = sorted(
        [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    )

    train_folders, val_folders = train_test_split(
        all_folders, test_size=val_split, random_state=seed
    )

    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def collect_files(folders):
        files = []
        for folder in folders:
            folder_path = os.path.join(data_dir, folder)
            for f in os.listdir(folder_path):
                if f.lower().endswith(img_extensions):
                    files.append(os.path.join(folder_path, f))
        return files

    train_files = collect_files(train_folders)
    val_files = collect_files(val_folders)

    print(
        f"IDs Split: {len(train_folders)} Train IDs | {len(val_folders)} Validation IDs"
    )
    print(
        f"Files Split: {len(train_files)} Train Files | {len(val_files)} Validation Files"
    )

    return train_files, val_files
