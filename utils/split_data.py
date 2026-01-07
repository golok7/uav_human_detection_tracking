import os
import random
import shutil

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

def create_dirs():
    for split in SPLITS:
        for cls in ["0", "1"]:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

def split_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_data.items():
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(PROCESSED_DIR, split, class_name, file)
            shutil.copy(src, dst)

def main():
    create_dirs()
    for cls in ["0", "1"]:
        split_class(cls)
    print("✅ Data split completed successfully.")

if __name__ == "__main__":
    main()
