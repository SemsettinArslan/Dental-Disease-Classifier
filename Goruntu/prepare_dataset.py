import os
import shutil
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Define source directories for each category
source_dirs = {
    "Calculus": r"data\Calculus\Calculus",
    "Caries": r"data\Data caries\Data caries\caries augmented data set",
    "Gingivitis": r"data\Gingivitis\Gingivitis",
    "Hypodontia": r"data\hypodontia\hypodontia",
    "Mouth Ulcer": r"data\Mouth Ulcer\Mouth Ulcer\Mouth_Ulcer_augmented_DataSet",
    "Tooth Discoloration": r"data\Tooth Discoloration\Tooth Discoloration\Tooth_discoloration_augmented_dataser"
}

# Output directories
base_out_dir = "dataset"

# Clean existing dataset folder if it exists to ensure a clean split
if os.path.exists(base_out_dir):
    print(f"Cleaning existing directory: {base_out_dir}...")
    shutil.rmtree(base_out_dir, ignore_errors=True)

train_dir = os.path.join(base_out_dir, "train")
val_dir = os.path.join(base_out_dir, "val")
test_dir = os.path.join(base_out_dir, "test")

# Create output directories
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

print("Starting dataset splitting (70% Train, 15% Val, 15% Test) and copying...")

for class_name, src_rel_path in source_dirs.items():
    # Make path absolute relative to workspace root (current working directory)
    src_path = os.path.abspath(src_rel_path)
    
    if not os.path.exists(src_path):
        print(f"Warning: Source path {src_path} does not exist. Skipping category {class_name}.")
        continue
        
    # Gather all image files
    images = []
    for root, _, files in os.walk(src_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                images.append(os.path.join(root, file))
                
    # Shuffle and split
    random.shuffle(images)
    num_val = int(len(images) * 0.15)
    num_test = int(len(images) * 0.15)
    
    val_images = images[:num_val]
    test_images = images[num_val:num_val+num_test]
    train_images = images[num_val+num_test:]
    
    # Create category directories under train, val, and test
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)
    
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)
    os.makedirs(class_test_dir, exist_ok=True)
    
    print(f"\nProcessing {class_name}:")
    print(f"  Total images found: {len(images)}")
    print(f"  Copying {len(train_images)} images to training set...")
    for img_path in tqdm(train_images, desc=f"Train {class_name}"):
        filename = os.path.basename(img_path)
        dest = os.path.join(class_train_dir, filename)
        if os.path.exists(dest):
            filename = f"{os.path.basename(os.path.dirname(img_path))}_{filename}"
            dest = os.path.join(class_train_dir, filename)
        shutil.copy2(img_path, dest)
        
    print(f"  Copying {len(val_images)} images to validation set...")
    for img_path in tqdm(val_images, desc=f"Val {class_name}"):
        filename = os.path.basename(img_path)
        dest = os.path.join(class_val_dir, filename)
        if os.path.exists(dest):
            filename = f"{os.path.basename(os.path.dirname(img_path))}_{filename}"
            dest = os.path.join(class_val_dir, filename)
        shutil.copy2(img_path, dest)

    print(f"  Copying {len(test_images)} images to test set...")
    for img_path in tqdm(test_images, desc=f"Test {class_name}"):
        filename = os.path.basename(img_path)
        dest = os.path.join(class_test_dir, filename)
        if os.path.exists(dest):
            filename = f"{os.path.basename(os.path.dirname(img_path))}_{filename}"
            dest = os.path.join(class_test_dir, filename)
        shutil.copy2(img_path, dest)

print("\nDataset preparation completed successfully!")
