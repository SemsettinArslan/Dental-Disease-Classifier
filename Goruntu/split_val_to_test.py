import os
import shutil
import random

def main():
    random.seed(42)
    val_dir = os.path.join("dataset", "val")
    test_dir = os.path.join("dataset", "test")

    if not os.path.exists(val_dir):
        print(f"Error: {val_dir} does not exist. Run prepare_dataset.py first.")
        return

    # Check if test folder already exists to avoid splitting again
    if os.path.exists(test_dir):
        print(f"Warning: {test_dir} already exists. To split again, delete it first.")
        return

    print("Splitting validation set 50/50 into validation and test sets...")
    
    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    for class_name in classes:
        class_val_path = os.path.join(val_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)
        os.makedirs(class_test_path, exist_ok=True)
        
        files = [f for f in os.listdir(class_val_path) if os.path.isfile(os.path.join(class_val_path, f))]
        random.shuffle(files)
        
        # Take 50% of files to move to test directory
        split_idx = len(files) // 2
        test_files = files[:split_idx]
        
        print(f"Class '{class_name}': Moving {len(test_files)} files out of {len(files)} to test set...")
        for file in test_files:
            shutil.move(
                os.path.join(class_val_path, file),
                os.path.join(class_test_path, file)
            )
            
    print("\nDataset split complete!")
    print(f"Validation directory: {val_dir}")
    print(f"Test directory (for school demo): {test_dir}")

if __name__ == "__main__":
    main()
