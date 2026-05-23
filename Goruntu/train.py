import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights, efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# For confusion matrix and metrics without requiring scikit-learn initially,
# but we can write a simple manual implementation or use scikit-learn if it gets installed.
# We'll write robust custom metrics calculation to ensure it works in any environment,
# and fall back to matplotlib for plotting.

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize CM for annotations
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle div by zero

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Dental Classification Model")
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34", "efficientnet"], help="Model backbone")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to processed dataset folder")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="File path to save the best model weights")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    # Check directories
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: {args.data_dir} directory is missing. Please run prepare_dataset.py first.")
        return

    # Image transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Detected {num_classes} classes: {class_names}")

    # Load Model
    if args.model == "resnet34":
        print("Initializing ResNet34...")
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        # Modify the fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print("Initializing EfficientNet-B0...")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Modify the classifier head
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)

    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed Precision Scaler for faster training and lower VRAM usage
    scaler = torch.cuda.amp.GradScaler()

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    print("\nStarting Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass & Optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update tqdm progress bar
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # Save Checkpoint if accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': class_names,
                'accuracy': best_val_acc
            }, args.save_path)
            print(f"  => Best checkpoint saved to {args.save_path} (Val Acc: {best_val_acc:.2f}%)")

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save Class names configuration for Flask app
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))

    # Plot Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig("learning_curves.png", dpi=150)
    plt.close()
    print("Learning curves plot saved as learning_curves.png")

    # Generate Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    print("Confusion matrix plot saved as confusion_matrix.png")

    # Print a classification report (Precision, Recall, F1)
    print("\nClassification Metrics Per Class:")
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 62)
    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:<22} | {precision:<10.2%} | {recall:<10.2%} | {f1:<10.2f}")

if __name__ == "__main__":
    main()
