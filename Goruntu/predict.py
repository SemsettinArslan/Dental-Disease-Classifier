import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Predict Dental Disease Class for an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to trained model weights")
    args = parser.parse_args()

    # Verify paths
    if not os.path.exists(args.image):
        print(f"Error: Image path {args.image} does not exist.")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist. Please train the model first.")
        return

    # Load checkpoint to get model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Initialize model
    model = resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Image preprocessing
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and predict
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # Get top prediction
    conf, pred_class_idx = torch.max(probabilities, 0)
    pred_class_name = class_names[pred_class_idx.item()]
    
    print("\n" + "="*40)
    print(f"Prediction Results for: {os.path.basename(args.image)}")
    print("="*40)
    print(f"Result: {pred_class_name} ({conf.item():.2%})")
    print("\nAll Probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  {class_names[idx]:<22} : {prob.item():.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
