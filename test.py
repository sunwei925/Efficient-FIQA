import os
import torch
from PIL import Image
from torchvision import transforms
import argparse
import models.FIQA_model as FIQA_model

def create_test_transform(image_size):
    """Create test image transformations"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_quality(model, image_path, transform, device):
    """Predict quality score for a single image"""
    try:
        # Read and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)  # Move image to device
        
        # Pass image through model
        with torch.no_grad():
            output = model(image)
        
        return output.item()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def setup_device(gpu_ids):
    """Setup and return compute device"""
    if gpu_ids.lower() == 'cpu':
        return torch.device('cpu')
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead")
        return torch.device('cpu')
    
    # Setup GPU
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
        print(f"Using GPU {gpu_ids}")
        return device
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")
        print("Falling back to CPU")
        return torch.device('cpu')

def main(config):
    # Setup device
    device = setup_device(config.gpu_ids)
    
    # Create model
    if config.model_name == 'FIQA_EdgeNeXt_XXS':
        model = FIQA_model.FIQA_EdgeNeXt_XXS(is_pretrained=False)
    elif config.model_name == 'FIQA_Swin_B':
        model = FIQA_model.FIQA_Swin_B(pretrained_path=None)
    print("Using model: ", config.model_name)
    
    # Load trained weights
    try:
        # Load model weights based on device type
        if device.type == 'cpu':
            state_dict = torch.load(os.path.join(config.model_weights_file), map_location='cpu')
        else:
            state_dict = torch.load(os.path.join(config.model_weights_file))
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully and moved to {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Create image transformations
    transform = create_test_transform(config.image_size)
    
    # Predict quality score
    score = predict_quality(model, config.image_file, transform, device)
    
    if score is not None:
        print(f"The quality score of the image {config.image_file} is {score:.4f}")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Face Image Quality Assessment")
    parser.add_argument('--model_name', type=str, default='FIQA_EdgeNeXt_XXS',
                      help='Model architecture to use (FIQA_EdgeNeXt_XXS or FIQA_Swin_B)')
    parser.add_argument('--model_weights_file', type=str, default='ckpts/checkpoint.pt',
                      help='Path to model weights file')
    parser.add_argument('--image_size', type=int, default=356,
                      help='Input image size')
    parser.add_argument('--image_file', type=str, default='demo_images/z06399.png',
                      help='Path to input image file')
    parser.add_argument('--gpu_ids', type=str, default='0',
                      help='GPU IDs to use (e.g., "0", "0,1", or "cpu" for CPU only)')

    config = parser.parse_args()
    return config


if __name__ == '__main__':
    config = parse_args()
    main(config)