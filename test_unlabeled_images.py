import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
from scipy.optimize import curve_fit

import models.FIQA_model as FIQA_model
from config_SwinB import TrainingConfig
from utils import logistic_func
from utils import performance_fit

def create_test_transform(image_size, image_crop):
    """Create test image transform"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_quality(model, image_path, transform, device):
    """Predict quality score for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image)
        
        return output.item()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def fit_logistic_mapping(y_label, y_output):
    """Fit logistic mapping between predicted scores and ground truth scores"""
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    return popt

def apply_logistic_mapping(scores, mapping_params):
    """Apply logistic mapping to scores using fitted parameters"""
    return logistic_func(scores, *mapping_params)

def main():
    # Initialize configuration
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and load model
    if config.model_name == 'FIQA_Swin_B':
        model = FIQA_model.FIQA_Swin_B(pretrained_path=config.pretrained_path, is_pretrained=False)
    elif config.model_name == 'FIQA_EdgeNeXt_XXS':
        model = FIQA_model.FIQA_EdgeNeXt_XXS(is_pretrained=False)

    checkpoint_path = os.path.join(config.model_save_dir, 'best_model.pth')
    

    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    # Create image transform
    transform = create_test_transform(config.image_size, config.image_crop)
    
    # Step 1: Process data.csv images to get quality scores
    df_data = pd.read_csv('data_file/train.csv', header=None, names=['image_name', 'score'])
    data_scores = []
    data_gt_scores = []
    print("Processing train.csv images...")
    for _, row in tqdm(df_data.iterrows()):
        image_name = str(row['image_name'])
        # Add .png extension if no extension present
        if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_name += '.png'
        img_path = os.path.join(config.data_dir, image_name)
        score = predict_quality(model, img_path, transform, device)
        if score is not None:
            data_scores.append(score)
            data_gt_scores.append(float(row['score']))
    
    # Step 2: Fit logistic mapping
    print("Fitting logistic mapping...")
    mapping_params = fit_logistic_mapping(np.array(data_gt_scores), np.array(data_scores))
    plcc, srcc, krcc, rmse = performance_fit(np.array(data_gt_scores), np.array(data_scores))
    print(f"SRCC={srcc:.4f}, KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")
    
    # Step 3: Process GFIQA images
    gfiqa_dir = '/data/sunwei_data/GFIQA/image'
    gfiqa_images = []
    for img_path in glob.glob(os.path.join(gfiqa_dir, '*')):  # Adjust pattern if needed
        gfiqa_images.append(img_path)
    
    # Predict and map scores
    results = []
    # print("Processing GFIQA images...")
    print("Processing face images...")
    for image_path in tqdm(gfiqa_images):
        name = os.path.basename(image_path)
        score = predict_quality(model, image_path, transform, device)
        
        if score is not None:
            # Apply logistic mapping
            mapped_score = logistic_func(score, *mapping_params)
            results.append({
                'Image': name,
                'raw_score': score,
                'mapped_score': mapped_score
            })
    
    # Save results
    df_results = pd.DataFrame(results)
    results_path = 'data_file/gfiqa_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Total images processed: {len(results)}")
    print(f"Raw scores - Mean: {df_results['raw_score'].mean():.4f}, Std: {df_results['raw_score'].std():.4f}")
    print(f"Mapped scores - Mean: {df_results['mapped_score'].mean():.4f}, Std: {df_results['mapped_score'].std():.4f}")

if __name__ == '__main__':
    main()
