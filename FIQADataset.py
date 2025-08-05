import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

class FIQADataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None, is_train=True, train_ratio=0.8, seed=42):
        """
        Args:
            data_dir (str): Root directory for image data
            csv_path (str): Path to train.csv file
            transform: Image transformation function
            is_train (bool): Whether this is training set
            train_ratio (float): Ratio of training data to total data
            seed (int): Random seed for dataset splitting
        """
        # Read CSV file, column 0 is filename, column 1 is score
        self.df = pd.read_csv(csv_path, header=None, names=['image_name', 'score'])
        
        # Split into training and test sets
        train_indices, test_indices = train_test_split(
            np.arange(len(self.df)),
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        
        # Select appropriate dataset based on is_train
        self.indices = train_indices if is_train else test_indices
        
        # Save necessary parameters
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
    def __getitem__(self, idx):
        # Get actual index
        real_idx = self.indices[idx]
        
        # Get image path and label
        row = self.df.iloc[real_idx]
        image_name = str(row['image_name'])
        # Add .png extension if filename doesn't have an extension
        if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_name += '.png'
        image_path = os.path.join(self.data_dir, image_name)
        
        try:
            # Read image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform
            if self.transform is not None:
                image = self.transform(image)
                
            # Get label (score) and convert to float32
            label = torch.tensor(float(row['score']), dtype=torch.float32)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # If error occurs, return first image in dataset (rarely happens)
            return self.__getitem__(0) if idx != 0 else None

    def __len__(self):
        return len(self.indices)

class FIQADatasetWithGFIQA(Dataset):
    def __init__(self, data_dir, gfiqa_data_dir, csv_path, gfiqa_csv_path, transform=None, is_train=True, train_ratio=0.8, seed=42):
        """
        Args:
            data_dir (str): Root directory for original image data
            gfiqa_data_dir (str): Root directory for GFIQA dataset
            csv_path (str): Path to original training data CSV file
            gfiqa_csv_path (str): Path to GFIQA dataset CSV file
            transform: Image transformation function
            is_train (bool): Whether this is training set
            train_ratio (float): Ratio of training data to total data
            seed (int): Random seed for dataset splitting
        """
        # Read original training data CSV file
        self.df_original = pd.read_csv(csv_path, header=None, names=['image_name', 'score'])
        self.df_original['source'] = 'original'  # Add source marker
        
        # Read GFIQA data CSV file
        self.df_gfiqa = pd.read_csv(gfiqa_csv_path)
        self.df_gfiqa = self.df_gfiqa.rename(columns={'Image': 'image_name', 'mapped_score': 'score'})
        self.df_gfiqa = self.df_gfiqa[['image_name', 'score']]  # Keep only needed columns
        self.df_gfiqa['source'] = 'gfiqa'  # Add source marker
        
        # Split original training data into train and validation sets
        train_indices, val_indices = train_test_split(
            np.arange(len(self.df_original)),
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        
        # Select appropriate data based on is_train parameter
        if is_train:
            # Training mode: use training portion of original data plus GFIQA data
            self.df = pd.concat([
                self.df_original.iloc[train_indices],
                self.df_gfiqa
            ], ignore_index=True)
        else:
            # Validation mode: use only validation portion of original data
            self.df = self.df_original.iloc[val_indices]
        
        # Save necessary parameters
        self.data_dir = data_dir
        self.gfiqa_data_dir = gfiqa_data_dir
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # Get image path and label
            row = self.df.iloc[idx]
            image_name = str(row['image_name'])
            # Ensure image name has correct extension
            if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                image_name += '.png'
            # Choose correct directory based on data source
            data_source = row['source']
            image_path = os.path.join(
                self.gfiqa_data_dir if data_source == 'gfiqa' else self.data_dir,
                image_name
            )
            
            # Read image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform
            if self.transform is not None:
                image = self.transform(image)
                
            # Get label (score) and convert to float32
            label = torch.tensor(float(row['score']), dtype=torch.float32)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_name} from {data_source} dataset: {str(e)}")
            # If error occurs, return first image in dataset (rarely happens)
            return self.__getitem__(0) if idx != 0 else None
