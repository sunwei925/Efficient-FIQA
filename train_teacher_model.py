import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config_SwinB import TrainingConfig
from FIQADataset import FIQADataset
import models.FIQA_model as FIQA_model
from utils import performance_fit
from utils import plcc_loss

def setup_logger(save_dir):
    """Set up logging configuration"""
    import logging
    log_file = save_dir / 'train.log'
    
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_transforms(config):
    """Get data augmentation and preprocessing methods"""
    train_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def validate(model, val_loader, criterion, device, epoch, writer, logger):
    """Validate model performance"""
    model.eval()
    val_loss = 0
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))
            
            val_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    predictions = np.array(predictions).squeeze()
    ground_truth = np.array(ground_truth)
    
    plcc, srcc, krcc, rmse = performance_fit(ground_truth, predictions)
    
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/SRCC', srcc, epoch)
    writer.add_scalar('Val/PLCC', plcc, epoch)
    writer.add_scalar('Val/RMSE', rmse, epoch)
    
    logger.info(f"Validation Epoch {epoch}: Loss={val_loss:.4f}, SRCC={srcc:.4f}, "
                f"KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")
    
    return val_loss, srcc


def validate_two_loss(model, val_loader, criterion1, criterion2, device, epoch, writer, logger):
    """Validate model performance with two loss functions"""
    model.eval()
    val_loss = 0
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion1(outputs, labels.view(-1, 1)) + criterion2(outputs, labels.view(-1, 1))
            
            val_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    predictions = np.array(predictions).squeeze()
    ground_truth = np.array(ground_truth)
    
    plcc, srcc, krcc, rmse = performance_fit(ground_truth, predictions)
    
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/SRCC', srcc, epoch)
    writer.add_scalar('Val/PLCC', plcc, epoch)
    writer.add_scalar('Val/RMSE', rmse, epoch)
    
    logger.info(f"Validation Epoch {epoch}: Loss={val_loss:.4f}, SRCC={srcc:.4f}, "
                f"KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")
    
    return val_loss, srcc

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if it exists"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def train(config):
    """Main training function"""
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    save_dir = Path(config.model_save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging and TensorBoard
    logger = setup_logger(save_dir)
    writer = SummaryWriter(save_dir / 'runs')
    
    # Log GPU usage
    logger.info(f"Using GPUs: {config.gpu_ids}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Get data preprocessing methods
    train_transform, val_transform = get_transforms(config)
    
    # Create datasets and data loaders
    train_dataset = FIQADataset(
        data_dir=config.data_dir,
        csv_path=config.csv_path,
        transform=train_transform,
        is_train=True,
        train_ratio=config.train_ratio
    )
    
    val_dataset = FIQADataset(
        data_dir=config.data_dir,
        csv_path=config.csv_path,
        transform=val_transform,
        is_train=False,
        train_ratio=config.train_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info(f"Load the model: {config.model_name}")
    if config.model_name == 'FIQA_Swin_B':
        model = FIQA_model.FIQA_Swin_B(pretrained_path=config.pretrained_path, is_pretrained=True)
    elif config.model_name == 'FIQA_EdgeNeXt_XXS':
        model = FIQA_model.FIQA_EdgeNeXt_XXS(is_pretrained=True)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Print model information
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable params: {param_num/1e6:.2f} million')
    
    # Define loss function and optimizer
    if config.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif config.loss_type == 'plcc':
        criterion = plcc_loss
    elif config.loss_type == 'mse+plcc':
        criterion1 = nn.MSELoss()
        criterion2 = plcc_loss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.decay_epochs,
        gamma=config.decay_ratio
    )
    
    # Training loop
    best_srcc = -1
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        batch_start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if config.loss_type == 'mse+plcc':
                loss = criterion1(outputs, labels.view(-1, 1)) + criterion2(outputs, labels.view(-1, 1))
            else:
                loss = criterion(outputs, labels.view(-1, 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Epoch [{epoch+1}/{config.num_epochs}], "
                    f"Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Time: {batch_time:.2f}s"
                )
                writer.add_scalar('Train/BatchLoss', avg_loss, 
                                epoch * len(train_loader) + batch_idx)
                batch_start_time = time.time()
        
        train_loss /= len(train_loader)
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        
        # Validation
        if config.loss_type == 'mse+plcc':
            val_loss, srcc = validate_two_loss(model, val_loader, criterion1, criterion2, device, epoch, writer, logger)
        else:
            val_loss, srcc = validate(model, val_loader, criterion, device, epoch, writer, logger)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Save best model
        if srcc > best_srcc:
            best_srcc = srcc
            best_epoch = epoch
            # Remove 'module.' prefix before saving
            model_state_dict = remove_module_prefix(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_srcc': best_srcc,
            }, save_dir / 'best_model.pth')
            logger.info(f"New best model saved at epoch {epoch+1}")
    
    # Training completed, record total time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/3600:.2f} hours")
    logger.info(f"Best SRCC: {best_srcc:.4f} at epoch {best_epoch+1}")
    
    writer.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument('--loss_type', type=str, default='mse')    
    # Load configuration
    config = TrainingConfig()
    
    # Print configuration information
    print("\nTraining Configuration:")
    print("-" * 100)
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("-" * 100 + "\n")
    
    # Start training
    train(config)