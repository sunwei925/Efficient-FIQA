from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data related
    data_dir: str = '/data/sunwei_data/VQualA_FIQA/train'
    csv_path: str = 'data_file/train.csv'
    train_ratio: float = 0.8
    
    # Training related
    num_epochs: int = 30
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    
    # Optimizer related
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    decay_ratio: float = 0.1
    decay_epochs: int = 10
    loss_type: str = 'mse+plcc'
    
    # Model related
    model_name: str = 'FIQA_Swin_B'
    pretrained_path: str  = None
    model_save_dir: str = '/data/sunwei_data/ModelFolder/VQualA_FIQA/FIQA_Swin_B'
    
    # Image processing
    image_size: int = 448
    image_crop: int = 448
    
    # Device
    gpu_ids: str = "0,1,2,3"  # GPU IDs to use, separated by commas
    
    # Logging
    log_interval: int = 200  # Print log every N batches
    save_interval: int = 1  # Save model every N epochs
