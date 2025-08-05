from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data related
    data_dir: str = '/data/sunwei_data/VQualA_FIQA/train'  # Original training data directory
    gfiqa_data_dir: str = '/data/sunwei_data/GFIQA/image'  # GFIQA dataset directory
    csv_path: str = 'data_file/train.csv'
    gfiqa_csv_path: str = 'data_file/gfiqa_results.csv'  # Path to GFIQA dataset CSV file
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
    loss_type: str = 'mse+plcc'  # Options: 'mse', 'plcc', 'mse+plcc'
    
    # Model related
    model_name: str = 'FIQA_EdgeNeXt_XXS'
    model_save_dir: str = '/data/sunwei_data/ModelFolder/VQualA_FIQA/FIQA_EdgeNeXt_XXS_with_gfiqa'
    
    # Image processing
    image_size: int = 352
    image_crop: int = 352
    
    # Device
    gpu_ids: str = "0"  # GPU IDs to use, separated by commas
    
    # Logging
    log_interval: int = 200  # Print log every N batches
