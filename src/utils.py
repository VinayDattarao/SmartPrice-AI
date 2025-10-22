import os
import logging
import requests
from PIL import Image
from io import BytesIO
import time
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import torch
import gc
import numpy as np

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Set up logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

def download_image(url: str, retries: int = 3, delay: int = 1) -> Optional[Image.Image]:
    """Download image from URL with retries and delay."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed to download image from {url}: {str(e)}")
                return None
            time.sleep(delay)
    return None

def get_gpu_memory() -> Tuple[float, float, float]:
    """Get the current gpu memory usage.
    Returns:
        total (float): Total memory in GB
        used (float): Used memory in GB
        free (float): Free memory in GB
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        free = total_memory - allocated
        return total_memory, allocated, free
    return 0, 0, 0

def check_system_resources() -> Tuple[bool, str]:
    """Check if system has enough resources for training."""
    if not torch.cuda.is_available():
        return False, "No GPU available"
    
    # Check GPU memory
    total_mem, used_mem, free_mem = get_gpu_memory()
    if free_mem < 4.0:  # Need at least 4GB free
        return False, f"Not enough GPU memory. Need 4GB, but only {free_mem:.1f}GB free"
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check again after clearing cache
    total_mem, used_mem, free_mem = get_gpu_memory()
    return True, f"System ready: {free_mem:.1f}GB GPU memory available"

def test_model_memory(model: torch.nn.Module, sample_batch: dict) -> Tuple[bool, str]:
    """Test if model can fit in memory with a sample batch."""
    try:
        # Move batch to device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in sample_batch.items()}
        
        # Try processing a batch with mixed precision
        with torch.cuda.amp.autocast():
            _ = model(batch)
        
        # Get memory stats
        _, used_mem, free_mem = get_gpu_memory()
        
        # Check if we have enough headroom (2x current usage)
        if free_mem < used_mem:
            return False, f"Not enough memory headroom. Need {used_mem*2:.1f}GB, have {free_mem+used_mem:.1f}GB total"
        
        return True, f"Model fits in memory with {free_mem:.1f}GB free"
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, "Out of memory during test"
        raise e

def optimize_batch_size(model, dataset, initial_batch_size: int = 32) -> int:
    """Find optimal batch size that fits in memory."""
    if not torch.cuda.is_available():
        return 16  # Default CPU batch size
        
    batch_size = initial_batch_size
    while batch_size > 1:
        try:
            # Try creating a dataloader and processing one batch
            from torch.utils.data import DataLoader
            temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            batch = next(iter(temp_loader))
            
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.cuda.amp.autocast():
                _ = model(batch)
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    return 1

def calc_smape(pred, target) -> float:
    """Calculate SMAPE between predictions and targets"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    # Avoid division by zero
    zero_mask = (np.abs(pred) + np.abs(target)) == 0
    non_zero_mask = ~zero_mask
    
    # Calculate SMAPE only for non-zero elements
    smape = np.zeros_like(pred, dtype=float)
    smape[non_zero_mask] = 200 * np.abs(pred[non_zero_mask] - target[non_zero_mask]) / (
        np.abs(pred[non_zero_mask]) + np.abs(target[non_zero_mask])
    )
    
    return float(np.mean(smape))

def download_images(df: pd.DataFrame, image_dir: str, image_column: str = 'image_link') -> List[str]:
    """Download all images from dataframe URLs and save them."""
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    
    for idx, row in df.iterrows():
        url = row[image_column]
        file_name = f"{row['sample_id']}.jpg"
        file_path = os.path.join(image_dir, file_name)
        
        if os.path.exists(file_path):
            image_paths.append(file_path)
            continue
            
        img = download_image(url)
        if img is not None:
            img.save(file_path)
            image_paths.append(file_path)
        else:
            image_paths.append("")
            
    return image_paths

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)