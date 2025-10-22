from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import yaml

from .data_processor import ProductDataset
from .model_trainer import PricePredictionModel
from .utils import calculate_smape, setup_logger

def predict_prices(config_path: str) -> None:
    """
    Run the prediction pipeline to generate price predictions for test data.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger('predict', config['paths']['log_dir'] + '/predict.log')
    logger.info("Starting prediction pipeline")
    
    # Initialize device
    device = torch.device(config['training']['device'])
    
    # Load test data
    test_dataset = ProductDataset(
        csv_file=config['data']['test_csv'],
        image_dir=config['data']['image_dir'],
        tokenizer_name=config['model']['text_model'],
        max_length=config['model']['max_text_length'],
        image_size=config['model']['image_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Load model
    model = PricePredictionModel(config)
    model_path = Path(config['paths']['model_save_dir']) / 'best_model.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Make predictions
    predictions = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get sample IDs before moving batch to device
            batch_sample_ids = batch['sample_id']
            
            # Move rest of batch to device
            device_batch = {k: v.to(device) for k, v in batch.items() if k != 'sample_id'}
            outputs = model(device_batch)
            
            predictions.extend(outputs.cpu().numpy().tolist())
            sample_ids.extend(batch_sample_ids)
    
    # Create and save predictions DataFrame
    predictions_df = pd.DataFrame({
        'sample_id': [int(sid) for sid in sample_ids],  # Convert to integer
        'price': [float(p) for p in predictions]  # Convert to float
    })
    
    # Ensure positive prices
    predictions_df['price'] = predictions_df['price'].abs()
    
    output_path = Path(config['paths']['prediction_output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False, float_format='%.6f')
    
    logger.info(f"Predictions saved to {output_path}")