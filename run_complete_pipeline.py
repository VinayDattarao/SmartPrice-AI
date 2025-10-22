import argparse
from pathlib import Path

from src.setup_dataset import setup_dataset
from src.train_pipeline import train_model
from src.predict_pipeline import predict_prices

def main():
    parser = argparse.ArgumentParser(description='PriceSage ML Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'complete'],
                      default='complete', help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Ensure config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Setup dataset first
    setup_dataset(args.config)
    
    # Run the pipeline based on mode
    if args.mode in ['train', 'complete']:
        train_model(args.config)
    
    if args.mode in ['predict', 'complete']:
        predict_prices(args.config)

if __name__ == '__main__':
    main()