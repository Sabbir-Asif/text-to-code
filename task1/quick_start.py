#!/usr/bin/env python3
"""
QUICK START GUIDE - Vanilla RNN Seq2Seq Training
This script provides an easy way to start training with common configurations.
"""

import subprocess
import sys
import argparse


def run_training(config='default', num_epochs=10):
    """
    Run training with predefined configurations.
    
    Args:
        config: Configuration to use ('default', 'quick', 'small', 'large')
        num_epochs: Number of epochs to train
    """
    
    configs = {
        'default': {
            'train_size': 5000,
            'val_size': 1000,
            'test_size': 1000,
            'batch_size': 32,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 1,
            'learning_rate': 0.001
        },
        'quick': {
            'train_size': 500,
            'val_size': 100,
            'test_size': 100,
            'batch_size': 16,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 1,
            'learning_rate': 0.001
        },
        'small': {
            'train_size': 2000,
            'val_size': 500,
            'test_size': 500,
            'batch_size': 32,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 1,
            'learning_rate': 0.001
        },
        'large': {
            'train_size': 10000,
            'val_size': 2000,
            'test_size': 2000,
            'batch_size': 64,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'learning_rate': 0.0005
        }
    }
    
    if config not in configs:
        print(f"Unknown config: {config}")
        print(f"Available configs: {', '.join(configs.keys())}")
        return
    
    cfg = configs[config]
    
    # Build command
    cmd = [
        sys.executable, 'train_vanilla_rnn.py',
        '--train_size', str(cfg['train_size']),
        '--val_size', str(cfg['val_size']),
        '--test_size', str(cfg['test_size']),
        '--batch_size', str(cfg['batch_size']),
        '--embedding_dim', str(cfg['embedding_dim']),
        '--hidden_dim', str(cfg['hidden_dim']),
        '--num_layers', str(cfg['num_layers']),
        '--learning_rate', str(cfg['learning_rate']),
        '--num_epochs', str(num_epochs)
    ]
    
    print("="*70)
    print(f"Starting Vanilla RNN Training - Config: {config.upper()}")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in cfg.items():
        print(f"  {key:.<20} {value}")
    print(f"  {'num_epochs':.<20} {num_epochs}")
    print("\nCommand:")
    print(f"  {' '.join(cmd)}\n")
    print("="*70 + "\n")
    
    # Run training
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick start Vanilla RNN training')
    parser.add_argument('--config', type=str, default='default',
                       choices=['quick', 'small', 'default', 'large'],
                       help='Configuration to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    sys.exit(run_training(config=args.config, num_epochs=args.epochs))
