#!/usr/bin/env python3
"""
Main training and evaluation script for Vanilla RNN Seq2Seq model.
Task 1: Vanilla RNN-based Seq2Seq implementation.
"""

import torch
import argparse
import os
from pathlib import Path
import json

from data_loader import get_dataloaders
from models import VanillaRNNSeq2Seq
from train import train
from metrics import evaluate_model, print_evaluation_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Vanilla RNN Seq2Seq model')
    
    # Data arguments
    parser.add_argument('--train_size', type=int, default=5000, help='Number of training examples')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation examples')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test examples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_docstring_len', type=int, default=50, help='Maximum docstring length')
    parser.add_argument('--max_code_len', type=int, default=80, help='Maximum code length')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, 
                       help='Teacher forcing ratio during training')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--save_dir', type=str, default='checkpoints/vanilla_rnn',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("VANILLA RNN SEQ2SEQ - TASK 1")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Train size: {args.train_size}")
    print(f"Val size: {args.val_size}")
    print(f"Test size: {args.test_size}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        max_docstring_len=args.max_docstring_len,
        max_code_len=args.max_code_len
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}\n")
    
    # Create model
    print("Creating Vanilla RNN Seq2Seq model...")
    model = VanillaRNNSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")
    
    # Train model
    print("Training model...")
    history = train(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    model = model.to(args.device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, tokenizer, args.device)
    print_evaluation_metrics(test_metrics)
    
    # Save test metrics
    metrics_path = os.path.join(args.save_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")
    
    # Save configuration
    config = {
        'model_type': 'VanillaRNNSeq2Seq',
        'vocab_size': vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'max_docstring_len': args.max_docstring_len,
        'max_code_len': args.max_code_len,
        'num_params': int(num_params)
    }
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}\n")
    
    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
