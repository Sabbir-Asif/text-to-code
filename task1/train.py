import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import os
from tqdm import tqdm
import json
from pathlib import Path


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5, clip=1.0):
    """
    Train for one epoch.
    
    Args:
        model: Seq2Seq model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        teacher_forcing_ratio: Probability of using teacher forcing
        clip: Gradient clipping value
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Get data
        source = batch['docstring'].to(device)
        source_lens = batch['docstring_len'].to(device)
        target = batch['code'].to(device)
        target_lens = batch['code_len'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(source, source_lens, target, target_lens, teacher_forcing_ratio)
        
        # Compute loss (ignore PAD token)
        # Reshape for loss computation
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * target_seq_len, vocab_size)
        targets_flat = target.view(-1)  # (batch_size * target_seq_len,)
        
        loss = criterion(outputs, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), clip)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: Seq2Seq model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Get data
            source = batch['docstring'].to(device)
            source_lens = batch['docstring_len'].to(device)
            target = batch['code'].to(device)
            
            # Forward pass
            outputs = model(source, source_lens, target, None, teacher_forcing_ratio=0.0)
            
            # Compute loss
            outputs = outputs.view(-1, outputs.size(-1))
            targets_flat = target.view(-1)
            
            loss = criterion(outputs, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
          device='cpu', save_dir='checkpoints'):
    """
    Train the Seq2Seq model.
    
    Args:
        model: Seq2Seq model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        history: Dictionary with training history
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining completed. History saved to {history_path}")
    
    return history


def create_vanilla_rnn_trainer(train_loader, val_loader, vocab_size, 
                               embedding_dim=128, hidden_dim=256, num_layers=1,
                               num_epochs=10, learning_rate=0.001, device='cpu',
                               save_dir='checkpoints/vanilla_rnn'):
    """
    Create and train a vanilla RNN Seq2Seq model.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of RNN layers
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        model: Trained model
        history: Training history
    """
    from models import VanillaRNNSeq2Seq
    
    # Create model
    model = VanillaRNNSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # Train
    history = train(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir
    )
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model, history
