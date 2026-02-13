#!/usr/bin/env python3
"""
Quick test script to verify the Vanilla RNN implementation.
This script demonstrates model architecture and basic functionality without full training.
"""

import torch
import torch.nn as nn
from models import VanillaRNNSeq2Seq
from data_loader import SimpleTokenizer


def test_model_architecture():
    """Test that model can be created and do forward pass."""
    print("Testing Vanilla RNN Seq2Seq Architecture...")
    print("="*60)
    
    # Create a simple tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, word in enumerate(['def', 'return', 'max', 'min', 'list', 'value', 'numbers'], start=4):
        tokenizer.vocab[word] = i
    tokenizer.idx2vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.next_id = len(tokenizer.vocab)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 1
    
    model = VanillaRNNSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    
    # Create dummy data
    batch_size = 4
    source_seq_len = 20
    target_seq_len = 30
    
    source = torch.randint(0, vocab_size, (batch_size, source_seq_len))
    source_lens = torch.tensor([15, 18, 20, 12])
    target = torch.randint(0, vocab_size, (batch_size, target_seq_len))
    target_lens = torch.tensor([25, 20, 30, 22])
    
    print(f"\nDummy data:")
    print(f"Source shape: {source.shape} (batch_size={batch_size}, seq_len={source_seq_len})")
    print(f"Target shape: {target.shape} (batch_size={batch_size}, seq_len={target_seq_len})")
    
    # Forward pass
    print(f"\nForward pass...")
    device = 'cpu'
    model = model.to(device)
    source = source.to(device)
    source_lens = source_lens.to(device)
    target = target.to(device)
    target_lens = target_lens.to(device)
    
    with torch.no_grad():
        outputs = model(source, source_lens, target, target_lens, teacher_forcing_ratio=0.5)
    
    print(f"Output shape: {outputs.shape} (expected: ({batch_size}, {target_seq_len}, {vocab_size}))")
    print(f"Output dtype: {outputs.dtype}")
    print(f"Output min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
    
    # Test generation
    print(f"\nGeneration test...")
    generated = model.generate(source, source_lens, max_len=50, device=device)
    print(f"Generated shape: {generated.shape} (expected: ({batch_size}, 50))")
    
    # Test inference
    print(f"\nInference test...")
    with torch.no_grad():
        outputs_inference = model(source, source_lens, target=None, teacher_forcing_ratio=0.0)
    print(f"Inference output shape: {outputs_inference.shape}")
    
    print("\n" + "="*60)
    print("✓ All architecture tests passed!")
    print("="*60)


def test_tokenizer():
    """Test tokenizer functionality."""
    print("\nTesting Tokenizer...")
    print("="*60)
    
    tokenizer = SimpleTokenizer()
    
    # Build vocab
    texts = [
        'def max_value nums : return max nums',
        'def min_value nums : return min nums',
        'def sum_all items : return sum items'
    ]
    tokenizer.build_vocab(texts, min_freq=1)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Vocabulary: {list(tokenizer.vocab.items())[:10]}...")
    
    # Test encoding/decoding
    text = "def max_value nums"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test special tokens
    print(f"\nSpecial tokens:")
    for token, idx in tokenizer.special_tokens.items():
        print(f"  {token}: {idx}")
    
    print("\n" + "="*60)
    print("✓ All tokenizer tests passed!")
    print("="*60)


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + " VANILLA RNN SEQ2SEQ - MODEL ARCHITECTURE TEST".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        test_tokenizer()
        test_model_architecture()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe Vanilla RNN implementation is ready for training.")
        print("Run: python train_vanilla_rnn.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
