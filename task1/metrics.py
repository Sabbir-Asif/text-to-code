import torch
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ast


def token_accuracy(predictions, targets, padding_idx=0):
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: (batch_size, seq_len, vocab_size) - logits from model
        targets: (batch_size, seq_len) - target token IDs
        padding_idx: Index of padding token to ignore
    
    Returns:
        accuracy: Proportion of correctly predicted tokens (excluding padding)
    """
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)  # (batch_size, seq_len)
    
    # Create mask for non-padding tokens
    mask = (targets != padding_idx)
    
    # Compute accuracy
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    
    return accuracy


def bleu_score(predictions, targets, tokenizer, max_n=4, padding_idx=0):
    """
    Calculate BLEU score.
    
    Args:
        predictions: (batch_size, seq_len, vocab_size) - logits from model
        targets: (batch_size, seq_len) - target token IDs
        tokenizer: Tokenizer object with decode method
        max_n: Maximum n-gram order
        padding_idx: Index of padding token
    
    Returns:
        avg_bleu: Average BLEU score across batch
    """
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)  # (batch_size, seq_len)
    
    batch_size = predictions.size(0)
    bleu_scores = []
    
    for i in range(batch_size):
        # Get predicted and target sequences (remove padding)
        pred_seq = pred_tokens[i]
        target_seq = targets[i]
        
        # Find end of sequence (first padding token)
        target_mask = (target_seq != padding_idx)
        if target_mask.any():
            target_end = target_mask.nonzero()[-1].item() + 1
        else:
            target_end = len(target_seq)
        
        # Trim to actual length
        pred_seq = pred_seq[:target_end].tolist()
        target_seq = target_seq[:target_end].tolist()
        
        # Decode to text
        pred_text = tokenizer.decode(pred_seq)
        target_text = tokenizer.decode(target_seq)
        
        # Split into words
        pred_words = pred_text.split()
        reference_words = [target_text.split()]
        
        # Calculate BLEU
        if len(pred_words) > 0:
            smooth_func = SmoothingFunction().method1
            try:
                bleu = sentence_bleu(reference_words, pred_words, 
                                    weights=[1/max_n]*max_n,
                                    smoothing_function=smooth_func)
            except:
                bleu = 0.0
        else:
            bleu = 0.0
        
        bleu_scores.append(bleu)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


def exact_match_accuracy(predictions, targets, tokenizer, padding_idx=0):
    """
    Calculate exact match accuracy (percentage of perfectly generated sequences).
    
    Args:
        predictions: (batch_size, seq_len, vocab_size) - logits from model
        targets: (batch_size, seq_len) - target token IDs
        tokenizer: Tokenizer object
        padding_idx: Index of padding token
    
    Returns:
        accuracy: Proportion of exactly matching sequences
    """
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)  # (batch_size, seq_len)
    
    batch_size = predictions.size(0)
    exact_matches = 0
    
    for i in range(batch_size):
        # Get predicted and target sequences
        pred_seq = pred_tokens[i]
        target_seq = targets[i]
        
        # Find actual lengths (before padding)
        target_mask = (target_seq != padding_idx)
        if target_mask.any():
            target_len = target_mask.nonzero()[-1].item() + 1
        else:
            target_len = len(target_seq)
        
        pred_seq = pred_seq[:target_len]
        target_seq = target_seq[:target_len]
        
        # Check exact match
        if torch.equal(pred_seq, target_seq):
            exact_matches += 1
    
    return exact_matches / batch_size


def is_valid_python(code_str):
    """
    Check if generated code is valid Python syntax.
    
    Args:
        code_str: Generated code string
    
    Returns:
        is_valid: Boolean indicating if code is syntactically valid
    """
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False


def parse_error_rate(predictions, targets, tokenizer, padding_idx=0):
    """
    Calculate syntax error rate.
    
    Args:
        predictions: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        tokenizer: Tokenizer object
        padding_idx: Padding token index
    
    Returns:
        valid_rate: Proportion of syntactically valid code
        error_rate: Proportion of code with syntax errors
    """
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)
    
    batch_size = predictions.size(0)
    valid_count = 0
    
    for i in range(batch_size):
        # Get predicted sequence
        pred_seq = pred_tokens[i]
        
        # Decode to text
        pred_text = tokenizer.decode(pred_seq.tolist())
        
        # Check syntax
        if is_valid_python(pred_text):
            valid_count += 1
    
    valid_rate = valid_count / batch_size
    error_rate = 1.0 - valid_rate
    
    return valid_rate, error_rate


def evaluate_model(model, dataloader, tokenizer, device, padding_idx=0):
    """
    Comprehensive evaluation of model performance.
    
    Args:
        model: Seq2Seq model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer object
        device: Device to evaluate on
        padding_idx: Padding token index
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    total_token_acc = 0.0
    total_bleu = 0.0
    total_exact_match = 0.0
    total_valid_rate = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            source = batch['docstring'].to(device)
            source_lens = batch['docstring_len'].to(device)
            target = batch['code'].to(device)
            
            # Forward pass
            outputs = model(source, source_lens, target, None, teacher_forcing_ratio=0.0)
            
            # Compute metrics
            token_acc = token_accuracy(outputs, target, padding_idx)
            total_token_acc += token_acc
            
            bleu = bleu_score(outputs, target, tokenizer, padding_idx=padding_idx)
            total_bleu += bleu
            
            exact_match = exact_match_accuracy(outputs, target, tokenizer, padding_idx)
            total_exact_match += exact_match
            
            valid_rate, _ = parse_error_rate(outputs, target, tokenizer, padding_idx)
            total_valid_rate += valid_rate
            
            num_batches += 1
    
    metrics = {
        'token_accuracy': total_token_acc / num_batches,
        'bleu_score': total_bleu / num_batches,
        'exact_match_accuracy': total_exact_match / num_batches,
        'valid_python_rate': total_valid_rate / num_batches
    }
    
    return metrics


def print_evaluation_metrics(metrics):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:.<40} {metric_value:.4f}")
    print("="*50 + "\n")
