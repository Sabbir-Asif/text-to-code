"""
Utility functions for inference and code generation.
"""

import torch
import torch.nn as nn


class CodeGenerator:
    """Generate code from docstrings using a trained model."""
    
    def __init__(self, model, tokenizer, device='cpu', max_len=80):
        """
        Args:
            model: Trained Seq2Seq model
            tokenizer: Tokenizer object
            device: Device to run on
            max_len: Maximum length of generated code
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()
    
    def generate(self, docstring):
        """
        Generate code from a docstring.
        
        Args:
            docstring: Text description of the function
        
        Returns:
            code: Generated Python code
        """
        with torch.no_grad():
            # Tokenize docstring
            tokens = self.tokenizer.encode(docstring)
            tokens = tokens[:50]  # Limit to max docstring length
            
            # Convert to tensor
            source = torch.tensor([tokens], dtype=torch.long).to(self.device)
            source_lens = torch.tensor([len(tokens)], dtype=torch.long).to(self.device)
            
            # Generate
            generated = self.model.generate(source, source_lens, max_len=self.max_len, device=self.device)
            
            # Decode
            code = self.tokenizer.decode(generated[0].tolist())
            
            return code
    
    def batch_generate(self, docstrings):
        """
        Generate code for multiple docstrings.
        
        Args:
            docstrings: List of docstrings
        
        Returns:
            codes: List of generated code strings
        """
        codes = []
        for docstring in docstrings:
            code = self.generate(docstring)
            codes.append(code)
        return codes


def get_generated_samples(model, tokenizer, test_loader, device, num_samples=5):
    """
    Get sample generations from the model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_loader: Test dataloader
        device: Device
        num_samples: Number of samples to generate
    
    Returns:
        samples: List of dicts with 'docstring', 'reference', 'generated'
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(samples) >= num_samples:
                break
            
            # Get data
            source = batch['docstring'].to(device)
            source_lens = batch['docstring_len'].to(device)
            target = batch['code'].to(device)
            
            # Generate
            generated = model.generate(source, source_lens, max_len=80, device=device)
            
            batch_size = source.size(0)
            for i in range(batch_size):
                if len(samples) >= num_samples:
                    break
                
                # Decode docstring
                docstring_tokens = source[i, :source_lens[i]].tolist()
                docstring = tokenizer.decode(docstring_tokens)
                
                # Decode reference
                target_tokens = target[i].tolist()
                reference = tokenizer.decode(target_tokens)
                
                # Decode generated
                gen_tokens = generated[i].tolist()
                generated_code = tokenizer.decode(gen_tokens)
                
                samples.append({
                    'docstring': docstring,
                    'reference': reference,
                    'generated': generated_code
                })
    
    return samples


def print_samples(samples):
    """
    Print sample generations in a readable format.
    
    Args:
        samples: List of sample dicts
    """
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}")
        print(f"{'='*60}")
        print(f"Docstring: {sample['docstring']}")
        print(f"\nReference Code:\n{sample['reference']}")
        print(f"\nGenerated Code:\n{sample['generated']}")
        print(f"{'='*60}")


def compare_predictions(predictions, targets, tokenizer, padding_idx=0, num_samples=5):
    """
    Compare predictions with targets and print samples.
    
    Args:
        predictions: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        tokenizer: Tokenizer
        padding_idx: Padding token index
        num_samples: Number of samples to print
    """
    pred_tokens = predictions.argmax(dim=-1)
    batch_size = min(num_samples, predictions.size(0))
    
    for i in range(batch_size):
        # Get sequences
        pred_seq = pred_tokens[i].tolist()
        target_seq = targets[i].tolist()
        
        # Decode
        pred_text = tokenizer.decode(pred_seq)
        target_text = tokenizer.decode(target_seq)
        
        # Print
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"{'='*60}")
        print(f"Target:\n{target_text}")
        print(f"\nPrediction:\n{pred_text}")
        print(f"{'='*60}")
