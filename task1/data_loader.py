import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re

class CodeSearchNetDataset(Dataset):
    """Dataset class for CodeSearchNet Python dataset."""
    
    def __init__(self, data, tokenizer, max_docstring_len=50, max_code_len=80):
        """
        Args:
            data: Hugging Face dataset object
            tokenizer: Tokenizer object with encode and decode methods
            max_docstring_len: Maximum docstring token length
            max_code_len: Maximum code token length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_docstring_len = max_docstring_len
        self.max_code_len = max_code_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        docstring = item['docstring'].strip()
        code = item['code'].strip()
        
        # Tokenize
        docstring_tokens = self.tokenizer.encode(docstring)
        code_tokens = self.tokenizer.encode(code)
        
        # Truncate
        docstring_tokens = docstring_tokens[:self.max_docstring_len]
        code_tokens = code_tokens[:self.max_code_len]
        
        return {
            'docstring': docstring_tokens,
            'code': code_tokens,
            'docstring_len': len(docstring_tokens),
            'code_len': len(code_tokens)
        }


class SimpleTokenizer:
    """Simple whitespace-based tokenizer."""
    
    def __init__(self):
        self.vocab = {}
        self.idx2vocab = {}
        self.next_id = 0
        self.special_tokens = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.next_id = 4
        
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.idx2vocab[idx] = token
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts."""
        from collections import Counter
        
        # Count tokens
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Add tokens with freq >= min_freq
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.special_tokens:
                self.vocab[token] = self.next_id
                self.idx2vocab[self.next_id] = token
                self.next_id += 1
    
    @staticmethod
    def tokenize(text):
        """Tokenize text by whitespace."""
        return text.lower().split()
    
    def encode(self, text):
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.special_tokens['<UNK>']) for token in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs to text."""
        tokens = [self.idx2vocab.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(tokens)
    
    def get_vocab_size(self):
        return len(self.vocab)


def collate_batch(batch):
    """Collate function for DataLoader."""
    docstrings = [item['docstring'] for item in batch]
    codes = [item['code'] for item in batch]
    docstring_lens = [item['docstring_len'] for item in batch]
    code_lens = [item['code_len'] for item in batch]
    
    # Pad sequences
    max_doc_len = max(docstring_lens)
    max_code_len = max(code_lens)
    
    docstring_padded = []
    code_padded = []
    
    for doc, code in zip(docstrings, codes):
        # Pad docstrings
        padded_doc = doc + [0] * (max_doc_len - len(doc))  # 0 is PAD token
        docstring_padded.append(padded_doc)
        
        # Pad codes with SOS and EOS tokens
        # Total length = 1 (SOS) + code_len + 1 (EOS) + padding = max_code_len + 2
        padded_code = [1] + code + [2] + [0] * (max_code_len - len(code))  # 1 is SOS, 2 is EOS
        code_padded.append(padded_code)
    
    return {
        'docstring': torch.tensor(docstring_padded, dtype=torch.long),
        'code': torch.tensor(code_padded, dtype=torch.long),
        'docstring_len': torch.tensor(docstring_lens, dtype=torch.long),
        'code_len': torch.tensor(code_lens, dtype=torch.long)
    }


def load_codesearchnet_data(split='train', num_samples=5000):
    """Load CodeSearchNet Python dataset."""
    print(f"Loading CodeSearchNet dataset ({split} split)...")
    dataset = load_dataset('Nan-Do/code-search-net-python', split=split, trust_remote_code=True)
    
    # Take subset
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    print(f"Loaded {len(dataset)} examples")
    return dataset


def get_dataloaders(train_size=5000, val_size=1000, test_size=1000, 
                   batch_size=32, max_docstring_len=50, max_code_len=80):
    """
    Get train, validation, and test dataloaders.
    """
    # Load dataset
    try:
        dataset = load_dataset('Nan-Do/code-search-net-python', split='train', trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and datasets library installed")
        raise
    
    # Take subset
    total_needed = train_size + val_size + test_size
    if len(dataset) > total_needed:
        dataset = dataset.select(range(total_needed))
    
    # Split into train/val/test
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, train_size + val_size + test_size))
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer()
    all_texts = [item['docstring'] for item in dataset] + [item['code'] for item in dataset]
    tokenizer.build_vocab(all_texts, min_freq=1)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create datasets
    train_dataset = CodeSearchNetDataset(train_data, tokenizer, max_docstring_len, max_code_len)
    val_dataset = CodeSearchNetDataset(val_data, tokenizer, max_docstring_len, max_code_len)
    test_dataset = CodeSearchNetDataset(test_data, tokenizer, max_docstring_len, max_code_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)
    
    return train_loader, val_loader, test_loader, tokenizer
