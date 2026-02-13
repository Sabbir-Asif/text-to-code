import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNEncoder(nn.Module):
    """Vanilla RNN encoder for Seq2Seq."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(VanillaRNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, sequences, seq_lens):
        """
        Args:
            sequences: (batch_size, seq_len)
            seq_lens: (batch_size,) actual length of each sequence
        
        Returns:
            encoder_outputs: (batch_size, seq_len, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through RNN
        packed_output, hidden = self.rnn(packed)
        
        # Unpack
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        return encoder_outputs, hidden


class VanillaRNNDecoder(nn.Module):
    """Vanilla RNN decoder for Seq2Seq."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(VanillaRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
    
    def forward(self, sequences, hidden):
        """
        Args:
            sequences: (batch_size, seq_len)
            hidden: (num_layers, batch_size, hidden_dim)
        
        Returns:
            output: (batch_size, seq_len, vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        logits = self.fc(output)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden


class VanillaRNNSeq2Seq(nn.Module):
    """Vanilla RNN-based Seq2Seq model without attention."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.0):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(VanillaRNNSeq2Seq, self).__init__()
        self.encoder = VanillaRNNEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = VanillaRNNDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.vocab_size = vocab_size
    
    def forward(self, source, source_lens, target=None, target_lens=None, teacher_forcing_ratio=0.5):
        """
        Args:
            source: (batch_size, source_seq_len) - docstring sequences
            source_lens: (batch_size,) - actual length of each source sequence
            target: (batch_size, target_seq_len) - code sequences
            target_lens: (batch_size,) - actual length of each target sequence
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: (batch_size, target_seq_len, vocab_size) - logits for each token
        """
        batch_size = source.size(0)
        target_seq_len = target.size(1) if target is not None else 80
        device = source.device
        
        # Encoder
        encoder_outputs, hidden = self.encoder(source, source_lens)
        
        # Decoder - use fixed-length context from final hidden state
        outputs = []
        
        if target is not None:
            # Training with teacher forcing
            for t in range(target_seq_len):
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Use ground truth token
                    decoder_input = target[:, t:t+1]
                else:
                    # Use previous prediction
                    if t == 0:
                        decoder_input = target[:, t:t+1]
                    else:
                        decoder_input = predictions[:, -1:].argmax(-1)
                
                logits, hidden = self.decoder(decoder_input, hidden)
                outputs.append(logits)
                
                if t == 0:
                    predictions = logits
                else:
                    predictions = torch.cat([predictions, logits], dim=1)
        else:
            # Inference - greedy decoding
            decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # SOS token
            
            for t in range(target_seq_len):
                logits, hidden = self.decoder(decoder_input, hidden)
                top_tokens = logits.argmax(dim=-1)
                outputs.append(logits)
                decoder_input = top_tokens
        
        outputs = torch.cat(outputs, dim=1)  # (batch_size, target_seq_len, vocab_size)
        return outputs
    
    def generate(self, source, source_lens, max_len=80, device='cpu'):
        """
        Generate target sequences from source sequences.
        
        Args:
            source: (batch_size, source_seq_len)
            source_lens: (batch_size,)
            max_len: Maximum length of generated sequence
            device: Device to run on
        
        Returns:
            generated: (batch_size, max_len) - generated token IDs
        """
        batch_size = source.size(0)
        
        # Encoder
        source = source.to(device)
        source_lens = source_lens.to(device)
        encoder_outputs, hidden = self.encoder(source, source_lens)
        
        # Decoder - greedy decoding
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # SOS token
        
        for t in range(max_len - 1):
            logits, hidden = self.decoder(generated[:, -1:], hidden)
            next_tokens = logits.argmax(dim=-1)  # (batch_size, 1)
            generated = torch.cat([generated, next_tokens], dim=1)
        
        return generated
