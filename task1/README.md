# Text-to-Code Generation with Seq2Seq Models

Implementation of sequence-to-sequence models for generating Python code from natural language docstrings.

## Task 1: Vanilla RNN Seq2Seq

This is the baseline implementation using vanilla RNNs (without LSTM or attention).

### Architecture

- **Encoder**: RNN that processes the input docstring
- **Decoder**: RNN that generates the output code token-by-token
- **Context**: Fixed-length context vector from encoder's final hidden state
- **No Attention**: All information from encoder is compressed into a single context vector

### Project Structure

```
├── data_loader.py          # Dataset loading and preprocessing
├── models.py               # Vanilla RNN Seq2Seq model architecture
├── metrics.py              # Evaluation metrics (BLEU, accuracy, etc.)
├── train.py                # Training loop and utilities
├── inference.py            # Inference and code generation utilities
├── train_vanilla_rnn.py    # Main training script
└── instructions/
    └── assignment.txt      # Assignment description
```

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install pygame torch torchaudio torchvision datasets transformers tqdm nltk
```

### Usage

#### Train the Vanilla RNN model:

```bash
python train_vanilla_rnn.py --num_epochs 10 --batch_size 32 --learning_rate 0.001
```

**Optional arguments:**
- `--train_size`: Number of training examples (default: 5000)
- `--val_size`: Number of validation examples (default: 1000)
- `--test_size`: Number of test examples (default: 1000)
- `--batch_size`: Batch size (default: 32)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_dim`: Hidden dimension (default: 256)
- `--num_layers`: Number of RNN layers (default: 1)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 0.001)
- `--device`: Device to train on ('cuda' or 'cpu', auto-detected)
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints/vanilla_rnn')
- `--seed`: Random seed (default: 42)

#### Example training command:

```bash
python train_vanilla_rnn.py \
    --train_size 5000 \
    --batch_size 32 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_epochs 10 \
    --learning_rate 0.001 \
    --device cuda
```

### Dataset

- **Source**: CodeSearchNet Python from Hugging Face
- **Language Pair**: English docstrings → Python code
- **Default Split**: 5000 training + 1000 validation + 1000 test examples
- **Sequence Lengths**: Docstrings up to 50 tokens, Code up to 80 tokens

### Model Details

#### Encoder
- Embedding layer with dimension 128
- RNN with 256 hidden units
- Produces a context vector from the final hidden state

#### Decoder
- Embedding layer with dimension 128
- RNN with 256 hidden units
- Uses teacher forcing during training (probability: 0.5)
- Greedy decoding during inference

#### Training Configuration
- **Loss**: Cross-entropy loss (ignoring padding tokens)
- **Optimizer**: Adam (lr=0.001)
- **Teacher Forcing**: Applied with probability 0.5
- **Gradient Clipping**: Max norm of 1.0
- **Batch Size**: 32 examples

### Evaluation Metrics

1. **Token Accuracy**: Percentage of correctly predicted tokens

2. **BLEU Score**: N-gram overlap between generated and reference code
   - Useful for evaluating code similarity
   - Scores range from 0 to 1

3. **Exact Match Accuracy**: Percentage of completely correct outputs
   - Strict metric: generated code must exactly match reference

4. **Syntax Validity**: Percentage of generated code with valid Python syntax
   - Uses Python AST to validate syntax

### Output

After training, checkpoints are saved to `checkpoints/vanilla_rnn/`:

- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_*.pt`: Checkpoint at each epoch
- `training_history.json`: Training and validation loss curves
- `config.json`: Model configuration
- `test_metrics.json`: Test set evaluation metrics

### Expected Results

The vanilla RNN model serves as a baseline and typically shows:

- **Token Accuracy**: ~30-45%
- **BLEU Score**: ~0.15-0.25
- **Exact Match**: ~5-15%
- **Valid Python Rate**: ~20-40%

Performance limitations:
- Struggles with longer docstrings (>30 tokens)
- Fixed-length context vector becomes bottleneck
- May forget important information from start of docstring
- Difficulty with long-range dependencies

### Known Limitations

1. **Vanishing Gradient Problem**: RNNs struggle with long sequences
2. **Fixed Context Bottleneck**: All information compressed into single vector
3. **Information Loss**: Early tokens may be forgotten
4. **Scalability**: Difficult to generate longer code sequences

These limitations are expected and overcome in Task 2 (LSTM) and Task 3 (Attention).

### Code Quality

- Well-documented classes and functions
- Type hints where applicable
- Modular design for easy extension
- Reproducible with fixed random seed
- Progress bars during training and evaluation

### References

- Sequence-to-Sequence Learning with Neural Networks (Sutskever et al., 2014)
- Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
- CodeSearchNet Dataset: https://github.com/github/CodeSearchNet
