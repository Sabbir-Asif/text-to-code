# Implementation Details: Seq2Seq RNN Code Generation

## Overview

This project implements and compares three Seq2Seq RNN architectures for Python code generation from natural language docstrings. The implementation consists of two separate, reproducible notebooks designed for independent execution on different machines:

1. **rnn-seq2seq.ipynb** - Training notebook (designed for Google Colab with GPU)
2. **rnn-seq2seq-analytics.ipynb** - Comprehensive analytics notebook (designed for local analysis on M1 MacBook)

---

## Architecture

### Three RNN Models Implemented

#### **Model 1: Vanilla RNN Seq2Seq**
- **Encoder**: Unidirectional RNN with embedding layer + dropout
- **Decoder**: RNN decoder with linear projection to vocabulary
- **Key Characteristics**:
  - Baseline architecture for comparison
  - Suffers from vanishing gradient problem on longer sequences
  - Context vector: Hidden state from encoder's last timestep
  - Parameters: ~1.5-2M (depends on vocabulary size and hyperparameters)

#### **Model 2: LSTM Seq2Seq**
- **Encoder**: Multi-layer bidirectional LSTM with embedding + dropout
- **Decoder**: Multi-layer LSTM with hidden and cell states
- **Key Characteristics**:
  - Improves on Vanilla RNN with gating mechanisms (input, forget, output gates)
  - Maintains both hidden state AND cell state across timestamps
  - Bidirectional encoding captures context from both directions
  - Multi-layer architecture (2 layers by default) for deeper feature extraction
  - Parameters: ~3-4M

#### **Model 3: LSTM with Attention Mechanism**
- **Encoder**: Multi-layer bidirectional LSTM (8000+ dimensional output)
- **Decoder**: LSTM with Bahdanau (additive) attention mechanism
- **Attention Mechanism**:
  - Scores each encoder output: `energy = tanh(W_v @ tanh(W_c @ [decoder_hidden; encoder_output]))`
  - Softmax normalization: `attention_weights = softmax(energy)`
  - Context vector: Weighted sum of encoder outputs
  - Concatenates [embedding, context] as decoder input
- **Key Characteristics**:
  - Removes fixed-context bottleneck of encoder-only RNNs
  - Decoder attends to relevant parts of input at each generation step
  - Attention weights are interpretable (can visualize as heatmaps)
  - Parameters: ~3.5-4.5M (plus attention mechanism overhead)

---

## Configuration

### Hyperparameters (Identical across all models)

```python
CONFIG = {
    # Dataset Configuration
    'TRAIN_SIZE': 10000,           # Training examples
    'VAL_SIZE': 1500,              # Validation examples  
    'TEST_SIZE': 1500,             # Test examples
    
    # Sequence Configuration
    'MAX_DOCSTRING_LEN': 50,       # Max input (source) tokens
    'MAX_CODE_LEN': 80,             # Max output (target) tokens
    
    # Architecture Configuration
    'EMBEDDING_DIM': 256,           # Word embedding dimensionality
    'HIDDEN_DIM': 256,              # RNN hidden state dimensionality
    'NUM_LAYERS': 2,                # Number of LSTM layers (for LSTM models)
    'DROPOUT': 0.3,                 # Dropout probability (regularization)
    'BIDIRECTIONAL': True,          # Use bidirectional encoders
    
    # Training Configuration
    'BATCH_SIZE': 64,               # Batch size for training
    'EPOCHS': 20,                   # Number of training epochs
    'LEARNING_RATE': 0.001,         # Adam optimizer learning rate
    'TEACHER_FORCING_RATIO': 0.5,  # Probability of using ground truth during training
    
    # Optimization Configuration
    'VOCAB_SIZE': 5000,             # Vocabulary size (shared for src and tgt)
    'GRADIENT_CLIP': 1.0,           # Gradient clipping threshold
    'WARMUP_STEPS': 1000,           # Learning rate warmup steps
    'EARLY_STOPPING_PATIENCE': 3,  # Patience for early stopping
    'BEAM_SIZE': 3,                 # Beam search size for inference
    'SCHEDULED_SAMPLING': True,     # Use scheduled sampling during training
}
```

### Key Design Choices

1. **Shared Configuration**: All three models use identical hyperparameters for fair comparison
2. **Dropout Regularization**: 30% dropout prevents overfitting on smaller dataset
3. **Bidirectional Encoding**: Captures context from both directions (critical for code understanding)
4. **Multi-layer Architecture**: 2 LSTM layers allow learning abstract patterns
5. **Gradient Clipping**: Prevents exploding gradients during backpropagation

---

## Dataset

### CodeSearchNet Python

- **Source**: [Hugging Face Datasets - Nan-Do/code-search-net-python](https://huggingface.co/datasets/Nan-Do/code-search-net-python)
- **Preprocessing**:
  - Filter: Both docstring and code must exist and be non-empty
  - Sample: 13,000 examples (10k train / 1.5k val / 1.5k test)
  - Shuffle with seed=42 for reproducibility

### Data Format

Each training example:
```
Input (Docstring):  "Returns the maximum value in the list"
Output (Code):       "return max ( values )"
```

Both are tokenized using whitespace tokenization with special character handling.

---

## Tokenization

### Custom Whitespace Tokenizer

**Special Tokens**:
- `<PAD>` (index 0): Padding token for variable-length sequences
- `<SOS>` (index 1): Start-of-sequence token (prepended to decoder inputs)
- `<EOS>` (index 2): End-of-sequence token (marks end of generation)
- `<UNK>` (index 3): Unknown token (for out-of-vocabulary words)

**Tokenization Strategy**:
1. Lowercase all text
2. Insert spaces around special characters: `()[]{}:,.=+-*/`
3. Split on whitespace
4. Map tokens to vocabulary indices

**Vocabulary Building**:
- Separate vocabularies for source (docstrings) and target (code)
- Selected most common 5000 tokens from training data
- Out-of-vocabulary words mapped to `<UNK>` token

**Serialization**: Tokenizers saved as JSON with:
- `word2idx`: Dictionary mapping tokens to indices
- `idx2word`: Dictionary mapping indices back to tokens
- `vocab_size`: Vocabulary size for reconstruction

---

## Training Pipeline

### Loss Function

- **Cross-Entropy Loss** with `ignore_index=0` (ignores padding tokens)
- Computed on flattened predictions vs flattened targets
- Not averaged over padding positions

### Optimizer

- **Adam** optimizer with learning rate = 0.001
- Adaptive learning rates for each parameter

### Teacher Forcing

- **Start of Training**: 50% of decoder inputs are ground truth tokens
- **Purpose**: Stabilizes training by reducing error accumulation
- **Alternative**: Scheduled sampling (decreases ratio gradually over training)

### Training Loop Pseudocode

```
for epoch in range(EPOCHS):
    # Training
    for batch in train_loader:
        src, tgt = batch
        
        # Forward pass
        predictions = model(src, tgt, teacher_forcing_ratio=0.5)
        
        # Loss calculation
        loss = cross_entropy_loss(predictions[:, 1:], tgt[:, 1:])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    
    # Save best model
    if val_loss < best_val_loss:
        save_checkpoint(model, val_loss, epoch)
    
    # Early stopping
    if no_improvement_for > 3 epochs:
        break
```

### Gradient Clipping

- **Method**: L2 norm clipping
- **Max Norm**: 1.0
- **Purpose**: Prevents exploding gradients in RNNs (common with long sequences)

---

## Inference Methods

### 1. Greedy Decoding

**Algorithm**:
```
decoder_input = <SOS> token
output_sequence = [<SOS>]

for t in range(MAX_CODE_LEN):
    prediction = model.decoder(decoder_input, hidden, cell)
    top_token = argmax(prediction)
    output_sequence.append(top_token)
    
    if top_token == <EOS>:
        break
    
    decoder_input = top_token  # Feed back to decoder
```

**Characteristics**:
- Fast inference (no search required)
- Often suboptimal (greedy choices may lead to bad global solutions)
- Local optima trap common with longer sequences

### 2. Beam Search Decoding

**Algorithm**:
```
# Maintain k=3 best hypotheses
sequences = [[<SOS>]]
scores = [0.0]

for t in range(MAX_CODE_LEN):
    candidates = []
    
    for each sequence in sequences:
        prediction = model.decoder(last_token, hidden, cell)
        
        # Get top-k predictions
        for each top_k_token in prediction:
            new_sequence = sequence + [token]
            new_score = score + log_probability
            candidates.append((new_score, new_sequence))
    
    # Keep only top k sequences
    sequences = top_k(candidates, k=3)

return best_sequence
```

**Characteristics**:
- Explores multiple hypotheses in parallel
- Trades computational cost for better quality
- Typically improves BLEU score by 5-15%
- Parameters: `k=3` (beam size, 3 best hypotheses maintained)

---

## Evaluation Metrics

### 1. BLEU Score (Bilingual Evaluation Understudy)

**Definition**:
$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

Where:
- $p_n$ = precision of n-grams (1-4 grams typically)
- $w_n$ = weights (usually uniform: 0.25 each)
- $BP$ = brevity penalty (penalizes overly short sequences)

**Interpretation**:
- **Range**: 0-100 (higher is better)
- **0-30**: Poor quality
- **30-50**: Reasonable quality
- **50+**: High quality
- **Advantage**: Automatic metric, considers n-gram overlap with reference
- **Disadvantage**: Doesn't capture semantic correctness, especially for code

### 2. Token Accuracy

**Definition**:
$$\text{Token Accuracy} = \frac{\text{# correctly predicted tokens}}{\text{# total tokens in reference}}$$

**Calculation**:
```
For each token position t:
    if predicted[t] == reference[t]:
        correct += 1

accuracy = correct / total_tokens * 100
```

**Interpretation**:
- **Range**: 0-100%
- More fine-grained than exact match
- Shows which positions the model struggles with
- **Advantage**: Direct measure of token-level correctness
- **Disadvantage**: Doesn't reward semantically equivalent variations

### 3. Exact Match Accuracy

**Definition**:
$$\text{Exact Match} = \frac{\text{# sequences matching reference exactly}}{\text{# total sequences}}$$

**Calculation**:
```
for each test example:
    if predicted_code.strip() == reference_code.strip():
        exact_matches += 1

exact_match_accuracy = exact_matches / total_examples * 100
```

**Interpretation**:
- **Range**: 0-100%
- Very strict metric (even minor differences count as failure)
- Most directly relevant for code generation
- **Advantage**: Clear pass/fail, no ambiguity
- **Disadvantage**: Penalizes syntactically correct but stylistically different code

---

## Performance Analysis

### Length-Based Performance Analysis

**Motivation**: Evaluate whether models degrade on longer sequences (indicating RNN limitations)

**Methodology**:
1. Bin test examples by docstring length: [0-10], [10-20], [20-30], [30-40], [40-50] tokens
2. Calculate token accuracy separately for each bin
3. Compare across three models

**Expected Results**:
- **Vanilla RNN**: Significant degradation on longer sequences (vanishing gradient problem)
- **LSTM**: Maintains better performance across length ranges
- **LSTM+Attention**: Relatively consistent performance (attention mechanism helps)

**Analysis Code**:
```python
for docstring, code in test_set:
    doc_length = len(tokenize(docstring))
    prediction = model.generate(docstring)
    
    # Categorize by length bin
    if 0 <= doc_length < 10:
        length_bin = '0-10'
    elif 10 <= doc_length < 20:
        length_bin = '10-20'
    # ... etc
    
    # Calculate accuracy for this example
    accuracy = token_accuracy(prediction, code)
    length_performance[length_bin].append(accuracy)

# Average accuracy per bin
for bin, accuracies in length_performance.items():
    avg_accuracy[bin] = mean(accuracies)
```

---

## Error Analysis

### Error Categories

The analytics notebook categorizes failure cases into 7 types:

1. **Empty Output** (~X%)
   - Model generates no tokens or just `<EOS>`
   - Indicates premature sequence termination
   - Cause: Attention weights collapse or decoder stuck

2. **Incomplete Code** (~X%)
   - Generated code significantly shorter than reference
   - Missing key parts of implementation
   - Cause: Early `<EOS>` triggering

3. **Missing Parentheses** (~X%)
   - Function calls lack parentheses: `print` instead of `print()`
   - Function argument lists missing
   - Cause: Operator prediction errors

4. **Missing Colons** (~X%)
   - Lost colon markers for Python structure: `if x` instead of `if x:`
   - Affects indentation and control flow readability
   - Cause: Structure token prediction failure

5. **Missing Return Statements** (~X%)
   - `return` keyword not generated
   - Critical for function correctness
   - Cause: Semantic understanding gap

6. **Wrong Operators** (~X%)
   - Comparison operators: `>` vs `<`, `==` vs `!=`
   - Arithmetic operators: `+` vs `-`, `*` vs `/`
   - Assignment context misplaced
   - Cause: Subtle semantic confusion

7. **Other Errors** (~X%)
   - Variable naming mistakes
   - Wrong number of arguments
   - Missing imports or dependencies
   - Complex semantic errors

### Error Analysis Visualization

```python
error_types = {
    'empty_output': [],
    'incomplete_code': [],
    'missing_parentheses': [],
    # ... etc
}

for prediction, reference in zip(predictions, references):
    if prediction == reference:
        continue  # Skip correct predictions
    
    # Categorize error
    if len(prediction) == 0:
        error_types['empty_output'].append((prediction, reference))
    elif len(prediction) < len(reference) / 2:
        error_types['incomplete_code'].append((prediction, reference))
    elif '(' not in prediction and '(' in reference:
        error_types['missing_parentheses'].append((prediction, reference))
    # ... etc

# Print statistics
for category, examples in error_types.items():
    percentage = len(examples) / total_errors * 100
    print(f"{category}: {len(examples)} ({percentage:.1f}%)")
```

---

## Attention Visualization

### What Are Attention Weights?

For each generated token at position $t$, attention mechanism computes weights over all encoder outputs:

$$\alpha_t = \text{softmax}(e_t)$$

Where $e_t$ is the "energy" or alignment score between decoder state and each encoder position.

### Visualization as Heatmap

- **X-axis**: Source tokens (docstring)
- **Y-axis**: Target tokens (generated code)  
- **Color Intensity**: Attention weight (0 = no attention, 1 = full attention)

**Interpretation**:
- Each row shows which input tokens are attended to when generating each output token
- Words with high attention (bright colors) are most influential
- Should show semantic alignment (e.g., "maximum" attends to "max" function)

### Example Analysis

```
Docstring: "return the maximum value"
Generated Code: "return max ( values )"

Attention Visualization:
                return  the   maximum  value
    return     0.8     0.1   0.05      0.05
    max        0.2     0.3   0.4       0.1
    (          0.3     0.2   0.2       0.3
    values     0.1     0.15  0.25      0.5
```

Interpretation:
- "return" token attends strongly to "return" (0.8)
- "max" token attends most to "maximum" (0.4) - semantic connection!
- "values" token attends to "value" (0.5) - word matching
- This shows the model learns meaningful alignments

---

## Reproducibility

### Design for Reproducibility

1. **Fixed Random Seed**: SEED=42 set in both notebooks
   ```python
   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   torch.backends.cudnn.deterministic = True
   ```

2. **Identical Dataset Indexing**:
   - Both notebooks use `.select()` to pick same examples in same order
   - Same train/val/test split indices

3. **Configuration Persistence**:
   - CONFIG saved to JSON in models/ directory
   - Analytics notebook loads this same CONFIG
   - Ensures consistent settings across executions

4. **Model Checkpoints**:
   - Save best model state_dict with:
     - Weights and biases
     - Metadata (epoch, loss, config)
   - Load with identical architecture to restore exact weights

5. **Tokenizer Persistence**:
   - Save tokenizers as JSON (human-readable, version-agnostic)
   - Load on analytics machine to decode identical token sequences

### Workflow for Reproducible Analysis

```
1. Execute rnn-seq2seq.ipynb on Google Colab:
   ├── Trains all three models (GPU-accelerated)
   ├── Saves best checkpoints to models/
   ├── Saves training history (pickled)
   ├── Saves tokenizers (JSON)
   └── Saves configuration (JSON)

2. Download models/ directory to local machine

3. Execute rnn-seq2seq-analytics.ipynb locally:
   ├── Load configuration from JSON
   ├── Load tokenizers from JSON
   ├── Reconstruct dataset (same indices, SEED=42)
   ├── Load trained models from checkpoints
   ├── Run evaluation with identical logic
   └── Generate consistent metrics and visualizations
```

---

## Key Advantages of This Implementation

1. **Separation of Concerns**: Training and analytics decoupled
2. **Reproducibility**: Same results across machines with proper seeds
3. **Interpretability**: Attention visualizations aid debugging
4. **Comprehensive Metrics**: BLEU, token accuracy, exact match, all three models
5. **Length Analysis**: Shows model behavior across input spectrum
6. **Error Analysis**: Categorizes failures for deep understanding
7. **Configuration Flexibility**: Easy to modify hyperparameters centrally
8. **Scalability**: Architecture supports larger vocabularies and datasets

---

## Running the Notebooks

### Google Colab Training (GPU Recommended)

1. Upload dataset preprocessing script
2. Open `rnn-seq2seq.ipynb` in Colab
3. Run cells sequentially with GPU runtime enabled
4. Download models/ directory when training completes

### Local Analytics (CPU OK)

1. Place downloaded models/ directory in workspace
2. Open `rnn-seq2seq-analytics.ipynb` in Jupyter
3. Run cells sequentially
4. View generated plots and metrics

**Estimated Times**:
- Training: 20-30 minutes on GPU (Colab)
- Analytics: 5-10 minutes on CPU (M1 MacBook)

---

## References

- **Seq2Seq**: Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)
- **Attention Mechanism**: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- **BLEU Score**: Papineni et al., "BLEU: A Method for Automatic Evaluation of Machine Translation" (2002)
- **CodeSearchNet**: Husain et al., "CodeSearchNet: Evaluation and Comparison of Neural Code Search Models" (2020)
