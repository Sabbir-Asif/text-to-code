# Text to Python Code Generation using Seq2Seq Models

## Project Overview
This project implements sequence to sequence learning models for generating Python code from natural language docstrings. Three architectures are implemented and compared: a vanilla RNN based Seq2Seq model, an LSTM based Seq2Seq model, and an LSTM with attention mechanism. The dataset used is the Python subset of the CodeSearchNet dataset, where each sample contains a docstring as input and its corresponding Python function as output.

## Objective
The goals of this project are to study the limitations of simple RNN models, evaluate improvements from LSTM units, and analyze how attention mechanisms improve alignment between input text and generated code. The models are evaluated quantitatively using BLEU score and qualitatively by inspecting generated outputs.

## Dataset
**Dataset name:** CodeSearchNet Python dataset  
**Input:** English docstring  
**Output:** Python function code

### Dataset split used in experiments
- Training samples: 7000
- Validation samples: approximately 700
- Test samples: approximately 700

### Sequence length constraints
- Maximum input length: 50 tokens
- Maximum output length: 80 tokens

## Training Configuration
```
{
  "TRAIN_SIZE": 10000,
  "VAL_SIZE": 1500,
  "TEST_SIZE": 1500,
  "MAX_DOCSTRING_LEN": 50,
  "MAX_CODE_LEN": 80,
  "EMBEDDING_DIM": 256,
  "HIDDEN_DIM": 256,
  "NUM_LAYERS": 2,
  "DROPOUT": 0.3,
  "BIDIRECTIONAL": true,
  "BATCH_SIZE": 64,
  "EPOCHS": 20,
  "LEARNING_RATE": 0.001,
  "TEACHER_FORCING_RATIO": 0.5,
  "VOCAB_SIZE": 5000,
  "GRADIENT_CLIP": 1.0,
  "WARMUP_STEPS": 1000,
  "EARLY_STOPPING_PATIENCE": 3,
  "BEAM_SIZE": 3,
  "SCHEDULED_SAMPLING": true
}
```
## Models Implemented

### Model 1: Vanilla RNN Seq2Seq
Encoder and decoder both use standard RNN cells. The model uses a fixed length context vector passed from encoder to decoder. This model suffers from vanishing gradient and limited ability to represent long sequences.

### Model 2: LSTM Seq2Seq
Encoder and decoder both use LSTM units. The model still uses a fixed length context vector but benefits from gating mechanisms that help preserve long term dependencies and reduce gradient issues.

### Model 3: LSTM with Attention
The encoder uses an LSTM, optionally bidirectional, and the decoder uses an LSTM with Bahdanau attention. At each decoding step the model attends over encoder hidden states, removing the fixed context bottleneck and improving alignment between docstring tokens and generated code tokens.

## Results

### BLEU Score Comparison
- Vanilla RNN Seq2Seq: 0.252
- LSTM Seq2Seq: 0.001847
- LSTM with attention greedy decoding: 2.194625
- LSTM with attention beam search: 2.288397

### Best Performing Model
The best result is obtained using the LSTM with attention model with beam search decoding, achieving a BLEU score of 2.288397

### Qualitative Observations
- Vanilla RNN outputs are mostly noisy and contain repeated or malformed tokens.
- LSTM outputs show slightly better structure but still contain syntax errors and missing components.
- Attention based model produces more coherent structure, better alignment with keywords from the docstring, and improved placement of tokens such as return, max, and loop constructs.

## Error Analysis

### Common error types observed in generated code
- Syntax errors such as missing parentheses and incorrect indentation
- Structural errors such as missing loops, missing return statements, or incomplete function definitions
- Token level errors such as repetition, incorrect variable names, and punctuation mistakes

### Performance versus input length
All models perform better on short docstrings. Performance degrades as input length increases. The attention model handles longer sequences better than both RNN and LSTM models, but still produces imperfect outputs due to task complexity.

## Conclusion
Vanilla RNN Seq2Seq models are not suitable for code generation tasks due to their inability to capture long term dependencies. LSTM improves stability but still suffers from the fixed context bottleneck. The attention mechanism significantly improves alignment between input and output and produces the best overall performance. However, code generation remains a challenging task and more advanced architectures such as transformer based models are expected to perform better.

## Project Structure
```
project
├── models
│   ├── best_model_attention.pt
│   ├── best_model_lstm.pt
│   └── tokenizer.json
├── notebooks
│   ├── vanilla_rnn_v2.ipynb
│   ├── lstm_rnn.ipynb
│   └── lstm_attention.ipynb
├── instructions
├── venv
└── gitignore
```

## How to Run

### Install dependencies
```
pip install torch transformers datasets nltk sentencepiece
```

### Train a model
```
python train.py
```

### Evaluate model
```
python evaluate.py
```

## Author
**Sabbir Hosen**  
BSSE 1333