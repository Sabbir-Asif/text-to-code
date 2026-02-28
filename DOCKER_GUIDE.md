# Seq2Seq Code Generation - Docker Setup

Complete Docker configuration for training and evaluating RNN, LSTM, and LSTM+Attention seq2seq models for code generation.

## 📋 Prerequisites

- **Docker**: [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Docker Compose**: Usually included with Docker Desktop
- **GPU Support** (optional): [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker)
- **Disk Space**: ~20GB recommended (models, datasets, and dependencies)

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

**Start Jupyter Lab:**
```bash
docker-compose up seq2seq-notebook
```

Then open your browser to: `http://localhost:8888`

**Interactive bash shell:**
```bash
docker-compose --profile shell up -d seq2seq-bash
docker-compose exec seq2seq-bash bash
```

### Option 2: Manual Docker Commands

**Build the image:**
```bash
docker build -t seq2seq:latest .
```

**Run Jupyter Lab:**
```bash
docker run --rm -it \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  seq2seq:latest
```

**Run bash shell:**
```bash
docker run --rm -it \
  -v $(pwd):/workspace \
  seq2seq:latest \
  /bin/bash
```

## 🔧 Configuration

### GPU Support

To enable GPU acceleration, ensure you have:

1. **NVIDIA Docker runtime installed**
   ```bash
   docker run --gpus all --rm nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
   ```

2. **Update docker-compose.yml** (already configured for `CUDA_VISIBLE_DEVICES=0`)
   
   Change `CUDA_VISIBLE_DEVICES=0` to use different GPUs:
   - `0` - First GPU
   - `0,1` - First and second GPU
   - `all` - All available GPUs

3. **Run with GPU:**
   ```bash
   docker-compose up seq2seq-notebook  # Already enables GPU
   ```

### Custom Ports

If port 8888 is already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8889:8888"  # Maps host port 8889 to container port 8888
```

### Environment Variables

Edit `docker-compose.yml` to add custom variables:
```yaml
environment:
  - JUPYTER_ENABLE_LAB=yes
  - CUDA_VISIBLE_DEVICES=0
  - MY_CUSTOM_VAR=value
```

## 📁 Project Structure

```
.
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Orchestration config
├── requirements.txt           # Python dependencies
├── .dockerignore              # Files to exclude from build
├── rnn-seq2seq.ipynb         # Training notebook
├── rnn-seq2seq-analytics.ipynb # Evaluation & visualization
├── models/                    # Trained model checkpoints
├── analytics/                 # Results & visualizations
└── instructions/              # Project documentation
```

## 🎯 Workflow

### Inside the Container

After launching Jupyter Lab, open a terminal in JupyterLab or your bash shell:

**1. Train Models (one-time, ~2-3 hours):**
```bash
# Open rnn-seq2seq.ipynb and run all cells sequentially
# Or use papermill from CLI:
jupyter nbconvert --to notebook --execute rnn-seq2seq.ipynb
```

**2. Run Analytics (after training completes):**
```bash
# Open rnn-seq2seq-analytics.ipynb and run all cells
# Or use papermill:
jupyter nbconvert --to notebook --execute rnn-seq2seq-analytics.ipynb
```

**3. View Results:**
```bash
# Outputs saved to analytics/ folder:
# - model_comparison.csv (metrics)
# - training_curves.png (loss plots)
# - metrics_comparison.png (BLEU, accuracy, etc.)
# - attention_example_*.png (attention visualizations)
# - error_distribution.png (error analysis)
# - length_performance.png (length vs accuracy)
```

## 📊 Model Details

### Architecture Options
- **Vanilla RNN**: Bidirectional encoder-decoder
- **LSTM**: 2-layer bidirectional with layer normalization
- **LSTM+Attention**: Attention mechanism for better long sequences

### Training Config (inside container)
- **Epochs**: 30
- **Batch Size**: 128
- **Learning Rate**: 0.0002 (with scheduler)
- **Dropout**: 0.6
- **Dataset**: CodeSearchNet Python (~13,000 samples)

### Expected Results
After full training (~3 hours on GPU):
- LSTM+Attention BLEU: 0.015-0.025
- Token Accuracy: 5-15%
- Training loss converges around epoch 10-15

## 📦 Dependency Management

### Update Dependencies

If you need to add/remove packages:

1. **Edit requirements.txt:**
   ```
   echo "new-package==1.0.0" >> requirements.txt
   ```

2. **Rebuild image:**
   ```bash
   docker-compose down
   docker-compose build --no-cache seq2seq-notebook
   docker-compose up seq2seq-notebook
   ```

### Installed Packages
- **Deep Learning**: torch, torchvision, torchaudio
- **Data**: numpy, pandas, datasets
- **Visualization**: matplotlib, seaborn
- **Metrics**: sacrebleu, scikit-learn
- **Notebooks**: jupyter, jupyterlab, ipython

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find service using port 8888
lsof -i :8888

# Kill process or use different port in docker-compose.yml
```

### Out of Memory
```bash
# Reduce batch size in notebook config
# Or increase Docker memory limit in Docker Desktop settings
```

### GPU Not Detected
```bash
# Inside container, check GPU availability:
python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure NVIDIA Docker is installed
```

### Slow Build Time
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build --no-cache
```

### Permission Denied (volumes)
```bash
# Run Docker with user privileges
docker-compose run --user $(id -u):$(id -g) seq2seq-notebook
```

## 🔄 Data Persistence

### Volume Mounts

The docker-compose.yml automatically mounts:
- **./** → `/workspace` - Your entire project
- **./models** → `/workspace/models` - Trained checkpoints
- **./analytics** → `/workspace/analytics` - Results

Changes in the container are automatically synced to your local machine.

### Backup Models

```bash
# Download trained models locally
docker cp seq2seq-training-analytics:/workspace/models ./models_backup

# Or use compose service:
docker-compose run seq2seq-notebook cp -r models /backup/
```

## 🚀 Production Deployment

For production use, create a minimal inference image:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "run_inference.py"]
```

## 📝 Useful Commands

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f seq2seq-notebook

# Stop all services
docker-compose down

# Remove all containers and images
docker-compose down -v
docker rmi seq2seq:latest

# Rebuild from scratch
docker-compose build --no-cache

# Interactive Python shell inside container
docker-compose run seq2seq-notebook python

# Run custom command
docker-compose run seq2seq-notebook pip list
```

## 🎓 Educational Use

This setup is ideal for:
- Learning seq2seq architectures
- Understanding attention mechanisms
- Experimenting with code generation
- Reproducible research

Run in JupyterLab for interactive exploration and visualization!

## 📄 License

Same as parent project. See LICENSE file.

## ❓ FAQ

**Q: Can I use CPU only?**
A: Yes, but training will be slow. Models will auto-detect CUDA availability.

**Q: How large are trained models?**
A: ~50-100MB each (3 models total ≈ 300MB).

**Q: Can I push custom models to the cloud?**
A: Yes! Models in `/workspace/models` persist with volume mounts.

**Q: How do I edit notebooks outside the container?**
A: Edit locally; changes sync automatically via volume mount.

**Q: Persistent Jupyter token?**
A: Generate one during first run or set in docker-compose.yml.

---

For more details on the project itself, see [readme.md](readme.md) and [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)
