# Using Local Models for LLM Judgment

This guide explains how to run LLM judgment using local Hugging Face models instead of relying exclusively on APIs.

## Installation

First, install the required dependencies:

```bash
# Install base requirements
pip install -r requirements.txt

# Install dependencies for local model inference 
pip install torch transformers accelerate sentencepiece protobuf safetensors

# Optional: Install for better performance (recommended for larger models)
pip install bitsandbytes optimum
```

For CUDA acceleration (NVIDIA GPUs):
```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: Install for improved performance
pip install flash-attn  # For faster attention computation
```

For ROCm acceleration (AMD GPUs):
```bash
# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

## Configuration

To use local models, you need to update your API configuration file to include models with the "huggingface_local" API type. See `config/api_config_local.yaml` for examples.

Example configuration for a local model:
```yaml
mistral/Mistral-7B-Instruct-v0.2-local:
    model_name: mistralai/Mistral-7B-Instruct-v0.2
    endpoints: null
    api_type: huggingface_local
    parallel: 1  # Keep this at 1 to avoid OOM errors
```

Then, update your judge_config.yaml file to use the local model as the judge:
```yaml
judge_model: mistral/Mistral-7B-Instruct-v0.2-local
```

## Running the Demo

Try the demo script first to ensure everything is working correctly:

```bash
# Basic usage
python local_model_demo.py --model "mistralai/Mistral-7B-Instruct-v0.2"

# With logprobs
python local_model_demo.py --model "mistralai/Mistral-7B-Instruct-v0.2" --logprobs

# Use 4-bit quantization for lower memory usage
USE_4BIT_QUANTIZATION=true python local_model_demo.py --model "meta-llama/Llama-3.1-8B-Instruct"
```

## Running Judgments with Local Models

Use the provided script to run judgments with local models:

```bash
./run_local_judgment.sh
```

Or run the command manually:

```bash
# Enable 4-bit quantization if needed (for larger models)
export USE_4BIT_QUANTIZATION=true

# Run judgment
python gen_judgment.py \
  --setting-file "config/judge_config_multipattern.yaml" \
  --endpoint-file "config/api_config_local.yaml" \
  --logprob_judgments
```

## Tips for Using Local Models

1. **Memory Requirements**:
   - 7B models: At least 16GB VRAM (8GB with 4-bit quantization)
   - 13B models: At least 24GB VRAM (14GB with 4-bit quantization)
   - For consumer GPUs, use 4-bit quantization by setting `USE_4BIT_QUANTIZATION=true`

2. **Apple Silicon Considerations**:
   - MPS (Metal Performance Shaders) on Apple Silicon has more limited memory
   - 7B models should work on M1/M2 Macs with 16GB+ RAM
   - 8B-9B models or larger will automatically fall back to CPU mode
   - For Apple Silicon, we recommend sticking with 7B models or smaller

3. **Performance and Caching**:
   - Set `parallel: 1` for local models to avoid out-of-memory errors
   - Models are now cached in memory for the duration of the script run
   - First model load is slow, but subsequent calls reuse the loaded model
   - The system automatically creates a unique cache key for each model configuration
   - This dramatically improves performance for batch inference with the same model
   - Models will fall back to CPU with memory-efficient settings if they don't fit in GPU memory

4. **Model Storage**:
   - Models are downloaded from Hugging Face Hub to your local cache
   - Ensure you have sufficient disk space (typically 5-15GB per model)
   - Disk offloading is used automatically when needed for larger models

5. **Troubleshooting**:
   - If you encounter out-of-memory errors, try:
     - Enable 4-bit quantization: `USE_4BIT_QUANTIZATION=true`
     - Use a smaller model (7B or smaller is recommended)
     - Reduce max_tokens in your config
     - The system will automatically try to fall back to CPU with memory-efficient settings
   - If your model is still failing to load even on CPU:
     - Try a smaller model (3B-7B parameter models work best on consumer hardware)
     - Ensure you have at least 32GB of free disk space for offloading
     - Increase your system's swap space if possible

## Supported Hardware Platforms

- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Apple Silicon (MPS)
- CPU (fallback)