#!/bin/bash
# Example script to run judgments using a local model

# Enable 4-bit quantization for lower memory usage (comment out if not needed)
export USE_4BIT_QUANTIZATION=true

# Run judgment with local model
python gen_judgment.py \
  --setting-file "config/judge_config_multipattern.yaml" \
  --endpoint-file "config/api_config_local.yaml" \
  --logprob_judgments

echo "Judgment completed using local models"