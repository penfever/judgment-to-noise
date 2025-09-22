#!/usr/bin/env python3
"""
Demo script for running a local Hugging Face model for inference.
This shows how to use the new chat_completion_huggingface_local function.

Example usage:
    # Run without logprobs
    python local_model_demo.py --model "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Run with logprobs
    python local_model_demo.py --model "mistralai/Mistral-7B-Instruct-v0.2" --logprobs
    
    # Enable 4-bit quantization for lower memory usage
    USE_4BIT_QUANTIZATION=true python local_model_demo.py --model "meta-llama/Llama-3.1-8B-Instruct"
"""

import argparse
import json
import os
import sys
from utils import chat_completion_huggingface_local

def main():
    parser = argparse.ArgumentParser(description="Demo for local Hugging Face model inference")
    parser.add_argument("--model", required=True, help="HuggingFace model ID to load (e.g., 'meta-llama/Llama-3.1-8B-Instruct')")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--logprobs", action="store_true", help="Return logprobs with generations")
    args = parser.parse_args()
    
    # Example conversation with a system prompt and user prompt
    conversation = [
        {"role": "system", "content": "You are a helpful, accurate, and concise assistant."},
        {"role": "user", "content": "Explain the concept of gradient descent in machine learning. Keep it brief."}
    ]
    
    print(f"\n🔄 Running inference with model: {args.model}")
    print(f"📝 Using temperature: {args.temperature}, max_tokens: {args.max_tokens}, logprobs: {args.logprobs}")
    
    # Run the model locally
    output = chat_completion_huggingface_local(
        args.model,
        conversation,
        args.temperature,
        args.max_tokens,
        args.logprobs
    )
    
    print("\n✅ Inference completed!")
    
    # Display the results
    if args.logprobs:
        print("\n📊 Model Output (with logprobs):")
        print(f"Generated Text: {output['content']}")
        print(f"\nToken Count: {len(output['logprobs']['content'])}")
        
        # Print first 5 tokens with their logprobs
        print("\nFirst 5 tokens with logprobs:")
        for i, token_info in enumerate(output['logprobs']['content'][:5]):
            print(f"  Token {i+1}: '{token_info['text']}' (logprob: {token_info['logprob']:.4f})")
    else:
        print("\n📊 Model Output:")
        print(output)
    
    print("\n🏁 Demo completed!")

if __name__ == "__main__":
    main()