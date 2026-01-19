import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    model_id = "meta-llama/Llama-Guard-3-1B"
    print(f"Downloading {model_id}...")
    
    # Download and cache locally
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
