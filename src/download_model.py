import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_model():
    model_id = "meta-llama/Llama-Guard-3-1B"
    print(f"Downloading {model_id}...")
    
    # Get Hugging Face token from environment variables
    # Support both HF_TOKEN and HUGGING_FACE_HUB_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found! Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
            "in your environment variables."
        )
    
    # Download and cache locally
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
