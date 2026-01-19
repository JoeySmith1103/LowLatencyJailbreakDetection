import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class LlamaGuardService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaGuardService, cls).__new__(cls)
            cls._instance.model_id = "meta-llama/Llama-Guard-3-1B"
            cls._instance.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
            # Get Hugging Face token from environment variables
            # Support both HF_TOKEN and HUGGING_FACE_HUB_TOKEN
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not hf_token:
                raise ValueError(
                    "Hugging Face token not found! Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
                    "in your .env file or environment variables."
                )
            
            print(f"Loading model {cls._instance.model_id} on {cls._instance.device}...")
            
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                cls._instance.model_id,
                token=hf_token
            )

            # For GPU, prefer device_map="auto" so HF dispatches weights onto CUDA.
            # For CPU/MPS, load normally and move to the selected device.
            if cls._instance.device == "cuda":
                cls._instance.model = AutoModelForCausalLM.from_pretrained(
                    cls._instance.model_id,
                    token=hf_token,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                cls._instance.model = AutoModelForCausalLM.from_pretrained(
                    cls._instance.model_id,
                    token=hf_token,
                    torch_dtype=torch.float32,
                ).to(cls._instance.device)
            print("Model loaded successfully.")
        return cls._instance

    def predict(self, text: str) -> str:
        # Template for Llama Guard 3
        chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": text}]
            },
        ]
        
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            pad_token_id=0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        prompt_len = input_ids.shape[1]
        generated_tokens = output[0][prompt_len:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"DEBUG: raw_result='{result}'")
        
        # Logic fix: "unsafe" contains "safe", so we must check for "unsafe" first
        first_line = result.split("\n")[0].lower().strip()
        
        if first_line == "unsafe":
            label = "unsafe"
        else:
            label = "safe"
            
        return label
