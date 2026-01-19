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
            print(f"Loading model {cls._instance.model_id} on {cls._instance.device}...")
            
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(cls._instance.model_id)
            cls._instance.model = AutoModelForCausalLM.from_pretrained(
                cls._instance.model_id,
                torch_dtype=torch.bfloat16 if cls._instance.device != "cpu" else torch.float32,
                device_map=cls._instance.device
            )
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
