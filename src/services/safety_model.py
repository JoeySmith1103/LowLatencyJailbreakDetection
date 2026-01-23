import os
import torch
import hashlib
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from src.services.unsafe_examples import UNSAFE_CATEGORIES, get_all_category_texts


# ============================================================
# Custom Stopping Criteria
# Stop generation immediately when "safe" or "unsafe" is detected
# ============================================================
class SafetyStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that stops when 'safe' or 'unsafe' token is generated.
    
    Why this works:
    - LLaMA Guard outputs "safe" or "unsafe" as the FIRST token
    - No need to wait for full generation (which might include category info like "unsafe\nS1")
    """
    
    def __init__(self, tokenizer, stop_tokens=None):
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens or ["safe", "unsafe"]
        self.stop_token_ids = set()
        for token in self.stop_tokens:
            for variant in [token, token.capitalize()]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    self.stop_token_ids.add(ids[0])
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token in self.stop_token_ids:
                return True
        return False


class LlamaGuardService:
    """
    Two-Layer Jailbreak Detection Service.
    
    Optimization Modes (via OPTIMIZATION_MODE env var):
    - baseline: Pure LLaMA Guard, no optimizations
    - stopping: LLaMA Guard with custom stopping criteria only
    - embedding: Embedding fast path only
    - full: Stopping + Embedding 
    
    Trade-off Control (via EMBEDDING_THRESHOLD env var):
    - Lower threshold (e.g., 0.50): More aggressive blocking, lower latency, but may have false positives
    - Higher threshold (e.g., 0.85): More conservative, higher latency, but more accurate
    - Default: 0.60 (balanced)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaGuardService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.model_id = "meta-llama/Llama-Guard-3-1B"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # ============================================================
        # Optimization Mode Configuration
        # Modes for Ablation Study:
        # - baseline: Pure LLaMA Guard, no optimizations
        # - stopping: LLaMA Guard with stopping criteria only
        # - embedding: Embedding fast path only (no stopping criteria)
        # - full: Stopping + Embedding
        # ============================================================
        self._optimization_mode = os.getenv("OPTIMIZATION_MODE", "full").lower()
        valid_modes = ["baseline", "stopping", "embedding", "full"]
        if self._optimization_mode not in valid_modes:
            print(f"Warning: Invalid mode '{self._optimization_mode}', using 'full'")
            self._optimization_mode = "full"
        
        print(f"Running in {self._optimization_mode.upper()} mode")
        
        # Determine which features are enabled
        self._use_stopping_criteria = self._optimization_mode in ["stopping", "full"]
        self._use_embedding_layer = self._optimization_mode in ["embedding", "full"]
        
        print(f"  - Stopping Criteria: {'ON' if self._use_stopping_criteria else 'OFF'}")
        print(f"  - Embedding Layer:   {'ON' if self._use_embedding_layer else 'OFF'}")
        
        # ============================================================
        # Embedding Threshold Configuration
        # ============================================================
        self._embedding_threshold = float(os.getenv("EMBEDDING_THRESHOLD", "0.60"))
        print(f"  - Embedding Threshold: {self._embedding_threshold}")
        
        # Get Hugging Face token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Hugging Face token not found! Please set HF_TOKEN"
                "in your .env file or environment variables."
            )
        
        # Load models
        print(f"Loading LLaMA Guard on {self.device}...")
        self._load_llama_guard(hf_token)
        
        if self._use_embedding_layer:
            print("Loading embedding model...")
            self._load_embedding_model()
        
        print("All models loaded successfully.")
    
    def _load_llama_guard(self, hf_token):
        """Load LLaMA Guard model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=hf_token
        )
        
        # Only create stopping criteria if enabled
        if self._use_stopping_criteria:
            self.stopping_criteria = StoppingCriteriaList([
                SafetyStoppingCriteria(self.tokenizer)
            ])
        else:
            self.stopping_criteria = None

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=hf_token,
                torch_dtype=torch.float32,
            ).to(self.device)
    
    def _load_embedding_model(self):
        """
        Load embedding model (all-MiniLM-L6-v2) and pre-compute category embeddings.
        """
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for all unsafe categories
        category_texts = get_all_category_texts()
        self._category_ids = [cat_id for cat_id, _ in category_texts]
        self._category_texts = [text for _, text in category_texts]
        
        self._category_embeddings = self.embedding_model.encode(
            self._category_texts, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        print(f"  Pre-computed {len(self._category_texts)} category embeddings")

    def _check_embedding_similarity(self, text: str) -> tuple[bool, float, str, int]:
        """
        Check if text is similar to any known unsafe category.
        
        Returns: (is_unsafe, max_similarity, matched_category_id, matched_index)
        """
        # Encode query (normalized for cosine similarity)
        query_embedding = self.embedding_model.encode(
            [text], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarities (since both are normalized, dot product = cosine)
        similarities = np.dot(self._category_embeddings, query_embedding)
        
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])
        matched_category = self._category_ids[max_idx]
        
        is_unsafe = max_similarity >= self._embedding_threshold
        
        return is_unsafe, max_similarity, matched_category, max_idx

    def _llama_guard_inference(self, text: str) -> tuple[str, int]:
        """
        Run LLaMA Guard inference.
        
        Returns: (label, tokens_generated)
        """
        chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": text}]
            },
        ]
        
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        
        # Use stopping criteria if enabled
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 10,
            "pad_token_id": 0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.stopping_criteria:
            generate_kwargs["stopping_criteria"] = self.stopping_criteria
        
        output = self.model.generate(**generate_kwargs)
        
        prompt_len = input_ids.shape[1]
        generated_tokens = output[0][prompt_len:]
        tokens_generated = len(generated_tokens)
        
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        first_line = result.split("\n")[0].lower().strip()
        
        if "unsafe" in first_line:
            return "unsafe", tokens_generated
        return "safe", tokens_generated

    def predict(self, text: str) -> tuple[str, str]:
        """
        Two-layer prediction.
        
        Returns: (label, layer) where layer is "embedding" or "llm"
        """
        # ============================================================
        # Layer 1: Embedding Fast Path (if enabled)
        # ============================================================
        if self._use_embedding_layer:
            is_unsafe, similarity, category, _ = self._check_embedding_similarity(text)
            if is_unsafe:
                return "unsafe", "embedding"
        
        # ============================================================
        # Layer 2: LLaMA Guard inference
        # ============================================================
        label, _ = self._llama_guard_inference(text)
        return label, "llm"

    def predict_detailed(self, text: str) -> dict:
        """
        Detailed prediction with debug information.
        Useful for analysis and understanding model behavior.
        
        Returns dict with:
        - text: input text
        - final_label: "safe" or "unsafe"
        - layer_used: "embedding" or "llm"
        - embedding_similarity: float (if embedding layer enabled)
        - matched_category: category ID (if embedding layer enabled)
        - matched_text: the text that matched (if embedding layer enabled)
        - tokens_generated: number of tokens generated by LLM (if LLM was used)
        """
        result = {
            "text": text,
            "final_label": None,
            "layer_used": None,
            "embedding_similarity": None,
            "matched_category": None,
            "matched_text": None,
            "tokens_generated": None,
            "threshold": self._embedding_threshold,
        }
        
        # Layer 1: Embedding
        if self._use_embedding_layer:
            is_unsafe, similarity, category, matched_idx = self._check_embedding_similarity(text)
            result["embedding_similarity"] = similarity
            result["matched_category"] = category
            result["matched_text"] = self._category_texts[matched_idx]
            
            if is_unsafe:
                result["final_label"] = "unsafe"
                result["layer_used"] = "embedding"
                return result
        
        # Layer 2: LLM
        label, tokens_generated = self._llama_guard_inference(text)
        result["final_label"] = label
        result["layer_used"] = "llm"
        result["tokens_generated"] = tokens_generated
        
        return result
    
    def get_mode_info(self) -> dict:
        """Return current mode configuration for reporting."""
        return {
            "optimization_mode": self._optimization_mode,
            "use_stopping_criteria": self._use_stopping_criteria,
            "use_embedding_fast_path": self._use_embedding_layer,
            "embedding_threshold": self._embedding_threshold if self._use_embedding_layer else 0.0,
        }


def reset_service():
    LlamaGuardService._instance = None
