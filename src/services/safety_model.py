import os
import torch
import hashlib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from src.services.unsafe_examples import UNSAFE_CATEGORIES, get_all_category_texts


# ============================================================
# Layer 3 Helper: Custom Stopping Criteria
# Stop generation immediately when "safe" or "unsafe" is detected
# ============================================================
class SafetyStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria that stops when 'safe' or 'unsafe' token is generated."""
    
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
        # Optimization Mode: "baseline" or "optimized"
        # Set via environment variable OPTIMIZATION_MODE
        # ============================================================
        self._optimization_mode = os.getenv("OPTIMIZATION_MODE", "optimized").lower()
        print(f"Running in {self._optimization_mode.upper()} mode")
        
        # ============================================================
        # Layer 1: Exact Match Cache (only in optimized mode)
        # ============================================================
        self._cache = {}
        self._cache_max_size = 10 if self._optimization_mode == "optimized" else 0
        
        # ============================================================
        # Layer 2: Embedding Similarity Configuration (only in optimized mode)
        # ============================================================
        self._embedding_threshold = 0.70
        self._use_embedding_layer = (self._optimization_mode == "optimized")
        
        # Get Hugging Face token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise ValueError(
                "Hugging Face token not found! Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
                "in your .env file or environment variables."
            )
        
        # Load models
        print(f"Loading LLaMA Guard on {self.device}...")
        self._load_llama_guard(hf_token)
        
        # Only load embedding model in optimized mode
        if self._use_embedding_layer:
            print("Loading embedding model...")
            self._load_embedding_model()
        
        print("All models loaded successfully.")
    
    def _load_llama_guard(self, hf_token):
        """Load Layer 3: LLaMA Guard model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=hf_token
        )
        
        self.stopping_criteria = StoppingCriteriaList([
            SafetyStoppingCriteria(self.tokenizer)
        ])

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
        """Load Layer 2: Embedding model and pre-compute category embeddings."""
        # Use a small, fast embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for all unsafe category texts
        category_texts = get_all_category_texts()
        self._category_ids = [cat_id for cat_id, _ in category_texts]
        texts = [text for _, text in category_texts]
        
        # Compute embeddings once at startup
        self._category_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        print(f"  Pre-computed {len(texts)} category embeddings")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key from text using SHA256 hash."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _check_embedding_similarity(self, text: str) -> tuple[bool, float]:
        """
        Layer 2: Check if text is similar to any known unsafe category.
        Returns (is_unsafe, max_similarity)
        """
        # Encode the query
        query_embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
        
        # Compute cosine similarity with all category embeddings
        similarities = np.dot(self._category_embeddings, query_embedding) / (
            np.linalg.norm(self._category_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        max_similarity = float(np.max(similarities))
        
        # If similarity exceeds threshold, classify as unsafe
        if max_similarity >= self._embedding_threshold:
            return True, max_similarity
        
        return False, max_similarity

    def _llama_guard_inference(self, text: str) -> str:
        """Layer 3: Run LLaMA Guard inference."""
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
            stopping_criteria=self.stopping_criteria,
        )
        
        prompt_len = input_ids.shape[1]
        generated_tokens = output[0][prompt_len:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        first_line = result.split("\n")[0].lower().strip()
        
        if "unsafe" in first_line:
            return "unsafe"
        return "safe"

    def predict(self, text: str) -> tuple[str, str]:
        """
        Multi-tier prediction:
        - Baseline mode: Only LLaMA Guard (no optimization)
        - Optimized mode: Cache → Embedding → LLaMA Guard
        
        Returns: (label, layer) where layer is "cache", "embedding", or "llm"
        """
        
        # ============================================================
        # Baseline Mode: Skip all optimizations, go directly to LLM
        # ============================================================
        if self._optimization_mode == "baseline":
            label = self._llama_guard_inference(text)
            return label, "llm"
        
        # ============================================================
        # Optimized Mode: Multi-tier prediction
        # ============================================================
        
        # Layer 1: Check exact match cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key], "cache"
        
        # Layer 2: Embedding similarity check
        if self._use_embedding_layer:
            is_unsafe, similarity = self._check_embedding_similarity(text)
            if is_unsafe:
                label = "unsafe"
                if len(self._cache) < self._cache_max_size:
                    self._cache[cache_key] = label
                return label, "embedding"
        
        # Layer 3: LLaMA Guard (final fallback)
        label = self._llama_guard_inference(text)
        
        # Cache the result
        if len(self._cache) < self._cache_max_size:
            self._cache[cache_key] = label
            
        return label, "llm"

