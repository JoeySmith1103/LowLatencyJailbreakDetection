from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from src.services.safety_model import LlamaGuardService

# Load environment variables from .env file
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

app = FastAPI(title="LLM Safety API Service", version="2.0")

# Initialize model on startup
service = LlamaGuardService()


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    label: str
    layer: str  # "embedding" or "llm"


class DetailedDetectResponse(BaseModel):
    """Detailed response with debug information."""
    text: str
    label: str
    layer: str
    embedding_similarity: Optional[float] = None
    matched_category: Optional[str] = None
    matched_text: Optional[str] = None
    tokens_generated: Optional[int] = None
    threshold: Optional[float] = None


@app.post("/v1/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Main detection endpoint.
    
    Returns:
    - label: "safe" or "unsafe"
    - layer: "embedding" (fast path) or "llm" (full inference)
    """
    try:
        label, layer = service.predict(request.text)
        return DetectResponse(label=label, layer=layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/detect/detailed", response_model=DetailedDetectResponse)
async def detect_detailed(request: DetectRequest):
    """
    Detailed detection endpoint for analysis and debugging.
    
    Returns full information about the prediction process.
    """
    try:
        result = service.predict_detailed(request.text)
        return DetailedDetectResponse(
            text=result["text"],
            label=result["final_label"],
            layer=result["layer_used"],
            embedding_similarity=result.get("embedding_similarity"),
            matched_category=result.get("matched_category"),
            matched_text=result.get("matched_text"),
            tokens_generated=result.get("tokens_generated"),
            threshold=result.get("threshold"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint with mode information."""
    return {
        "status": "ok",
        "mode_info": service.get_mode_info()
    }
