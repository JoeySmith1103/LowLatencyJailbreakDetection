from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os
from src.services.safety_model import LlamaGuardService, reset_service

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


class ConfigRequest(BaseModel):
    """Configuration update request."""
    optimization_mode: Optional[str] = None  # baseline, stopping, embedding, full
    embedding_threshold: Optional[float] = None  # 0.0-1.0


class ConfigResponse(BaseModel):
    """Current configuration."""
    optimization_mode: str
    embedding_threshold: float
    use_stopping_criteria: bool
    use_embedding_fast_path: bool


@app.post("/v1/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Main detection endpoint (follows spec).
    
    Returns:
    - label: "safe" or "unsafe"
    """
    try:
        label, _ = service.predict(request.text)
        return DetectResponse(label=label)
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


@app.get("/admin/config", response_model=ConfigResponse)
def get_config():
    """Get current configuration."""
    mode_info = service.get_mode_info()
    return ConfigResponse(
        optimization_mode=mode_info["optimization_mode"],
        embedding_threshold=mode_info["embedding_threshold"],
        use_stopping_criteria=mode_info["use_stopping_criteria"],
        use_embedding_fast_path=mode_info["use_embedding_fast_path"]
    )


@app.post("/admin/config", response_model=ConfigResponse)
def update_config(config: ConfigRequest):
    """
    Update service configuration and reinitialize.
    
    This allows experiments to dynamically change optimization settings.
    """
    global service
    
    try:
        # Update environment variables
        if config.optimization_mode:
            os.environ["OPTIMIZATION_MODE"] = config.optimization_mode
        if config.embedding_threshold is not None:
            os.environ["EMBEDDING_THRESHOLD"] = str(config.embedding_threshold)
        
        # Reset and reinitialize service
        reset_service()
        service = LlamaGuardService()
        
        # Return new configuration
        return get_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")
