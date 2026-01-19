from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from src.services.safety_model import LlamaGuardService

app = FastAPI(title="LLM Safety API Service", version="1.0")

# Initialize model on startup
service = LlamaGuardService()

class DetectRequest(BaseModel):
    text: str

class DetectResponse(BaseModel):
    label: str

@app.post("/v1/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    try:
        label = service.predict(request.text)
        return DetectResponse(label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
