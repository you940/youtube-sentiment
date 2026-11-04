from typing import Dict, List
from pydantic import BaseModel, Field


class BatchRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    text: str
    sentiment: int          # -1, 0 ou 1
    label: str              # négatif, neutre, positif
    confidence: float


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    statistics: Dict
    processing_time_ms: float