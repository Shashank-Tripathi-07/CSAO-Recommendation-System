from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Unique ID for the user")
    session_id: str = Field(..., description="Active session ID for tracing")
    cart_items: List[str] = Field(default_factory=list, description="List of items currently in the cart")
    context: Dict[str, str] = Field(default_factory=dict, description="Metadata: lat, long, cuisine, area")

class RecommendationItem(BaseModel):
    item_id: str
    score: float
    metadata: Dict[str, str] = Field(default_factory=dict)

class RecommendationResponse(BaseModel):
    request_id: str
    recommendations: List[RecommendationItem]
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
