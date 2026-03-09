from fastapi import FastAPI, HTTPException
from csao.api.schemas import RecommendationRequest, RecommendationResponse
from csao.core.engine import RecommendationEngine
from csao.utils.monitoring import monitor

app = FastAPI(title="CSAO Recommender", version="1.0.0")
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = RecommendationEngine()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        response = engine.get_recommendations(request)
        monitor.log_metrics({"request_count": 1}, context="api_recommend")
        return response
    except Exception as e:
        monitor.log_metrics({"error_count": 1}, context="api_recommend", metadata={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
