from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security.api_key import APIKeyHeader
from fastapi.encoders import jsonable_encoder
from csao.api.schemas import RecommendationRequest, RecommendationResponse, ErrorResponse
from csao.core.engine import RecommendationEngine
from csao.utils.monitoring import monitor
from csao.utils.config import settings
from csao.utils.exceptions import CSAOError, AuthenticationError

app = FastAPI(title="CSAO Recommender", version="1.0.0")
engine = None

API_KEY = "csao-secret-key-123" # In production, load from settings/env
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Exception Handlers
@app.exception_handler(CSAOError)
async def csao_exception_handler(request: Request, exc: CSAOError):
    return JSONResponse(
        status_code=400 if exc.error_code == "VALIDATION_ERROR" else 500,
        content=jsonable_encoder(ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details
        ))
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Invalid request body",
            details={"errors": exc.errors()}
        ))
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    monitor.log_metrics({"unhandled_exception": 1}, context="api_global", metadata={"error": str(exc)})
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__}
        ))
    )

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise AuthenticationError("Could not validate credentials")

@app.on_event("startup")
def startup_event():
    global engine
    try:
        engine = RecommendationEngine()
    except Exception as e:
        monitor.log_metrics({"startup_failure": 1}, context="api_startup", metadata={"error": str(e)})
        # We still start the app, but engine will be None or partially loaded
        # The recommend endpoint will handle this

@app.get("/health")
def health_check():
    if not engine or not engine.bst_lgb:
        return JSONResponse(status_code=503, content={"status": "degraded", "reason": "model_not_loaded"})
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest, 
    api_key: str = Depends(get_api_key)
):
    if not engine:
         raise CSAOError("Recommendation engine not initialized", error_code="ENGINE_NOT_READY")
    try:
        response = engine.get_recommendations(request)
        monitor.log_metrics({"request_count": 1}, context="api_recommend")
        return response
    except CSAOError as e:
        raise e
    except Exception as e:
        monitor.log_metrics({"error_count": 1}, context="api_recommend", metadata={"error": str(e)})
        raise CSAOError(f"Engine failed: {str(e)}", error_code="ENGINE_FAILURE")
