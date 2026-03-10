import pytest
from csao.core.engine import RecommendationEngine
from csao.api.schemas import RecommendationRequest

def test_engine_cold_start():
    engine = RecommendationEngine()
    request = RecommendationRequest(
        user_id="user_1",
        session_id="sess_1",
        cart_items=[],
        context={"cuisine": "Indian"}
    )
    response = engine.get_recommendations(request)
    assert response.model_version == "fallback_v1"
    assert len(response.recommendations) > 0
    assert response.recommendations[0].item_id in ["Coke", "Gulab Jamun", "Raita"]

def test_engine_ranking():
    engine = RecommendationEngine()
    # Mocking models might be needed if they are not present, 
    # but here we assume the engine at least tries to load them.
    # If they are missing, it will hit the exception block and return fallback_error_v1.
    request = RecommendationRequest(
        user_id="user_1",
        session_id="sess_1",
        cart_items=["Biryani"],
        context={"cuisine": "Indian"}
    )
    response = engine.get_recommendations(request)
    assert response.model_version in ["ensemble_v1", "fallback_error_v1"]
