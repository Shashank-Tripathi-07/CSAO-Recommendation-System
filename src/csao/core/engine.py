import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
import pickle
import os
from typing import List, Dict, Any
from csao.utils.config import settings
from csao.utils.monitoring import monitor
from csao.api.schemas import RecommendationRequest, RecommendationResponse, RecommendationItem

class RecommendationEngine:
    def __init__(self):
        self._load_models()
        
    def _load_models(self):
        try:
            self.bst_lgb = lgb.Booster(model_file='model_lgb.txt')
            self.bst_xgb = xgb.Booster(model_file='model_xgb.json')
            self.bst_cb = CatBoostRanker()
            self.bst_cb.load_model('model_cb.bin')
            with open('pipeline.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            monitor.log_metrics({"models_loaded": 1}, context="engine_init")
        except Exception as e:
            monitor.log_metrics({"model_load_error": 1}, context="engine_init", metadata={"error": str(e)})
            
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        # 1. Stage 1: Retrieval (Mock/Vector)
        candidates = ["Raita", "Salan", "Gulab Jamun", "Coke", "Thumbs Up"]
        
        # 2. Stage 2: Ranking
        # Create scoring dataframe
        df = pd.DataFrame([{
            'session_id': 1,
            'cart_items': '|'.join(request.cart_items),
            'candidate_item': c,
            'cuisine': request.context.cuisine,
            'area': request.context.area,
            'candidate_category': 'side', # simplified
            'meal_time': 'dinner',
            'rating': 4.5,
            'cart_size': len(request.cart_items),
            'hour_of_day': 19
        } for c in candidates])
        
        X = self.pipeline.transform(df)
        cols_to_drop = ['session_id', 'label', 'timestamp', 'restaurant_id', 'cuisine', 'area', 'cart_items', 
                        'candidate_item', 'candidate_category', 'meal_time', 'cost_tier', 'is_complement']
        features = [c for c in X.columns if c not in cols_to_drop]
        
        scores_lgb = self.bst_lgb.predict(X[features])
        scores_xgb = self.bst_xgb.predict(xgb.DMatrix(X[features]))
        scores_cb = self.bst_cb.predict(X[features])
        
        # Simple blend
        final_scores = (scores_lgb + scores_xgb + scores_cb) / 3.0
        df['score'] = final_scores
        
        results = df.sort_values('score', ascending=False).head(5)
        
        items = [RecommendationItem(item_id=row['candidate_item'], score=row['score'], metadata={}) for _, row in results.iterrows()]
        
        return RecommendationResponse(
            request_id="req_123",
            recommendations=items,
            model_version="ensemble_v1"
        )
