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

from csao.utils.exceptions import ModelLoadError, InferenceError

class RecommendationEngine:
    def __init__(self):
        self.bst_lgb = None
        self.bst_xgb = None
        self.bst_cb = None
        self.pipeline = None
        self._load_models()
        
    def _load_models(self):
        errors = []
        try:
            if os.path.exists('model_lgb.txt'):
                self.bst_lgb = lgb.Booster(model_file='model_lgb.txt')
            else:
                errors.append("LGBM model file missing")

            if os.path.exists('model_xgb.json'):
                self.bst_xgb = xgb.Booster(model_file='model_xgb.json')
            else:
                errors.append("XGBoost model file missing")

            if os.path.exists('model_cb.bin'):
                self.bst_cb = CatBoostRanker()
                self.bst_cb.load_model('model_cb.bin')
            else:
                errors.append("CatBoost model file missing")

            if os.path.exists('pipeline.pkl'):
                with open('pipeline.pkl', 'rb') as f:
                    self.pipeline = pickle.load(f)
            else:
                errors.append("Pipeline pickle missing")

            if errors:
                raise ModelLoadError("One or more models failed to load", details={"missing_components": errors})

            monitor.log_metrics({"models_loaded": 1}, context="engine_init")
        except Exception as e:
            monitor.log_metrics({"model_load_error": 1}, context="engine_init", metadata={"error": str(e)})
            if not isinstance(e, ModelLoadError):
                raise ModelLoadError(f"Unexpected error during model loading: {str(e)}")
            raise e
            
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        # 1. Stage 1: Retrieval (Mock/Vector)
        # In a real system, this would be a vector search against a Faiss/Pinecone index
        candidates = ["Raita", "Salan", "Gulab Jamun", "Coke", "Thumbs Up", "Papad", "Ice Cream"]
        
        # 2. Cold-Start Handling: Empty Cart or New Session
        if not request.cart_items:
            # Fallback to Global Popular Items (Cold Start)
            popular_items = [
                RecommendationItem(item_id="Coke", score=0.9, metadata={"reason": "Popular choice"}),
                RecommendationItem(item_id="Gulab Jamun", score=0.85, metadata={"reason": "Top dessert"}),
                RecommendationItem(item_id="Raita", score=0.8, metadata={"reason": "Classic side"})
            ]
            return RecommendationResponse(
                request_id=f"req_{np.random.randint(1000)}",
                recommendations=popular_items,
                model_version="fallback_v1"
            )

        # 3. Stage 2: Ranking
        try:
            # Create scoring dataframe
            df = pd.DataFrame([{
                'session_id': 1,
                'cart_items': '|'.join(request.cart_items),
                'candidate_item': c,
                'cuisine': request.context.get('cuisine', 'Indian'),
                'area': request.context.get('area', 'Hitech City'),
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
            
            items = [RecommendationItem(item_id=row['candidate_item'], score=float(row['score']), metadata={}) for _, row in results.iterrows()]
            
            return RecommendationResponse(
                request_id=f"req_{np.random.randint(1000)}",
                recommendations=items,
                model_version="ensemble_v1"
            )
        except Exception as e:
            # Fallback on model failure
            monitor.log_metrics({"model_inference_error": 1}, context="engine_recommend", metadata={"error": str(e)})
            return RecommendationResponse(
                request_id=f"req_{np.random.randint(1000)}",
                recommendations=[RecommendationItem(item_id="Coke", score=0.5, metadata={"note": "fallback"})],
                model_version="fallback_error_v1"
            )
