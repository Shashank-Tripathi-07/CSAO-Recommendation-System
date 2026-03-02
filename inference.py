import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
import time

class RecommendationEngine:
    def __init__(self, model_lgb_path='model_lgb.txt', model_xgb_path='model_xgb.json', pipeline_path='pipeline.pkl', catalog_path='item_catalog.json'):
        self.bst_lgb = lgb.Booster(model_file=model_lgb_path)
        self.bst_xgb = xgb.Booster(model_file=model_xgb_path)
        with open(pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)
            
        with open(catalog_path, 'r') as f:
            self.item_catalog = json.load(f)
            
        self.all_items = list(self.item_catalog.keys())
        
        # Mock database of restaurant features (In prod this would be Redis)
        self.restaurant_store = {
            1: {'cuisine': 'North Indian', 'rating': 4.5, 'cost_tier': 1},
            2: {'cuisine': 'Biryani', 'rating': 4.2, 'cost_tier': 0}
        }
        
    def _generate_candidates(self, restaurant_id: int, cuisine: str, cart_items: list) -> list:
        # Stage 1: Candidate Generation
        # Get all mapped items for cuisine or fallback to popularity globally
        # Because we don't have the original CUISINE_MAPPING dict dynamically loaded, we filter self.item_catalog
        candidates = [item for item in self.all_items if item not in cart_items]
        
        # To hit 50-100 items latency budget, we just randomly sample 50 from remaining menu if large
        # Here we just use all, as our menus are small (~15-20 max per cuisine)
        return candidates
        
    def get_recommendations(self, user_id: str, restaurant_id: int, cart_items: list, context: dict):
        start_time = time.time()
        
        # Cold start handling for restaurant
        is_cold_start = False
        if restaurant_id not in self.restaurant_store:
            # Fallback to average/default features
            is_cold_start = True
            rest_feat = {'cuisine': 'North Indian', 'rating': 3.5, 'cost_tier': 1}
        else:
            rest_feat = self.restaurant_store[restaurant_id]
            
        # 1. Candidate Generation
        candidates = self._generate_candidates(restaurant_id, rest_feat['cuisine'], cart_items)
        if not candidates:
            return {'recommendations': [], 'latency_ms': 0, 'is_cold_start': is_cold_start}
            
        # 2. Build feature dataframe
        cart_str = '|'.join(cart_items)
        records = []
        for cand in candidates:
            row = {
                'restaurant_id': restaurant_id,
                'cuisine': rest_feat['cuisine'],
                'area': context.get('location', 'unknown'),
                'cost_tier': rest_feat['cost_tier'],
                'rating': rest_feat['rating'],
                'cart_items': cart_str,
                'candidate_item': cand,
                'candidate_category': self.item_catalog.get(cand, 'main'),
                'hour_of_day': context.get('hour', 12),
                'meal_time': context.get('meal_time', 'lunch'),
                'is_complement': False, # In prod, query Redis for precomputed rules
            }
            records.append(row)
            
        df = pd.DataFrame(records)
        df_feat = self.pipeline.transform(df)
        
        # Keep track of items to append scores later
        cols_to_drop = ['session_id', 'label', 'timestamp', 'restaurant_id', 'cuisine', 'area', 'cart_items', 
                        'candidate_item', 'candidate_category', 'meal_time', 'cost_tier', 'is_complement']
        # Use LGBM's robust feature list to guarantee exact order match
        features = self.bst_lgb.feature_name()
        
        X = df_feat[features]
        
        # Scoring with LightGBM
        lgb_scores = self.bst_lgb.predict(X)
        
        # Scoring with XGBoost
        dtest = xgb.DMatrix(X, feature_names=features)
        xgb_scores = self.bst_xgb.predict(dtest)
        
        # Standardize for balanced ensembling
        lgb_std = np.std(lgb_scores)
        xgb_std = np.std(xgb_scores)
        
        if lgb_std > 0:
            lgb_norm = (lgb_scores - np.mean(lgb_scores)) / lgb_std
        else:
            lgb_norm = 0.0
            
        if xgb_std > 0:
            xgb_norm = (xgb_scores - np.mean(xgb_scores)) / xgb_std
        else:
            xgb_norm = 0.0
            
        stacked_scores = lgb_norm + xgb_norm
        
        df['score'] = stacked_scores
        
        # Sort descending
        df_sorted = df.sort_values('score', ascending=False)
        
        # Post-rank Diversity Filter (max 2 per category)
        final_recs = []
        cat_counts = {'main': 0, 'side': 0, 'dessert': 0, 'drink': 0}
        
        # Prices in INR (₹)
        base_prices = {0: 100, 1: 200, 2: 400} 
        
        for _, row in df_sorted.iterrows():
            cat = row['candidate_category']
            if cat_counts.get(cat, 0) < 2:
                final_recs.append({
                    'item_name': row['candidate_item'],
                    'category': cat,
                    'estimated_price': base_prices.get(rest_feat['cost_tier'], 200),
                    'score': float(row['score']),
                    'is_cold_start_prediction': is_cold_start
                })
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                
            if len(final_recs) == 10:
                break
                
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'latency_ms': latency_ms,
            'is_cold_start': is_cold_start,
            'recommendations': final_recs
        }

if __name__ == "__main__":
    print("Loading Engine...")
    engine = RecommendationEngine()
    print("Engine Loaded.")
    
    # Test request
    req = {
        'user_id': "u123",
        'restaurant_id': 1,
        'cart_items': ['Chicken Biryani'],
        'context': {'hour': 19, 'meal_time': 'dinner', 'location': 'Koramangala'}
    }
    
    print("Running Inference...")
    res = engine.get_recommendations(**req)
    if res['latency_ms'] > 300:
        print(f"FAILED LATENCY SLA: {res['latency_ms']:.2f} ms")
    else:
        print(f"PASSED LATENCY SLA: {res['latency_ms']:.2f} ms")
        
    print(json.dumps(res, indent=2))
