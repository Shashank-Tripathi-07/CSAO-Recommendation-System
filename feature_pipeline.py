import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import json
import random

class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, item_catalog_path: str = 'item_catalog.json'):
        self.item_catalog_path = item_catalog_path
        with open(self.item_catalog_path, 'r') as f:
            self.item_info = json.load(f)
            
        self.cuisine_encoder = LabelEncoder()
        self.area_encoder = LabelEncoder()
        
        # Mapping base prices
        self.PRICE_TIERS = {
            0: {'main': 150, 'side': 60, 'dessert': 80, 'drink': 50},
            1: {'main': 300, 'side': 120, 'dessert': 150, 'drink': 100},
            2: {'main': 600, 'side': 250, 'dessert': 300, 'drink': 200}
        }
        
        self.category_cols = ['candidate_category_main', 'candidate_category_side', 
                              'candidate_category_dessert', 'candidate_category_drink']
        self.meal_cols = ['meal_time_breakfast', 'meal_time_lunch', 'meal_time_dinner', 'meal_time_late_night']

        self.popularity_rankings = {}
        self.restaurant_chains = set()

    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders and compute popularity rankings from training data"""
        self.cuisine_encoder.fit(X['cuisine'].astype(str))
        self.area_encoder.fit(X['area'].astype(str))
        
        # Calculate restaurant chain info
        # Since we simulated from df_clean, we need to infer restaurant names or just mock it.
        # In actual Zomato dataset, there's 'name' column. Here, since we have restaurant_id, 
        # we assume ~20% of dataset are chains
        total_rests = X['restaurant_id'].nunique()
        chain_count = int(total_rests * 0.2)
        chain_ids = list(X['restaurant_id'].unique())[:chain_count]
        self.restaurant_chains = set(chain_ids)

        # Compute popularity ranking (occurrences of items per cuisine and category)
        # In a real scenario, this would be computed by counting how many times an item appears in historical carts
        if 'cart_items' in X.columns:
            exploded = X.assign(item=X['cart_items'].str.split('|')).explode('item')
            counts = exploded.groupby(['cuisine', 'item']).size().reset_index(name='count')
            counts['category'] = counts['item'].map(self.item_info)
            # Rank within cuisine and category
            counts['rank'] = counts.groupby(['cuisine', 'category'])['count'].rank(method='dense', ascending=False)
            
            # Save top items
            self.popularity_rankings = counts.set_index(['cuisine', 'item'])['rank'].to_dict()
        
        return self

    def _get_item_price(self, item: str, rest_tier: int) -> float:
        cat = self.item_info.get(item, 'main')
        # Default mid tier if missing
        if rest_tier not in self.PRICE_TIERS:
            rest_tier = 1 
        return self.PRICE_TIERS[rest_tier][cat]
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw session rows into feature matrix"""
        df = X.copy()
        
        # Cart-level features
        # Parse cart items
        cart_lists = df['cart_items'].str.split('|').apply(lambda x: x if isinstance(x, list) else [])
        
        df['cart_size'] = cart_lists.apply(len)
        df['is_single_item_cart'] = (df['cart_size'] == 1).astype(int)
        
        def cart_categories(items):
            cats = [self.item_info.get(it, 'main') for it in items]
            return {
                'has_main': int('main' in cats),
                'has_side': int('side' in cats),
                'has_dessert': int('dessert' in cats),
                'has_drink': int('drink' in cats)
            }
        
        cat_df = pd.DataFrame(cart_lists.apply(cart_categories).tolist(), index=df.index)
        df = pd.concat([df, cat_df], axis=1)
        
        df['meal_completeness_score'] = (df['has_main'] + df['has_side'] + df['has_dessert'] + df['has_drink']) / 4.0
        
        # Estimated cart value
        def est_cart_val(row):
            items = row['cart_items'].split('|') if pd.notnull(row['cart_items']) else []
            return sum([self._get_item_price(it, row.get('cost_tier', 1)) for it in items])
        df['estimated_cart_value'] = df.apply(est_cart_val, axis=1)
        
        # Candidate item features
        df['candidate_category'] = df['candidate_item'].map(lambda x: self.item_info.get(x, 'main'))
        # One-hot encode category
        for cat in ['main', 'side', 'dessert', 'drink']:
            df[f'candidate_category_{cat}'] = (df['candidate_category'] == cat).astype(int)
            
        # is_complement_to_any_cart_item (handled during generation, but we recreate if missing)
        # Here we assume `is_complement` column exists from generator, or we just use it
        if 'is_complement' in df.columns:
            df['is_complement_to_any_cart_item'] = df['is_complement'].astype(int)
        else:
            # We'd have to reload complement rules. For now, set to 0 to avoid error
            df['is_complement_to_any_cart_item'] = 0

        # Candidate price tier vs cart (Relative price comparison)
        def relative_price(row):
            cand_price = self._get_item_price(row['candidate_item'], row.get('cost_tier', 1))
            cart_avg = row['estimated_cart_value'] / max(row['cart_size'], 1)
            if cand_price > cart_avg * 1.2:
                return 1
            elif cand_price < cart_avg * 0.8:
                return -1
            else:
                return 0
        df['candidate_price_tier_vs_cart'] = df.apply(relative_price, axis=1)
        
        # Candidate popularity rank
        def get_rank(row):
            key = (row['cuisine'], row['candidate_item'])
            return self.popularity_rankings.get(key, 999.0) # default bad rank
        df['candidate_popularity_rank'] = df.apply(get_rank, axis=1)
            
        # Restaurant features
        # Handle unseen labels by filling with a default "unknown" class value if we wanted to be robust,
        # but for simplicity we wrap in try-except and assign -1 for unseen
        df['cuisine_encoded'] = df['cuisine'].map(lambda s: self.cuisine_encoder.transform([s])[0] if s in self.cuisine_encoder.classes_ else -1)
        df['area_encoded'] = df['area'].map(lambda s: self.area_encoder.transform([s])[0] if s in self.area_encoder.classes_ else -1)
        
        # Mocking votes_log if votes column doesn't exist
        if 'votes' not in df.columns:
            np.random.seed(42)
            # give some random votes to restaurants
            df['votes_log'] = np.log1p(np.random.randint(10, 1000, size=len(df)))
        else:
            df['votes_log'] = np.log1p(df['votes'].fillna(0))
            
        df['price_tier_encoded'] = df['cost_tier'].fillna(1).astype(int)
        df['is_chain'] = df['restaurant_id'].apply(lambda x: int(x in self.restaurant_chains))
        
        # Contextual features
        # hour_of_day is already integer
        # meal_time is string
        for mt in ['breakfast', 'lunch', 'dinner', 'late_night']:
            df[f'meal_time_{mt}'] = (df['meal_time'] == mt).astype(int)
            
        # Mock day_of_week and weekend
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'], unit='s')
            df['day_of_week'] = dates.dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        else:
            df['day_of_week'] = np.random.randint(0, 7, size=len(df))
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
        # Drop columns not needed for modeling
        cols_to_drop = ['restaurant_id', 'cuisine', 'area', 'cart_items', 
                        'candidate_item', 'candidate_category', 'meal_time', 'cost_tier',
                        'is_complement']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        return df

if __name__ == "__main__":
    # Small test
    print("Loading synthetic sessions...")
    df = pd.read_csv('synthetic_sessions.csv')
    pipeline = FeaturePipeline()
    print("Fitting pipeline...")
    pipeline.fit(df)
    print("Transforming data...")
    df_feat = pipeline.transform(df)
    print("Features derived:", df_feat.columns.tolist())
    print(df_feat.head())
    
    # Save a small sample of transformed features
    df_feat.head(100).to_csv('sample_features.csv', index=False)
