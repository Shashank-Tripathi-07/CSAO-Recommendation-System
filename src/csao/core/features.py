import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict

class FeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        self.categorical_cols = ['cuisine', 'area', 'candidate_category', 'meal_time']
        self.numeric_cols = ['rating', 'cart_size', 'hour_of_day']

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str).unique().tolist() + ['unknown'])
            self.label_encoders[col] = le
        
        # Simple stats for popularity
        self.item_popularity = df.groupby('candidate_item')['label'].sum().to_dict()
        
        feat_df = self._engineer_features(df)
        self.numeric_features = [c for c in feat_df.columns if c in self.numeric_cols or 'popularity' in c or 'vs_cart' in c]
        self.scaler.fit(feat_df[self.numeric_features].fillna(0))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = self._engineer_features(df)
        feat_df[self.numeric_features] = self.scaler.transform(feat_df[self.numeric_features].fillna(0))
        return feat_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Category-based flags
        df['is_complement_to_any_cart_item'] = df.apply(
            lambda x: 1 if random_heuristic_check(x['cart_items'], x['candidate_item']) else 0, axis=1
        )
        
        # 2. Popularity Rank
        df['candidate_popularity_rank'] = df['candidate_item'].map(lambda x: self.item_popularity.get(x, 0))
        
        # 3. Encoding
        for col in self.categorical_cols:
            le = self.label_encoders.get(col)
            if le:
                df[f'{col}_encoded'] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else le.transform(['unknown'])[0])
        
        return df

def random_heuristic_check(cart_str, candidate):
    # Mocking the heuristic for the pipeline structure
    return "Biryani" in cart_str and candidate in ["Raita", "Salan"]
