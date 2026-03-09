import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
import pickle
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(df, name, score_col):
    # Mean of P@5, NDCG@5, NDCG@10 per session
    def get_metrics(group):
        group = group.sort_values(score_col, ascending=False)
        top_5 = group.head(5)
        top_10 = group.head(10)
        
        p5 = top_5['label'].mean()
        r5 = top_5['label'].sum() / group['label'].sum() if group['label'].sum() > 0 else 0
        
        # NDCG simplified
        def dcg(labels):
            return np.sum(labels / np.log2(np.arange(2, len(labels) + 2)))
        
        idcg5 = dcg(sorted(group['label'].values, reverse=True)[:5])
        dcg5 = dcg(top_5['label'].values)
        ndcg5 = dcg5 / idcg5 if idcg5 > 0 else 0
        
        idcg10 = dcg(sorted(group['label'].values, reverse=True)[:10])
        dcg10 = dcg(top_10['label'].values)
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0
        
        return pd.Series({'P@5': p5, 'R@5': r5, 'NDCG@5': ndcg5, 'NDCG@10': ndcg10})

    metrics = df.groupby('session_id').apply(get_metrics).mean()
    print(f"\n--- {name} Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

def main():
    print("Loading test sessions...")
    df_test = pd.read_csv('test_sessions.csv')
    
    print("Loading models and pipeline...")
    bst_lgb = lgb.Booster(model_file='model_lgb.txt')
    bst_xgb = xgb.Booster(model_file='model_xgb.json')
    bst_cb = CatBoostRanker()
    bst_cb.load_model('model_cb.bin')
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    print("Applying pipeline...")
    X_test = pipeline.transform(df_test)
    
    features = ['cart_size', 'hour_of_day', 'is_complement_to_any_cart_item', 'candidate_popularity_rank', 
                'cuisine_encoded', 'area_encoded', 'candidate_category_encoded', 'meal_time_encoded']
    
    # Predict
    print("Predicting scores...")
    df_test['lgb_score'] = bst_lgb.predict(X_test[features])
    
    dtest = xgb.DMatrix(X_test[features])
    df_test['xgb_score'] = bst_xgb.predict(dtest)

    df_test['cb_score'] = bst_cb.predict(X_test[features])
    
    # Standardize predictions per session and average for Ensemble
    def blend_scores(group):
        for col in ['lgb_score', 'xgb_score', 'cb_score']:
            std = group[col].std()
            if std > 0:
                group[col + '_std'] = (group[col] - group[col].mean()) / std
            else:
                group[col + '_std'] = 0.0
        group['stacked_score'] = (group['lgb_score_std'] + group['xgb_score_std'] + group['cb_score_std']) / 3.0
        return group
        
    print("Computing stacked ensemble scores...")
    df_test = df_test.groupby('session_id').apply(blend_scores)
    
    # Baselines
    df_test['popularity_score'] = df_test['candidate_popularity_rank']
    df_test['random_score'] = np.random.rand(len(df_test))

    evaluate_model(df_test, "LightGBM", 'lgb_score')
    evaluate_model(df_test, "XGBoost", 'xgb_score')
    evaluate_model(df_test, "CatBoost", 'cb_score')
    evaluate_model(df_test, "Stacked Ensemble", 'stacked_score')
    evaluate_model(df_test, "Popularity Baseline", 'popularity_score')
    evaluate_model(df_test, "Random Baseline", 'random_score')

if __name__ == "__main__":
    main()
