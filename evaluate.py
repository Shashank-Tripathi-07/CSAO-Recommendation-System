import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    ideal_dcg = dcg_at_k(sorted(r, reverse=True), k)
    if not ideal_dcg:
        return 0.
    return dcg_at_k(r, k) / ideal_dcg

def compute_metrics(group, score_col, k_list=[5, 10]):
    # Sort by score
    sorted_group = group.sort_values(score_col, ascending=False)
    y_true = sorted_group['label'].values
    
    metrics = {}
    
    # Precision & Recall
    num_relevant = np.sum(y_true)
    for k in k_list:
        top_k = y_true[:k]
        metrics[f'P@{k}'] = np.sum(top_k) / k
        metrics[f'R@{k}'] = np.sum(top_k) / num_relevant if num_relevant > 0 else 0.0
        metrics[f'NDCG@{k}'] = ndcg_at_k(y_true, k)
        
    return pd.Series(metrics)

def main():
    print("Loading test sessions...")
    df_test = pd.read_csv('test_sessions.csv')
    
    print("Loading models and pipeline...")
    bst_lgb = lgb.Booster(model_file='model_lgb.txt')
    bst_xgb = xgb.Booster(model_file='model_xgb.json')
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    print("Applying pipeline...")
    df_feat = pipeline.transform(df_test)
    
    cols_to_drop = ['session_id', 'label', 'timestamp', 'restaurant_id', 'cuisine', 'area', 'cart_items', 
                    'candidate_item', 'candidate_category', 'meal_time', 'cost_tier', 'is_complement']
    features = [c for c in df_feat.columns if c not in cols_to_drop]
    
    X_test = df_feat[features]
    y_test = df_feat['label'].values
    
    # Predict
    print("Predicting scores...")
    df_test['lgb_score'] = bst_lgb.predict(X_test)
    
    dtest = xgb.DMatrix(X_test)
    df_test['xgb_score'] = bst_xgb.predict(dtest)
    
    # Standardize predictions per session and average for Ensemble
    def blend_scores(group):
        for col in ['lgb_score', 'xgb_score']:
            std = group[col].std()
            if std > 0:
                group[col + '_std'] = (group[col] - group[col].mean()) / std
            else:
                group[col + '_std'] = 0.0
        group['stacked_score'] = group['lgb_score_std'] + group['xgb_score_std']
        return group
        
    print("Computing stacked ensemble scores...")
    df_test = df_test.groupby('session_id', as_index=False).apply(blend_scores)
    # the apply might return index that causes issues, safely reset it
    df_test = df_test.reset_index(drop=True)
    
    # Baselines
    np.random.seed(42)
    df_test['random_score'] = np.random.rand(len(df_test))
    
    # Popularity uses df_feat
    df_test['popularity_score'] = 1.0 / (df_feat['candidate_popularity_rank'] + 1)
    
    # Overall AUC
    auc_model, auc_pop, auc_rand = 0.0, 0.0, 0.0
    try:
        auc_model = roc_auc_score(y_test, df_test['lgb_score'])
        auc_pop = roc_auc_score(y_test, df_test['popularity_score'])
        auc_rand = roc_auc_score(y_test, df_test['random_score'])
    except ValueError:
        pass # All 0s or something
        
    print(f"\n--- OVERALL AUC ---")
    print(f"LGBM AUC:       {auc_model:.4f}")
    print(f"Popularity AUC: {auc_pop:.4f}")
    print(f"Random AUC:     {auc_rand:.4f}")

    def evaluate_model(df, name, score_col):
        group_metrics = df.groupby('session_id', as_index=False).apply(lambda g: compute_metrics(g, score_col))
        
        # Select only numeric columns for mean to avoid pandas ambiguity on strings/indices
        numeric_cols = [col for col in group_metrics.columns if col != 'session_id']
        mean_metrics = group_metrics[numeric_cols].mean()
        
        print(f"\n--- {name} Metrics ---")
        for k, v in mean_metrics.items():
            print(f"{k}: {v:.4f}")

    evaluate_model(df_test, "LightGBM", 'lgb_score')
    evaluate_model(df_test, "XGBoost", 'xgb_score')
    evaluate_model(df_test, "Stacked Ensemble", 'stacked_score')
    evaluate_model(df_test, "Popularity Baseline", 'popularity_score')
    evaluate_model(df_test, "Random Baseline", 'random_score')
    
    # Breakdowns
    print("\n\n--- BREAKDOWNS (Model P@5 / NDCG@10) ---")
    # Cost tier: 0 is budget, 2 is premium
    df_test['segment'] = df_test['cost_tier'].apply(lambda x: 'budget' if x == 0 else ('premium' if x == 2 else 'mid'))
    df_test['cart_size_group'] = df_test['cart_size'].apply(lambda x: '1 item' if x == 1 else '2+ items')
    
    for breakdown_col in ['segment', 'meal_time', 'cart_size_group', 'cuisine']:
        print(f"\nBreakdown by {breakdown_col.upper()}:")
        session_info = df_test[['session_id', breakdown_col]].drop_duplicates()
        
        # We merge group metrics back to grab session slice
        group_metrics = df_test.groupby('session_id').apply(lambda g: compute_metrics(g, 'stacked_score')).reset_index()
        merged = pd.merge(group_metrics, session_info, on='session_id')
        
        agg = merged.groupby(breakdown_col)[['P@5', 'NDCG@10']].mean()
        print(agg)

if __name__ == "__main__":
    main()
