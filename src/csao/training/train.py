import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool
import pickle
import os
from csao.core.features import FeaturePipeline
from csao.training.data_generation import generate_synthetic_sessions

def main():
    print("Step 1: Generating Training Data...")
    df = generate_synthetic_sessions(None, num_sessions=50000)
    
    # Split into train/val
    session_ids = df['session_id'].unique()
    np.random.shuffle(session_ids)
    train_ids = session_ids[:int(0.8 * len(session_ids))]
    val_ids = session_ids[int(0.8 * len(session_ids)):]
    
    df_train = df[df['session_id'].isin(train_ids)].sort_values('session_id')
    df_val = df[df['session_id'].isin(val_ids)].sort_values('session_id')
    
    print(f"Train sessions: {len(train_ids)}, Val sessions: {len(val_ids)}")
    
    print("Step 2: Fitting Feature Pipeline...")
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(df_train)
    X_val = pipeline.transform(df_val)
    
    # Save pipeline
    with open('pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    y_train = df_train['label'].values
    y_val = df_val['label'].values
    
    q_train = df_train.groupby('session_id')['session_id'].count().values
    q_val = df_val.groupby('session_id')['session_id'].count().values
    
    features = ['cart_size', 'hour_of_day', 'is_complement_to_any_cart_item', 'candidate_popularity_rank', 
                'cuisine_encoded', 'area_encoded', 'candidate_category_encoded', 'meal_time_encoded']
    
    print(f"Training on {len(features)} features: {features}")
    
    print("Step 3: Training LightGBM...")
    train_data = lgb.Dataset(X_train[features], label=y_train, group=q_train)
    val_data = lgb.Dataset(X_val[features], label=y_val, group=q_val, reference=train_data)
    
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [5],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbosity': -1,
        'seed': 42
    }
    
    bst_lgb = lgb.train(lgb_params, train_data, num_boost_round=100, valid_sets=[val_data])
    bst_lgb.save_model('model_lgb.txt')
    
    print("Step 4: Training XGBoost...")
    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtrain.set_group(q_train)
    dval = xgb.DMatrix(X_val[features], label=y_val)
    dval.set_group(q_val)
    
    xgb_params = {
        'objective': 'rank:ndcg',
        'eta': 0.1,
        'max_depth': 6,
        'eval_metric': 'ndcg@5',
        'seed': 42
    }
    
    bst_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=[(dval, 'val')], verbose_eval=False)
    bst_xgb.save_model('model_xgb.json')
    
    print("Step 5: Training CatBoost...")
    train_pool = Pool(data=X_train[features], label=y_train, group_id=df_train['session_id'])
    val_pool = Pool(data=X_val[features], label=y_val, group_id=df_val['session_id'])
    
    cb_model = CatBoostRanker(iterations=100, learning_rate=0.1, depth=6, loss_function='PairLogit', bootstrap_type='No', random_seed=42, verbose=False)
    cb_model.fit(train_pool, eval_set=val_pool)
    cb_model.save_model('model_cb.bin')
    
    print("Training Complete. All models saved.")

if __name__ == "__main__":
    main()
