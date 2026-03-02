import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
import pickle
from feature_pipeline import FeaturePipeline
import os

def prepare_data(df, pipeline, is_train=False):
    if is_train:
        df_feat = pipeline.fit_transform(df)
    else:
        # For val/test, handle unseen categories gracefully or just transform
        df_feat = pipeline.transform(df)
        
    # LightGBM lambdarank needs data grouped by query (session_id)
    # The data needs to be sorted by session_id to get contiguous groups
    # df_feat already has session_id, label before we dropped it in pipeline?
    # Wait, my pipeline dropped `timestamp`, `session_id`, `label`!
    # I should change pipeline to NOT drop session_id, label, timestamp, so we can group.
    return df_feat

def get_xy_group(df_feat):
    # Sort by session_id to ensure groupby is contiguous
    df_feat = df_feat.sort_values('session_id')
    
    # Get group counts
    group = df_feat.groupby('session_id').size().values
    
    # Extract labels
    y = df_feat['label'].values
    
    # Drop non-feature columns
    cols_to_drop = ['session_id', 'label', 'timestamp', 'restaurant_id', 'cuisine', 'area', 'cart_items', 
                    'candidate_item', 'candidate_category', 'meal_time', 'cost_tier', 'is_complement']
    features = [c for c in df_feat.columns if c not in cols_to_drop]
    X = df_feat[features]
    
    return X, y, group, features

def main():
    print("Loading synthetic sessions...")
    df = pd.read_csv('synthetic_sessions.csv')
    
    print(f"Total rows: {len(df)}")
    
    # Ensure sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Split temporally (80/10/10) WITHOUT shuffling
    n = len(df)
    train_idx = int(n * 0.8)
    val_idx = int(n * 0.9)
    
    df_train = df.iloc[:train_idx]
    df_val = df.iloc[train_idx:val_idx]
    df_test = df.iloc[val_idx:]
    
    print(f"Train/Val/Test sizes: {len(df_train)} / {len(df_val)} / {len(df_test)}")
    
    # Initialize and fit feature pipeline
    print("Applying Feature Pipeline...")
    pipeline = FeaturePipeline()
    
    # Fit and transform
    X_train_full = prepare_data(df_train, pipeline, is_train=True)
    X_val_full = prepare_data(df_val, pipeline, is_train=False)
    X_test_full = prepare_data(df_test, pipeline, is_train=False)
    
    # Get XY data
    X_train, y_train, group_train, feature_cols = get_xy_group(X_train_full)
    X_val, y_val, group_val, _ = get_xy_group(X_val_full)
    X_test, y_test, group_test, _ = get_xy_group(X_test_full)
    
    # Create lgb datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, free_raw_data=False)
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Optuna objective
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [5, 10],
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
            'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [20, 50, 100]),
            'feature_fraction': trial.suggest_categorical('feature_fraction', [0.7, 0.8, 0.9]),
            'verbose': -1,
            'seed': 42
        }
        
        # Train with early stopping
        # LightGBM valid logic: the callback early_stopping is used.
        bst = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            valid_names=['valid'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Return best ndcg@10 on validation set
        return bst.best_score['valid']['ndcg@10']

    print("Starting Optuna sequence...")
    study = optuna.create_study(direction="maximize")
    # Limiting to 10 trials to keep runtimes manageable for this exercise
    study.optimize(objective, n_trials=10)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best validation NDCG@10: ", study.best_trial.value)
    print("Best params: ")
    for key, value in study.best_trial.params.items():
        print("  {}: {}".format(key, value))
        
    # Retrain final model on best params
    best_params = study.best_trial.params
    best_params['objective'] = 'lambdarank'
    best_params['metric'] = 'ndcg'
    best_params['eval_at'] = [5, 10]
    best_params['verbose'] = -1
    best_params['seed'] = 42
    
    print("Training final model...")
    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Save Model and Pipeline
    model_path = 'model_lgb.txt'
    pipeline_path = 'pipeline.pkl'
    
    final_model.save_model(model_path)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"LightGBM Model saved to {model_path} and pipeline to {pipeline_path}")
    
    # --- XGBoost Model ---
    print("\nTraining XGBoost model...")
    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dtrain_xgb.set_group(group_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)
    dval_xgb.set_group(group_val)
    
    xgb_params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1000,
        evals=[(dval_xgb, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    xgb_model_path = 'model_xgb.json'
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost Model saved to {xgb_model_path}")
    
    # Optionally save test set for evaluate.py
    df_test.to_csv('test_sessions.csv', index=False)
    print("Test sessions saved to test_sessions.csv for evaluation.")

if __name__ == "__main__":
    main()
