# CSAO Recommendation Engine: Model Evaluation Results

This document outlines the evaluation metrics and required criteria for the three recommendation approaches tested: **LightGBM**, **XGBoost (XGBRanker)**, and the combined **Stacked Ensemble**. 

Performance is evaluated across standard ranking metrics (NDCG and Precision) and system SLA constraints.

---

## 1. Overall Model Performance
Performance was evaluated on a held-out test set (`test_sessions.csv`) composed of ~97.5K synthetic user cart sessions.

| Metric | Random Baseline | Popularity Baseline | LightGBM | XGBoost | Stacked Ensemble |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AUC** | 0.4982 | 0.5161 | 0.6073 | 0.6074 | **0.6080** |
| **NDCG@5** | - | - | 0.4411 | 0.4447 | **0.4456** |
| **NDCG@10** | - | - | 0.5782 | 0.5810 | **0.5809** |
| **Precision@5** | - | - | 0.2800 | 0.2802 | **0.2818** |
| **Recall@5**  | - | - | 0.5407 | 0.5435 | **0.5465** |
| **Recall@10** | - | - | 0.8700 | 0.8698 | **0.8709** |

*Note: The Stacked Ensemble provided a consistent, marginal improvement across almost all ranking metrics, particularly in `Precision@5` and `Recall@5`.*

---

## 2. Segment Breakdowns (P@5 / NDCG@10)
To ensure the models generalise well, we evaluated performance across different user contexts using the Stacked Ensemble.

### By Cost Tier
* **Budget**: P@5 = 0.2831 / NDCG@10 = 0.5841
* **Mid**: P@5 = 0.2804 / NDCG@10 = 0.5780
* **Premium**: P@5 = 0.2815 / NDCG@10 = 0.5785

### By Cart Size
* **1 Item**: P@5 = 0.2974 / NDCG@10 = 0.6085
* **2+ Items**: P@5 = 0.2687 / NDCG@10 = 0.5562
*(The model is slightly more accurate at predicting the immediate next item on small carts where signals are less noisy).*

---

## 3. SLA & System Criteria

### Latency Constraint: `< 300ms`
- **Candidate Generation**: Extracting valid menu items and filtering existing cart items.
- **Scoring**: Running both `bst_lgb.predict()` and `xgb.DMatrix` inferences in parallel in-memory, then standardizing and blending the scores.
- **Post-Processing**: Applying diversity filters (max 2 items per category).
- **Result**: **~143ms** per request.
- **SLA Status**: **PASSED** ✅ (Performance leaves ~150ms buffer for network/database overhead).

### Diversity Filter
- **Constraint**: Ensure recommendations are not monotonous (e.g., suggesting 5 drinks).
- **Result**: The inference engine successfully caps categories at `max=2`, ensuring a mix of mains, sides, desserts, and drinks are presented in a single rail.
- **Status**: **PASSED** ✅

### Cold Start Handling
- **Constraint**: Must handle new restaurants without historical interactions or feature mappings.
- **Result**: The system defaults to average metrics (Rating: 3.5, Cost Tier: 1) and falls back to generating generic cuisine-level candidates.
- **Status**: **PASSED** ✅
