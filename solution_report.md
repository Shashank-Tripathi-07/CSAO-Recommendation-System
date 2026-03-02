# Project Report: Intelligent CSAO Recommendation System

## 1. Problem Statement & Objectives
The goal was to design and implement a **Cart Super Add-On (CSAO)** recommendation rail for a food delivery platform. The system must suggest contextually relevant items (e.g., suggesting "Drinks" if "Biryani" is in the cart) in real-time to maximize:
*   **Average Order Value (AOV)**
*   **Cart-to-Order Conversion Rate**
*   **Add-on Acceptance Rate**

### Key Constraints:
*   **Latency**: Recommendations must be served within **200-300ms**.
*   **Real-time**: The rail must refresh every time an item is added to the cart.
*   **Relevance**: Suggestions must be logically compatible and diverse.

---

## 2. Technical Stack
*   **Language**: Python 3.12
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: LightGBM (Primary), XGBoost (Advanced)
*   **Optimization**: Optuna (Hyperparameter Tuning)
*   **Pipeline**: Scikit-Learn (Feature Engineering)
*   **Serialization**: Pickle, JSON

---

## 3. The Approach: Two-Stage Ranking Architecture
To balance precision and low latency, we implemented a two-stage pipeline:

### Stage 1: Candidate Generation (Heuristic)
Instead of scoring all 50,000+ restaurants or 1,000+ items, we use high-speed heuristics to find ~50-100 candidates:
*   **Cuisine Mappings**: Items matching the restaurant's cuisine.
*   **Complement rules**: Known logic (Cart has Pizza -> Candidate Garlic Bread).
*   **Popularity Fallback**: Top-selling items in the user's area.

### Stage 2: Ranking (ML Ensemble)
A machine learning layer scores the candidates based on complex interactions. 
*   **Features**: We extract 28 features across four categories: Cart-level, Candidate-specific, Restaurant-metadata, and Context (Meal-time, Hour, etc.).
*   **Objective**: Pointwise/Pairwise ranking using LambdaRank logic.

---

## 4. Mathematical Foundations

### Loss Function: LambdaRank
Standard classification (Log-loss) doesn't optimize for ranking order. We use **LambdaRank**, which defines a virtual gradient ("Lambda") for each item based on its potential to improve the NDCG if moved up or down the list.
*   $NDCG_k = \frac{DCG_k}{IDCG_k}$ where $DCG_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}$

### Ensemble Normalization: Z-Score Blending
Since LightGBM and XGBoost output scores on different scales, simple addition $S_1 + S_2$ is biased toward the model with higher variance. We use Z-score normalization for blending:
$$S_{final} = \frac{S_{LGBM} - \mu_{LGBM}}{\sigma_{LGBM}} + \frac{S_{XGB} - \mu_{XGB}}{\sigma_{XGB}}$$
This ensures both models contribute equally to the final rank.

---

## 5. Experimentation History

### Failed/Sub-Optimal Approaches:
1.  **Single-Stage Global Ranking**: Attempting to score every item in the database for every request. **Result**: Latency exceeded 2,000ms.
2.  **Raw Score Blending**: Adding LGBM and XGB scores directly. **Result**: XGBoost (higher score range) effectively ignored LGBM's signals.
3.  **Random Data Shuffling**: Splitting data randomly for training. **Result**: Data leakage occurred because future interactions influenced historical predictions, leading to over-optimistic precision metrics.

### Approaches That Worked:
1.  **Temporal Splitting**: 80/10/10 split based on timestamps, ensuring the model learns to predict "future" items from "past" data.
2.  **Stacked Ensemble (LGBM + XGBoost)**: Combining models improved NDCG@10 from 0.578 to 0.581, confirming that different GBDT implementations capture slightly different feature subspaces.
3.  **Post-Ranking Diversity Filter**: Restricting category frequency (max 2 items per category) significantly improved the visual "rail" quality without hurting NDCG.

---

## 6. Final Performance Summary
*   **SLA Compliance**: 143ms (Multi-model ensemble) — **SUCCESS** 
*   **Accuracy (Ensemble)**: NDCG@10: 0.5809 / Precision@5: 0.2818
*   **Business Impact (Projected)**: ~15% lift in add-on completion rate compared to popularity-only baselines.
