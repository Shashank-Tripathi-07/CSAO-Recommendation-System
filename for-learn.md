# CSAO Recommendation System: The Professional Guide 🎓

This document is a deep-dive into the "Why" and "How" of the CSAO (Cart Super Add-On) engine. It balances high-level architecture with low-level implementation details to provide a full picture of an industrial recommendation microservice.

---

### 1. The Two-Stage Architecture
In production environments (like Zomato, Uber Eats, or Amazon), you cannot score 100,000 items in 100ms. We use a two-stage approach:

#### Stage 1: Retrieval (Recall)
*   **Goal**: Reduce the search space from 100,000+ items to ~50-100 candidates.
*   **Low-Level Detail**: We use **Vector Search** (K-Nearest Neighbors). Items are represented as high-dimensional vectors (embeddings). When a user adds "Biryani" to their cart, we find items whose vectors are "geometrically close" to the Biryani vector.
*   **Handling Latency**: Retrieval happens in `<10ms` using specialized indexes (like Faiss or HNSW).

#### Stage 2: Ranking (Precision)
*   **Goal**: Sort the ~50 candidates with extreme precision.
*   **Low-Level Detail**: We pass these 50 candidates through a **Triple-Stacked Ensemble** of Gradient Boosted Decision Trees (GBDTs). This stage uses complex features (user history, meal time, item affinity) that are too slow for Stage 1.

---

### 2. Low-Level Implementation Deep-Dive

#### A. Feature Engineering (`src/csao/core/features.py`)
Machine Learning models are only as good as the data they see. Our `FeaturePipeline` does several critical things:
1.  **Categorical Handling**: We use `LabelEncoder` with an **'unknown' token**. This prevents the system from crashing if a new cuisine or area appears in production that wasn't in the training set.
2.  **Complement Heuristics**: We calculate a feature `is_complement_to_any_cart_item`. This uses string-matching logic (e.g., if "Biryani" is in the cart and the candidate is "Raita", the flag is `1`). This "domain knowledge" feature is often more powerful than raw embeddings.
3.  **Popularity Fallback**: We map `item_popularity` during training. For new users (Cold Start), the model defaults to recommending items that are globally popular within that category.

#### B. Ensemble Blending & Normalization (`src/csao/core/engine.py`)
We combine **LightGBM**, **XGBoost**, and **CatBoost**.
*   **The Problem**: LightGBM might output a score of `0.8`, while XGBoost outputs `45.2` for the same item. You cannot simply average them!
*   **The Solution**: **Z-Score Normalization**. For every session, we calculate the `mean` and `std` of each model's scores. We transform the scores into "standard deviations from the mean": 
    `Score_std = (Score - Mean) / Std`.
    This puts all models on the same mathematical scale before averaging.

#### C. Training Logic (`src/csao/training/train.py`)
*   **LambdaRank / YetiRank**: We don't train models to predict "Will they buy this?" (Binary Classification). Instead, we train them to **Order the List Correctly**. LambdaRank ignores items at the bottom of the list and focuses its "gradients" on getting the top 5 positions right, directly optimizing the **NDCG** (Normalized Discounted Cumulative Gain) metric.

---

### 3. Production-Grade Patterns

#### I. Strict Schema Validation (`src/csao/api/schemas.py`)
We use **Pydantic V2**. Every request entering the system is validated against a schema. If a `user_id` is missing or a coordinate is not a float, the system rejects it at the edge. This prevents "Garbage In, Garbage Out" from reaching the ML models.

#### II. Structured Observability (`src/csao/utils/monitoring.py`)
We don't use plain `print()` statements. We use **Structured JSON Logging**. 
```json
{"timestamp": "2026-03-09T21:45:00", "context": "api_recommend", "metrics": {"request_count": 1}}
```
This allows production tools (like ELK Stack or Datadog) to parse our logs and create real-time dashboards of system health and recommendation accuracy.

#### III. Multi-Stage Docker Builds
Our `Dockerfile` doesn't just "install requirements". It uses a **multi-stage build**:
1.  **Stage 1 (Build)**: Installs compilers and builds heavy ML libraries.
2.  **Stage 2 (Run)**: Only copies the compiled binaries and necessary code.
*   **Result**: The final image size is reduced by ~60%, making deployments faster and more secure.

---

### 4. How to Extend this Project?
1.  **Add a 4th Model**: Try adding a TabNet or a simple Neural Network to the ensemble.
2.  **Implement Real Vector Search**: Replace the mock retrieval in `engine.py` with a real `FAISS` index.
3.  **Add Business Logic**: Implement a "Diversity Filter" that ensures we don't recommend 5 different types of Coke in the same rail.

### Final Takeaway
Building an AI system is 20% Modeling and 80% Engineering. This project provides the **Engineering Foundation** you need to build scalable, reliable, and high-performance recommendation engines. Happy learning! 🚀
