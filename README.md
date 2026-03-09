# 🚀 High-Performance CSAO Recommendation System

```text
      [ USER REQUEST ]
             |
             v
    +-----------------+
    |   REST API      | <--- FastAPI / Pydantic (Validation & OpenAPI)
    +-----------------+
             |
      [ SCHEMATIZED REQ ]
             |
             v
    +-----------------------+     +-----------------------+
    | STAGE 1: RETRIEVAL     |     |  ITEM CATALOG (EMB)   |
    | (Recall / Vector NN)  | <--> |  [50k+ Items Space]   |
    +-----------------------+     +-----------------------+
             |
      [ ~50 CANDIDATES ]
             |
             v
    +-----------------------+     +-----------------------+
    | STAGE 2: RANKING       |     |   GDBT ENSEMBLE       |
    | (Precision / Lambda)  | <--- |   [LGBM + XGB + CB]   |
    +-----------------------+     +-----------------------+
             |
      [ SORTED LIST ]
             |
             v
    +-----------------+
    | DIVERSITY FILTER| <--- Business Logic Tier
    +-----------------+
             |
      [ FINAL RAIL ]
```

[![CI](https://github.com/shashank-tripathi/csao-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/shashank-tripathi/csao-recommender/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a professional-grade **Cart Super Add-On (CSAO)** recommendation engine. It is architected for **sub-200ms P99 latency** and high horizontal scalability.

## 🏗️ System Design Deep-Dive

### 1. The Two-Stage Paradigm
Standard industry practice (Uber Eats, Pinterest) dictates that scoring every item in a large catalog is computationally prohibitive for a 300ms budget.
- **Recall (Stage 1)**: We use **Vector Search** on item embeddings. This treats recommendation as a geometry problem, fetching the top 50 "nearby" items in a high-dimensional space.
- **Precision (Stage 2)**: A **Triple-Stacked Ensemble** scores these 50 candidates. We use **LambdaRank** because it optimizes the list gradient directly (NDCG) rather than just independent point probabilities.

### 2. Engineering Motivators
- **FastAPI / Pydantic**: Eliminates manual schema handling. By enforcing strict schemas at the entry point, we guarantee data integrity throughout the internal pipeline.
- **Z-Score Normalization**: Essential for ensembling. LightGBM, XGBoost, and CatBoost have different score distributions. Without standardization, the model with the highest variance would unfairly dominate the rank.
- **Multi-Stage Docker**: Standard for senior devops. Keeping build dependencies out of the final image reduces the attack surface and ensures binary compatibility across environments.

## 🛠️ Tech Stack & CI/CD
- **Inference**: FastAPI, Pydantic, CatBoost, XGBoost, LightGBM
- **Infrastructure**: Docker (Multi-stage), GitHub Actions
- **Quality**: Pytest (Unit/E2E), Ruff (Lint), Mypy (Types), Locust (Load Test)

## 🚀 Deployment
```bash
# Build production image
docker build -t csao-recsys:latest .

# Run performance suite
locust -f tests/load_test.py
```
