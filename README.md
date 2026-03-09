# 🚀 High-Performance CSAO Recommendation System

> [!IMPORTANT]
> **New to Recommendation Systems?** Checkout our [Detailed Learning Guide](file:///d:/Zomato%20Datathon/csao_recommendation/for-learn.md) for a deep-dive into the low-level implementation, feature engineering, and ensemble blending logic used in this project.

## 🌟 Introduction
This repository contains a **production-grade Cart Super Add-On (CSAO) Recommendation System**. Unlike simple toy projects, this microservice is architected for **sub-200ms P99 latency** and high horizontal scalability, using the same "Two-Stage Retrieval & Ranking" paradigm employed by industry leaders like Uber Eats and Pinterest.

We combine **Vector Search** for massive retrieval with a **Triple-Stacked GBDT Ensemble** (LightGBM + XGBoost + CatBoost) to deliver hyper-relevant recommendations at the point of checkout.

[![CI](https://github.com/shashank-tripathi/csao-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/shashank-tripathi/csao-recommender/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
