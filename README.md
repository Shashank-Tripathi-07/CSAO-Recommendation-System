# 🚀 CSAO: The Intelligent Cart Super Add-On Engine

> **"Solving the 300ms Cross-Sell Problem with Professional-Grade ML"**

## 🌟 The Problem: Why Does This Project Exist?
In modern e-commerce and food delivery apps (like Zomato), the "Checkout" moment is critical. Users are about to pay, and a timely, relevant suggestion can significantly increase the **Average Order Value (AOV)**.

However, recommending the *right* add-on (like Raita with Biryani or Fries with a Burger) is hard because:
1.  **Scale**: You might have 100,000+ items. You can't score all of them in a split second.
2.  **Latency**: Users won't wait. You have exactly **300ms** to provide a recommendation before they pay and leave.
3.  **Relevance**: If the recommendation is generic, users ignore it. It must feel personal and context-aware.

---

## 🛠️ The Solution: A Two-Stage Recommendation Architecture
This project implements a professional **Cart Super Add-On (CSAO)** microservice. It uses the same architectural pattern used by industry giants like Uber Eats, Pinterest, and Amazon.

### 1. Stage 1: The "Quick Scan" (Vector Retrieval)
We don't look at all 100,000 items. Instead, we use **Vector Search**. We treat every item as a mathematical point in a high-dimensional space.
*   **How it works**: When you add an item to your cart, we instantly find the top 50 "nearest" items that "mathematically" belong together.
*   **Result**: 100,000 items → 50 relevant candidates in **<10ms**.

### 2. Stage 2: The "Expert Verdict" (Triple-Stacked Ensemble)
Once we have 50 candidates, we need to be *sure* about the ranking. We pass these 50 items through an **Ensemble of 3 Machine Learning Models**:
- **LightGBM**: Fast and efficient with session data.
- **XGBoost**: Excellent at capturing subtle correlations.
- **CatBoost**: Masters categorical data (like Cuisines and Areas) without complex preprocessing.

By "stacking" these models, we average out individual biases and deliver a hyper-accurate, sorted list of recommendations.

---

## 🏗️ Technical Infra at a Glance
```text
      [ USER REQUEST ]
             |
             v
    +-----------------+
    |   FastAPI       | <--- Clean Entry Point (Validation & Documentation)
    +-----------------+
             |
    +-----------------------+     +-----------------------+
    | Stage 1: Retrieval     |     |  Vector Space Index   |
    | (Recall / Vector NN)  | <--> |  [Semantic Search]    |
    +-----------------------+     +-----------------------+
             |
    +-----------------------+     +-----------------------+
    | Stage 2: Ranking       |     |   ML Ensemble Tier    |
    | (Precision / Lambda)  | <--- |   [LGBM + XGB + CB]   |
    +-----------------------+     +-----------------------+
             |
      [ FINAL RECOMMENDATIONS ]
```

---

## 🎓 For Learners
If you are a student or a developer looking to understand how production-grade ML systems work, we have prepared a dedicated **[Detailed Learning Guide (for-learn.md)](file:///d:/Zomato%20Datathon/csao_recommendation/for-learn.md)**.

In that file, we explain:
- **Low-level Feature Engineering**: How we turn raw text into numbers.
- **Ensemble Normalization**: How we mathematically blend 3 different ML models.
- **Production DevOps**: CI/CD pipelines, Dockerization, and Load Testing.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose

### 2. Fast API Serve
```bash
# Start the production-ready microservice
docker-compose up --build
```
The API will be available at `http://localhost:8000`. You can explore the interactive documentation (Swagger UI) at `/docs`.

### 3. Training & Evaluation
```bash
# Generate synthetic data and train the ensemble
python src/csao/training/train.py

# Evaluate the performance (NDCG@10, Precision)
python src/csao/training/evaluate.py
```

---
**CSAO Recommendation System** — *Built for performance, engineered for accuracy, designed for learning.* 🚀
