# CSAO Rail Recommendation System

> [!IMPORTANT]
> **Hackathon Judges**: For a deep-dive into the technical architecture, mathematical foundations, and detailed evaluation benchmarks, please refer to the following documents:
> - **[Solution Report (Architecture & Math)](solution_report.md)**: Details on the two-stage ranking system and LambdaRank objective.
> - **[Test Results (Benchmarks)](test_result.md)**: Full breakdown of LightGBM vs XGBoost vs Stacked Ensemble scores.

## Project Overview
This project implements an intelligent Cart Super Add-On (CSAO) recommendation rail for a food delivery platform. It is designed to suggest relevant add-on items in real-time (<300ms latency) as customers build their carts, aiming to increase Average Order Value (AOV) and add-on acceptance rates. 

## Design Decisions
1. **Synthetic Data Generation**: Because the public Zomato dataset lacks transaction logs, we synthesized ~1M cart sessions using a domain-knowledge rules engine (incorporating cuisine-to-item hierarchical mappings, complementary probabilities, and price-tier bounds). This provides a rich dataset to train and evaluate Learning-To-Rank (LTR) algorithms.
2. **Two-Stage Architecture**: 
    - **Stage 1 (Candidate Generation)**: Heuristics based on menu availability and meal completeness rules to rapidly fetch ~50 candidates.
    - **Stage 2 (Ranking)**: A LightGBM `lambdarank` model. Tree-based LTR handles tabular categorical/continuous feature interactions well, has blazing fast inference speeds, and natively optimizes for NDCG.
3. **Feature Engineering Strategy**: Features were divided into cart-level context, candidate relative metrics, restaurant tier indicators, and temporal signals. 
4. **Post-Ranking Diversity Filter**: We cap recommendations at a maximum of 2 items per category (e.g., max 2 sides, 2 desserts) to ensure visual variety in the UI and avoid cannibalization.
5. **Strict Latency Budget**: The pipeline leverages pre-fitted `scikit-learn` transformation paths, dictionary lookups for catalogs/restaurant features, and in-memory LightGBM inference to consistently resolve far below the 300ms SLA.

## Limitations of Synthetic Data
**CRITICAL LIMITATION**: This model is trained entirely on synthetic heuristics.
- The offline metrics (`NDCG`, `Precision`) reported during evaluation merely reflect how well the LightGBM model successfully reverse-engineered our synthetic probability rules, **not actual human behavior**. 
- The popularity distributions and complementary affinities are mocked based on broad domain knowledge.
- **Production Path**: In a live environment, collect real add-to-cart implicit feedback logs (clicks, add, ignore) for 2 to 4 weeks. Swap the synthetic `train.csv` with these authentic logs, retrain the `FeaturePipeline` and LightGBM model, and deploy the new artifact.

## Business Impact Projection
Assuming a successful transition to real-world logs mirroring our high synthetic NDCG scores:
- **Acceptance Rate Improvement**: Moving from a heuristic popularity baseline to personalized `lambdarank` typically lifts add-on acceptance by 15-25%. 
- **AOV Lift**: If 10% of users accept an average ₹250 add-on on a ₹1,500 base cart, overall AOV will conservatively lift by ~1.6%.
- **Cart-to-Order Rate**: Presenting highly relevant complements (like drinks with a dry meal) reduces cart abandonment, yielding an estimated +0.5% conversion rate jump at checkout.
