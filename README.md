# Recommendation_Model

# 🎬 Feature-Enriched Neural Movie Recommendation System

## 📽 Overview
This project implements a **hybrid deep learning recommendation system** that predicts user ratings for movies by combining collaborative filtering signals with rich user and movie metadata. It builds on a **two-tower neural architecture** inspired by systems used at YouTube and Google Play.Rather than relying solely on user and item IDs, this model integrates **structured features** such as user age, occupation, average rating behavior, movie genres, and release year — enabling **personalized, content-aware recommendations**, even in cold-start scenarios.

---


## 🧠 Key Features 
  - **Two-Tower Neural Architecture**
  Separate user and movie towers encode inputs into dense vectors for scalable representation learning.
  - **Collaborative + Content Hybrid**
  Combines user-item interactions with side features for improved accuracy and cold-start performance.
  - **Embedding-Based Learning**
  Learns latent user and item preferences via trainable embeddings.
  - **Custom Feature Engineering**
    - One-hot encoded genres and occupations
    -  Normalized age and average user rating
    -   Movie release year    -
    -   User rating standard deviation and favorite genre
  - **Deep Interaction Modeling**
    Dense layers allow for non-linear feature interaction modeling beyond simple dot products.
  - **Evaluation Metrics**
    Optimized with MSE, with plans to expand to ranking-based metrics like Recall@K and NDCG.

    ---
## 📊 Model Architecture


User Tower Inputs:
  - User ID (embedded)
  - Age
  - One-hot Occupation
  - Avg Rating

Movie Tower Inputs:
  - Movie ID (embedded)
  - One-hot Genre
  - Release Year

Combined Vector → Dense(128) → ReLU → Dense(64) → ReLU → Output (Predicted Rating)

---
