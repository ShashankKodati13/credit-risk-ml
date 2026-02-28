# Credit Risk Prediction API (Production ML System)

## Overview

This project is an end-to-end Machine Learning system that predicts customer credit default risk using financial behavior data and serves real-time predictions via a deployed API.

It simulates a real-world fintech risk scoring system used in loan approval pipelines.

---

## Problem Statement

Financial institutions face major losses due to loan defaults.

The objective of this system was to:

- Predict probability of customer default
- Minimize False Negatives (high-risk customers classified as safe)
- Provide explainability for decision transparency

---

## System Architecture

Raw Data → Feature Engineering → Model Training → Evaluation
→ Model Packaging → API Layer → Cloud Deployment → Explainability

---

## Dataset

**UCI Credit Card Default Dataset**

- 30,000 customer records  
- 23 financial & behavioral features  
- Includes repayment history and bill/payment patterns  

---

## Model Development

| Model | ROC-AUC | Recall (Default Class) |
|-------|--------|------------------------|
| Logistic Regression | 0.72 | 0.61 |
| Random Forest | 0.78 | 0.67 |
| XGBoost (Final) | **0.81** | **0.72** |

---

## Business-Oriented Optimization

Instead of accuracy, the system was optimized for:

Recall of defaulters

Final model:

- Achieved **72% recall** of high-risk customers  
- Improved detection of defaulters by **18% vs baseline Logistic Regression**  
- Maintained strong ROC-AUC of **0.81**

This reflects real-world banking priorities where:

Missing a defaulter = financial loss

---

## Explainability Layer

Integrated SHAP for prediction transparency.

The system provides:

- Feature-level contribution
- Individual prediction reasoning
- Interpretability support for decision-making

Key insights observed:

- Payment delays strongly increase default risk  
- Higher credit limits reduce predicted risk  
- Recent repayment behavior is a major signal  

---

## API Deployment

Production-ready inference API built using:

- FastAPI
- Docker
- Cloud Deployment (Render)

Capabilities:

- Real-time prediction
- Input validation
- Explainable output
- Containerized deployment

---

## System Performance

| Metric | Value |
|--------|------|
| ROC-AUC | 0.81 |
| Recall (Defaulters) | 0.72 |
| Precision | 0.63 |
| Inference Time | <100 ms |
| Deployment | Live |

---

## Tech Stack

- Python  
- Scikit-learn  
- XGBoost  
- SHAP  
- FastAPI  
- Docker  
- Render  

---

## Real-World Relevance

This system mirrors production fintech ML workflows:

- Risk-sensitive evaluation  
- Explainability for compliance  
- Containerized deployment  
- Cloud inference serving  

---

## Author

**Shashank Kodati**  
Machine Learning Engineer
