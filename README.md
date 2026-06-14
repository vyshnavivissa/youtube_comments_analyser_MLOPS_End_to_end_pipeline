#  YouTube Comment Analyzer – End-to-End MLOps Pipeline

An end-to-end MLOps platform for multilingual YouTube comment sentiment analysis, built using modern Machine Learning, MLOps, Monitoring, Data Lineage, and AI Evaluation tools.

The project demonstrates the complete lifecycle of a production-ready machine learning system—from data validation and feature engineering to deployment, monitoring, lineage tracking, drift detection, and evaluation.

---

# Overview

The YouTube Comment Analyzer automatically extracts YouTube comments, analyzes sentiment, generates insights, and provides monitoring and observability capabilities using modern MLOps tools.

The platform supports multilingual comments using transformer-based embeddings and includes experiment tracking, data versioning, data validation, drift monitoring, lineage tracking, and performance monitoring.

---

#  Features

* YouTube Comment Collection
* Multilingual Sentiment Analysis
* TF-IDF Feature Engineering
* Sentence Transformer Embeddings
* Multilingual Embedding Support
* Logistic Regression Model
* XGBoost Model
* Hyperparameter Tuning using GridSearchCV
* Model Comparison
* Word Cloud Generation
* REST API using FastAPI
* Data Validation using Great Expectations
* Data Versioning using DVC
* Experiment Tracking using MLflow
* Data Drift Monitoring using Evidently AI
* AI/LLM Evaluation using Promptfoo
* Data Lineage Tracking using OpenLineage
* Marquez Lineage Visualization
* Docker Containerization
* Prometheus Metrics Collection
* Grafana Monitoring Dashboards

---

#  Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn

## Machine Learning

* Scikit-Learn
* Logistic Regression
* XGBoost
* TF-IDF Vectorization
* Sentence Transformers
* Multilingual Embeddings
* GridSearchCV
* Hugging Face Models

## Data Processing

* Pandas
* NumPy

## MLOps

* MLflow
* DVC
* Great Expectations
* Evidently AI
* OpenLineage
* Marquez
* Promptfoo

## Monitoring

* Prometheus
* Grafana

## Deployment

* Docker

---

#  Project Structure

```text
youtube-comment-analyzer/

├── api/
│   └── main.py

├── src/
│   ├── train.py
│   ├── train_xgboost.py
│   ├── predict.py
│   ├── generate_embeddings.py
│   └── data_validation.py

├── reports/
│   └── drift_report.py

├── frontend/

├── data/

├── models/

├── mlruns/

├── lineage/

├── Dockerfile

├── requirements.txt

└── README.md
```

---

#  End-to-End MLOps Pipeline

```text
YouTube Comments
        ↓
Data Validation (Great Expectations)
        ↓
Data Versioning (DVC)
        ↓
Feature Engineering
(TF-IDF + Sentence Transformers)
        ↓
Hyperparameter Tuning
(GridSearchCV)
        ↓
Model Training
(Logistic Regression / XGBoost)
        ↓
Experiment Tracking (MLflow)
        ↓
Model Serving (FastAPI)
        ↓
Docker Containerization
        ↓
Data Lineage Tracking
(OpenLineage + Marquez)
        ↓
Drift Detection
(Evidently AI)
        ↓
AI Evaluation
(Promptfoo)
        ↓
Metrics Collection
(Prometheus)
        ↓
Visualization
(Grafana)
```

---

#  Monitoring Dashboard

The monitoring stack tracks:

* Total Predictions
* API Requests
* Positive Predictions
* Negative Predictions
* Sentiment Distribution
* Application Metrics
* Model Usage Statistics
* Prometheus Monitoring
* Grafana Dashboards

---

#  Data Lineage

The project uses OpenLineage and Marquez to track:

* Dataset Lineage
* Training Runs
* Model Dependencies
* Pipeline Metadata
* End-to-End Data Flow

This enables observability and traceability across the machine learning lifecycle.

---

# Experiment Tracking

Using MLflow:

* Hyperparameter Tracking
* Model Comparison
* Metrics Logging
* Artifact Storage
* Experiment Management

Tracked Parameters:

* TF-IDF Configuration
* Embedding Model Selection
* Logistic Regression Parameters
* XGBoost Parameters
* GridSearchCV Results

Tracked Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Cross Validation Score

---

#  Model Monitoring

Using Evidently AI:

* Data Drift Detection
* Feature Distribution Monitoring
* Model Performance Monitoring
* Dataset Comparison Reports

---

#  AI Evaluation

Using Promptfoo:

* Prompt Testing
* Output Evaluation
* AI Quality Assessment
* Regression Testing for AI Systems

---

#  Deployment

The application is containerized using Docker and can be deployed consistently across different environments.

Components:

* FastAPI Service
* Prometheus
* Grafana
* Marquez
* PostgreSQL

---

#  Business Impact

This project demonstrates production-grade MLOps practices including:

* Reproducible ML Pipelines
* Experiment Tracking
* Data Validation
* Data Versioning
* Hyperparameter Optimization
* Data Lineage
* Monitoring & Observability
* Drift Detection
* AI Evaluation
* Containerized Deployment

---

# Future Enhancements

* GitHub Actions CI/CD
* Kubernetes Deployment
* ArgoCD GitOps Deployment
* AWS Deployment
* Automated Model Retraining
* Real-Time Comment Streaming
* RAG-based YouTube Insights Assistant
* LLM-Powered Comment Summarization
* Multi-Class Sentiment Analysis

