# YouTube Comment Analyzer - End-to-End MLOps Pipeline

An AI-powered YouTube Comment Analysis platform that automates sentiment analysis, comment insights generation, monitoring, and MLOps workflows using FastAPI, Machine Learning, Feature Stores, Experiment Tracking, and Monitoring tools.

## Overview

The YouTube Comment Analyzer is an end-to-end MLOps project designed to analyze YouTube comments, classify sentiment, generate insights, and demonstrate a production-ready machine learning workflow.

The system includes:

* YouTube comment extraction
* Sentiment analysis using Machine Learning
* Text embedding generation
* Feature Store integration
* Experiment tracking
* Data validation
* Data drift monitoring
* API serving
* Docker containerization
* Prometheus monitoring
* Grafana dashboards

## Features

* YouTube comment collection
* Sentiment classification (Positive / Negative)
* Sentence Transformer embeddings
* Word Cloud generation
* Feature Store using Feast
* Data validation using Great Expectations
* Data versioning using DVC
* Experiment tracking using MLflow
* Data drift monitoring using Evidently AI
* REST API using FastAPI
* Dockerized deployment
* Prometheus metrics collection
* Grafana monitoring dashboards
* End-to-end MLOps workflow

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* Logistic Regression
* XGBoost
* Sentence Transformers
* Hugging Face Models

### MLOps

* MLflow
* DVC
* Feast
* Great Expectations
* Evidently AI

### Monitoring

* Prometheus
* Grafana

### Data Processing

* Pandas
* NumPy

### Deployment

* Docker

## Project Structure

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

├── youtube_feature_store/
│   └── feature_repo/

├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## MLOps Pipeline

```text
Data Collection
        ↓
Data Validation (Great Expectations)
        ↓
Data Versioning (DVC)
        ↓
Feature Engineering
        ↓
Feature Store (Feast)
        ↓
Model Training
        ↓
Experiment Tracking (MLflow)
        ↓
Model Serving (FastAPI)
        ↓
Docker Containerization
        ↓
Drift Monitoring (Evidently AI)
        ↓
Metrics Collection (Prometheus)
        ↓
Visualization (Grafana)
```

## Monitoring Dashboard

The project includes a complete monitoring stack:

* Prediction Count Monitoring
* API Request Monitoring
* Model Usage Analytics
* Prometheus Metrics Collection
* Grafana Visualization Dashboards

## Future Enhancements

* GitHub Actions CI/CD
* AWS Deployment
* Kubernetes Deployment
* Automated Model Retraining
* Multi-Class Sentiment Analysis
* Real-Time Streaming Analytics

