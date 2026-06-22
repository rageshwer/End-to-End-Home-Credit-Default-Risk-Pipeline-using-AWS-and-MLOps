#  Home Credit Risk Scoring Portal
### End-to-End Enterprise Credit Risk Modeling, Explainable AI & MLOps Platform

An enterprise-grade credit risk scoring platform that predicts **Probability of Default (PD)** for retail lending portfolios while providing **real-time explainability**, **portfolio-level analytics**, and **production-ready deployment infrastructure**.

The project was built using the Home Credit Default Risk datasets (Application, bureau and bureau balance) and demonstrates the complete lifecycle of a modern credit risk system—from raw data engineering and model development to explainable AI, cloud deployment, containerization, and CI/CD automation.

---

##  Business Problem

Financial institutions must accurately assess the likelihood of borrower default before approving loans.

Traditional scorecards often lack:

- Real-time prediction capabilities
- Transparency into model decisions
- Scalable deployment architecture
- Automated model lifecycle management

This project addresses these challenges by building a production-grade risk scoring system capable of:

- Predicting Probability of Default (PD)
- Providing analyst-friendly explanations
- Monitoring portfolio-level risk drivers
- Supporting scalable cloud deployment

---

##  Project Highlights

### Credit Risk Modeling
- Built a Probability of Default (PD) prediction engine using **LightGBM**
- Trained on Home Credit financial datasets
- Optimized model performance from **AUC 0.73 → 0.779**
- Achieved **KS Statistic: 41%**
- Generated portfolio-level risk predictions for thousands of applicants

### Explainable AI (XAI)
Implemented real-time model interpretability using SHAP values:

#### Individual-Level Explainability
- Waterfall Plot
- Risk Contribution Analysis
- Baseline vs Final Prediction Path
- Feature-Level Risk Drivers

#### Portfolio-Level Explainability
- SHAP Beeswarm Analysis
- Global Feature Importance
- SHAP Dependence Plots
- Feature Interaction Analysis

### Production Engineering
- FastAPI Backend
- Streamlit Frontend
- Dockerized Services
- AWS Deployment
- Automated CI/CD
- MLflow Experiment Tracking

---

#  System Architecture

```text
                     ┌──────────────────────┐
                     │   Streamlit Frontend │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │    FastAPI Backend   │
                     └──────────┬───────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼

      LightGBM Model      SHAP Engine       Portfolio Analytics

             ▲
             │
             ▼

       AWS S3 Model Store

             ▲
             │
             ▼

     Dockerized EC2 Deployment (Images automatically deployed to ECR)
```

---

#  Dataset Overview

Dataset Source:

- Home Credit Default Risk Competition
- Kaggle

Core datasets used:

| Dataset | Description |
|----------|-------------|
| Application | Customer demographic and loan application information |
| Bureau | Historical credit bureau records |
| Bureau Balance | Monthly repayment status of previous loans |

### Scale

- 307,511 Training Customers
- 48,744 Test Customers
- Millions of historical bureau records
- 1,000+ engineered features

---

#  Feature Engineering Pipeline

A comprehensive feature engineering framework was developed to transform raw financial data into predictive risk signals.

## Application Dataset Features

Created financial ratios and behavioral indicators:

- Credit-to-Income Ratio
- Annuity-to-Income Ratio
- Income-to-Age Ratio
- Goods-to-Credit Ratio
- Employment Stability Metrics
- External Risk Source Aggregations

### External Source Features

Generated advanced statistical combinations:

- EXT Mean
- EXT Median
- EXT Maximum
- EXT Minimum
- EXT Standard Deviation
- EXT Multiplication Features
- EXT Range Features

---

## Bureau Dataset Aggregations

Created customer-level bureau intelligence including:

### Active Loan Features
- Active loan count
- Active debt exposure
- Active overdue amount
- Active credit utilization

### Closed Loan Features
- Historical repayment quality
- Closed loan exposure
- Loan completion patterns

### Loan-Type Specific Aggregations

Engineered risk features across:

- Consumer Loans
- Credit Cards
- Mortgages
- Car Loans
- Microfinance Loans

---

## Bureau Balance Features

Generated repayment behavior indicators:

- Maximum Delinquency
- Average DPD
- Recent Payment Status
- NPA Flags
- Monthly Trend Features
- Loan Aging Metrics
- Rolling Statistics

---

## Final Dataset

| Metric | Value |
|---------|---------|
| Training Rows | 307,511 |
| Test Rows | 48,744 |
| Engineered Features | 1,000+ |

---

#  Model Development

## Algorithm

### LightGBM

Chosen because of:

- Superior performance on tabular financial data
- Native categorical feature support
- Fast training and inference
- High interpretability through SHAP

---

## Evaluation Strategy

- Stratified Cross Validation
- Out-of-Fold Predictions
- ROC-AUC Evaluation
- KS Statistic Monitoring

---

## Results

| Metric | Score |
|----------|---------|
| ROC-AUC | 0.779 |
| KS Statistic | 41% |
| Model Type | LightGBM |
| Target | Probability of Default |

### Performance Improvement

| Stage | AUC |
|---------|---------|
| Initial Baseline | 0.730 |
| Optimized Model | 0.779 |

Improvement achieved through:

- Advanced feature engineering
- Bureau intelligence aggregation
- Hyperparameter optimization
- Feature selection
- Extensive experiment tracking

---

#  Experiment Tracking with MLflow

All experiments were tracked using MLflow.

Tracked artifacts:

- Model Versions
- Hyperparameters
- Metrics
- Feature Importance

Benefits:

- Reproducibility
- Version Control
- Experiment Comparison
- Model Governance

---

#  Explainable AI Dashboard

The platform includes a complete explainability layer.

---

## 1. Executive Summary

Provides:

- Probability of Default
- Approval / Decline Recommendation
- Prediction Status
- Risk Classification

---

## 2. Individual Risk Diagnostics

### Waterfall Plot

Explains:

- Baseline Risk
- Positive Contributors
- Negative Contributors
- Final Prediction Path

Enables analysts to understand exactly why a prediction was generated.

---

## 3. Global Portfolio Analytics

### SHAP Beeswarm Plot

Shows:

- Global Feature Importance
- Direction of Impact
- Portfolio-Wide Risk Drivers

### SHAP Dependence Plot

Displays:

- Non-linear Feature Relationships
- Interaction Effects
- Risk Threshold Analysis

---

#  Frontend Application

Built using **Streamlit**.

Features:

- Interactive Dashboard
- Real-Time Scoring
- SHAP Visualizations
- Portfolio Analytics
- Risk Recommendation Engine
- Analyst-Friendly Interface

---

#  Backend API

Built using **FastAPI**.

Capabilities:

- High-performance asynchronous APIs
- Real-time prediction endpoints
- SHAP explanation endpoints
- Portfolio analytics endpoints
- Scalable microservice architecture

---

#  AWS Deployment

Deployed on AWS using a containerized architecture.

### Infrastructure

- AWS EC2
- AWS S3
- Amazon ECR
- Docker Networking

### Deployment Flow

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Amazon ECR
   │
   ▼
AWS EC2
   │
   ▼
Docker Containers
```

---

#  Containerization

Each application layer is isolated inside Docker containers.

## Containers

### Frontend Container
- Streamlit UI

### Backend Container
- FastAPI Service

### Model Layer
- LightGBM Model
- SHAP Artifacts

### Storage Layer
- AWS S3

Benefits:

- Reproducibility
- Scalability
- Isolation
- Simplified Deployment

---

#  CI/CD Pipeline

Implemented automated DevOps workflows using GitHub Actions.

---

## Continuous Integration

On every push:

- Install dependencies
- Run automated tests
- Validate feature engineering pipelines
- Validate training pipelines
- Verify model code integrity

---

## Continuous Deployment

After successful validation:

- Build Docker images
- Push images to Amazon ECR
- Deploy updated containers

---

## Automated Test Coverage

Unit tests cover:

### Feature Engineering
- Application Features
- Bureau Features
- Bureau Balance Features

### Data Pipeline
- Dataset Merging
- Final Matrix Generation

### Model Training
- Training Pipeline
- Artifact Generation

---

#  Repository Structure

```text
HOME_CREDIT_DEFAULT_RISK
│
├── api/
│   ├── main.py
│   ├── api_data/
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
├── src/
│   ├── features/
│   ├── pipeline/
│   └── training/
│
├── tests/
│
├── data/
│
├── mlruns/
│
├── models/
│
├── notebooks/
│
├── .github/workflows/
│
└── README.md
```

---

#  Tech Stack

### Machine Learning
- Python
- LightGBM
- SHAP
- Scikit-Learn

### Data Engineering
- Pandas
- NumPy

### MLOps
- MLflow
- Docker
- GitHub Actions

### Backend
- FastAPI

### Frontend
- Streamlit

### Cloud
- AWS EC2
- AWS S3
- Amazon ECR

### Testing
- PyTest

---

# Key Skills Demonstrated

### Credit Risk Analytics
- Probability of Default Modeling
- Risk Segmentation
- Feature Engineering
- Portfolio Analytics

### Machine Learning
- Gradient Boosting Models
- Explainable AI
- Model Evaluation
- Hyperparameter Optimization

### Data Engineering
- Large-Scale Data Aggregation
- Pipeline Development
- Feature Stores

### MLOps & Deployment
- CI/CD Automation
- Docker Containerization
- AWS Deployment
- Model Lifecycle Management

---

# Future Enhancements

- Real-time Feature Store
- Automated Model Retraining
- Drift Monitoring
- Credit Portfolio Monitoring Dashboard

---

## Author

**Rageshwer Singh**
- M.Sc. Data Science & Artificial Intelligence (BITS Pilani)
- Credit Risk Analytics | Data Science | Machine Learning | MLOps
