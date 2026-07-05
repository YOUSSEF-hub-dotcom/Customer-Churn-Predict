# 🚀 Customer Churn Prediction Platform

An end-to-end **customer churn prediction platform** built with **CatBoost, MLflow, FastAPI, and Streamlit**.
The project is designed to predict churn risk, support proactive retention strategies, and demonstrate a **production-oriented ML workflow** covering experimentation, model lifecycle management, API serving, and business-facing analytics.

---

## 📌 Project Overview

Customer churn is one of the most important business problems in subscription-based services. Losing customers early reduces customer lifetime value, increases acquisition pressure, and directly impacts recurring revenue.

This project tackles churn prediction as a **binary classification problem**:

* **0 → Stayed**
* **1 → Churned**

Using customer demographics, service subscriptions, contract details, payment behavior, and billing information from the **Telco Customer Churn dataset**.

The platform goes beyond model training and includes a full workflow for:

* **data preparation and feature engineering**
* **exploratory and business-focused analysis**
* **CatBoost model training with validation**
* **threshold optimization for business-aware decisions**
* **MLflow experiment tracking and model registry**
* **FastAPI deployment for inference**
* **Streamlit dashboard for interactive monitoring and usage**

---

## 🎯 Objectives

This project was built to answer a practical business question:

> **Can we identify customers who are likely to churn early enough to support proactive retention actions?**

### The platform is designed to:

* Predict customer churn probability
* Prioritize high-risk customers for retention campaigns
* Support decision-making with **calibrated churn risk scores**
* Provide an end-to-end ML workflow with **tracking, packaging, and deployment**
* Demonstrate a **production-oriented architecture** rather than a notebook-only model

---

## 🧠 End-to-End System Architecture

The solution is organized as a full ML application stack:

```text id="0a3q2n"
User / Analyst
      ↓
Streamlit Dashboard
      ↓
FastAPI Prediction API
      ↓
MLflow Registered Model
      ↓
CatBoost Classifier
      ↓
Prediction Logging / Monitoring
```

### Pipeline flow

```text id="gq6g57"
Raw Data
   ↓
Data Cleaning & Validation
   ↓
EDA + Business Insight Discovery
   ↓
Feature Engineering
   ↓
CatBoost Training + Cross Validation
   ↓
Threshold Optimization
   ↓
MLflow Tracking + Model Registry
   ↓
FastAPI Inference Service
   ↓
Streamlit Dashboard + Monitoring
```

---

## 📊 Dataset

**Dataset:** Telco Customer Churn
**Task:** Binary classification
**Target variable:** `Churn`

### Example feature groups

* **Customer profile:** gender, senior citizen, partner, dependents
* **Contract & tenure:** contract type, tenure, paperless billing
* **Services:** internet service, tech support, online security, device protection
* **Billing & payments:** monthly charges, total charges, payment method

---

## 🧹 Data Pipeline & Feature Engineering

The preprocessing pipeline prepares the raw Telco data for modeling.

### Core preprocessing steps

* Data loading and validation
* Duplicate and missing-value checks
* `TotalCharges` conversion to numeric
* Binary mapping for yes/no style fields
* Distribution analysis and skewness handling
* Outlier inspection
* Feature creation for business-relevant customer behavior

### Engineered features

* **`NumServices`** → number of subscribed services
* **`TechSupport_OnlineSecurity`** → combined support/protection signal

### Why this matters

The project does not treat churn prediction as “fit a model on raw columns.”
Instead, it adds **domain-aware features** that reflect customer stickiness, service depth, and protection/support behavior — all of which are highly relevant to churn risk.

---

## 📈 Exploratory Data Analysis

The project includes both **basic EDA** and **business-oriented analytical EDA**.

### Examples of key churn patterns identified

* **Month-to-month contracts** have the highest churn rates
* **New customers (0–12 months)** are significantly more likely to churn
* **Fiber customers with high charges** represent a critical high-risk segment
* Customers without **Tech Support** or **Online Security** churn much more often
* **Electronic check** users churn more than customers using auto-pay methods
* High monthly charges combined with short tenure create one of the riskiest customer segments

These findings help justify both the modeling approach and the downstream retention recommendations.

---

## 🤖 Machine Learning Model

### Model

**CatBoostClassifier**

CatBoost was selected because it is a strong fit for churn modeling with mixed tabular features:

* handles categorical patterns effectively
* performs well on structured business data
* supports robust probability-based ranking
* provides strong performance without overly complex preprocessing

### Training setup

* **Stratified 5-Fold Cross Validation**
* **Class imbalance handling**
* **Early stopping**
* **Threshold optimization**
* **Feature importance analysis**
* **Calibration / probability analysis**

---

## 📊 Final Model Performance

The optimized production-oriented version achieved the following results:

| Metric                |      Score |
| --------------------- | ---------: |
| **Accuracy**          | **76.79%** |
| **AUC**               | **84.66%** |
| **F1 Score**          | **63.79%** |
| **Recall (Churn)**    | **77.01%** |
| **Precision (Churn)** | **54.44%** |
| **CV AUC Mean**       | **84.85%** |
| **CV AUC Std**        |  **1.09%** |

### Interpretation

* The model identifies **~77% of churners**
* AUC indicates strong ranking ability for churn risk prioritization
* Cross-validation stability suggests the model is relatively consistent across splits
* Threshold tuning makes the model more useful for **business intervention workflows**, not just raw classification

---

## 🎯 Business-Oriented Thresholding

Instead of relying only on a fixed `0.5` decision threshold, the project supports **risk-based segmentation**.

### Example risk tiers

* **Critical Risk** → immediate intervention
* **High Risk** → retention campaign / account manager alert
* **Medium Risk** → closer monitoring and engagement
* **Low Risk** → standard retention treatment

This makes the output more actionable for business teams because the model becomes a **customer prioritization tool**, not just a binary predictor.

---

## 🔍 Key Business Insights from the Analysis

A major strength of this project is that it connects modeling results to business action.

### Examples of high-risk segments discovered

* **0–12 months + high monthly charges**
* **Fiber optic customers with medium/high charges**
* **Month-to-month customers**
* **Electronic check users**
* Customers without **Tech Support + Online Security**

### Examples of retention signals

* Long-term contracts strongly reduce churn risk
* Security/support bundles improve retention
* Auto-pay methods correlate with lower churn
* Long-tenure customers are much more stable than new customers

---

## 🔄 MLflow Lifecycle Management

The project includes an MLflow-based lifecycle workflow to make experimentation and deployment more reproducible.

### MLflow components used

* **Experiment tracking**
* **Parameter logging**
* **Metric logging**
* **Artifact logging**
* **Model signature inference**
* **PyFunc packaging**
* **Model registry**
* **promotion workflow based on quality criteria**

### Example quality gate idea

A model can be promoted only if it satisfies performance criteria such as:

* minimum **AUC**
* minimum **Recall**
* acceptable cross-validation stability

This turns the project into more than just a trained model — it becomes a **managed ML workflow**.

---

## 🌐 FastAPI Inference Layer

The project exposes the model through a **FastAPI prediction service**.

### API responsibilities

* load the registered MLflow model
* receive customer data for scoring
* return churn probability and prediction outputs
* support prediction logging and monitoring workflows
* provide a reusable inference interface for external systems

### Example responsibilities included in the project

* input validation
* prediction endpoints
* logging / observability support
* API-serving layer separated from training code

---

## 🖥 Streamlit Dashboard

A Streamlit dashboard is included to make the solution easier to consume by analysts or business users.

### Dashboard goals

* score customers interactively
* visualize churn risk
* inspect churn distributions and segments
* expose model outputs in a more business-friendly interface

This adds an important “last mile” layer: the project is not only trainable and deployable, but also **presentable and usable**.

---

## 🛠️ Tech Stack

| Layer                | Technology                      | Purpose                            |
| -------------------- | ------------------------------- | ---------------------------------- |
| Modeling             | **CatBoost**                    | Customer churn prediction          |
| Data                 | **Pandas, NumPy**               | Data preparation and analysis      |
| Visualization        | **Matplotlib, Seaborn, Plotly** | EDA and dashboard visuals          |
| MLOps                | **MLflow**                      | Tracking, registry, packaging      |
| API                  | **FastAPI**                     | Prediction service                 |
| Frontend             | **Streamlit**                   | Interactive dashboard              |
| Validation / Serving | **Pydantic, Uvicorn**           | API validation and serving         |
| ML Utilities         | **Scikit-learn**                | metrics, CV, preprocessing helpers |

---

## 📁 Project Structure

```bash id="3gbx1f"
project/
│
├── data_pipeline.py          # preprocessing + feature engineering
├── basic_eda.py             # foundational exploratory analysis
├── advanced_eda.py          # business-focused analytical EDA
├── model.py                 # CatBoost training and evaluation
├── mlflow_lifecycle.py      # experiment tracking + registry workflow
│
├── api.py                   # FastAPI inference service
├── app.py                   # Streamlit dashboard
│
├── MLproject                # MLflow project definition
├── conda.yaml               # reproducible environment
├── README.md
└── docs/                    # business / technical documentation
```

> Update the structure section if your repository contains additional folders such as `artifacts/`, `notebooks/`, `logs/`, or `models/`.

---

## 🏁 How to Run

### 1) Train the model with MLflow

```bash id="f7vfhs"
mlflow run . -P iterations=1500 -P learning_rate=0.03 -P depth=6
```

### 2) Start the FastAPI service

```bash id="5gx6w4"
uvicorn api:app --reload
```

### 3) Launch the Streamlit dashboard

```bash id="17n8wq"
streamlit run app.py
```

---

## 📌 What Makes This Project Strong as a Portfolio Project

This repository demonstrates more than a churn model. It shows an end-to-end workflow that combines:

* **business understanding of churn drivers**
* **feature engineering on tabular customer data**
* **CatBoost-based predictive modeling**
* **cross-validation and threshold optimization**
* **MLflow experiment and model lifecycle management**
* **API deployment with FastAPI**
* **dashboard delivery with Streamlit**

In other words, it reflects a **full applied machine learning workflow** from raw data to deployable prediction service.

---

## 🌱 Future Improvements

Potential next steps for extending the project:

* add batch scoring endpoints and scheduled churn scoring jobs
* integrate a real database-backed prediction store
* add model drift and data drift monitoring
* include SHAP explanations for prediction transparency
* connect the system to CRM-style retention workflows
* build next-best-action recommendations on top of churn scores
* add CI/CD for model validation and deployment automation

---

## 📚 Documentation

This repository is supported by project documentation that covers:

* churn analysis findings
* segment-level business insights
* model evaluation details
* MLflow lifecycle workflow
* deployment design and operational recommendations

---

## 👨‍💻 Author

**Youssef Mahmoud**
AI / Data Science Student

[LinkedIn](https://www.linkedin.com/in/youssef-mahmoud-63b243361)

---

## ⭐ Final Note

This project is designed as an **end-to-end churn prediction platform** rather than a single notebook model.
It combines **machine learning, MLOps, API deployment, and business-facing analytics** to demonstrate how churn prediction can be turned into a practical, deployable system for proactive customer retention.

