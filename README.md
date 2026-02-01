# 🚚 Scalable Truck Delay Prediction System
## Spark ML + GPU-Accelerated XGBoost + Production MLOps on AWS

Predict truck delivery delays at scale using distributed Spark ML, GPU-accelerated XGBoost, and a fully automated MLOps pipeline with monitoring, drift detection, and CI/CD.

---

## 📊 Latest Model Performance

| Metric | Score |
|---|---|
| **AUC** | **0.81** |
| **Precision** | **0.776** |
| **Recall** | **0.729** |
| **F1-Score** | **0.752** |


<img width="1550" height="575" alt="Screenshot 2026-02-01 100158" src="https://github.com/user-attachments/assets/115421f1-f34b-4f91-be9b-f5fd51f6f0aa" />
<img width="598" height="88" alt="Screenshot 2026-02-01 101639" src="https://github.com/user-attachments/assets/c364a97d-004c-471c-9bde-5d4d2b5ce7db" />

---

## 📌 Problem Statement

Logistics companies face significant losses due to unexpected shipment delays.  
This project builds a scalable ML system to predict whether a truck will be delayed **before dispatch**, enabling proactive routing, scheduling, and SLA management.

**Challenges addressed:**

- Millions of rows of tabular logistics data
- Imbalanced binary classification
- Need for distributed training
- Requirement for production monitoring
- Model drift from seasonality, routes, and weather patterns

---

## 🏗️ System Architecture

```
Raw Data (AWS S3)
        ↓
Spark ETL & Feature Engineering
        ↓
XGBoost4J-Spark + NVIDIA RAPIDS (GPU)
        ↓
Model Evaluation
        ↓
SageMaker Pipeline (CI/CD)
        ↓
Production Endpoint
        ↓
Evidently AI Monitoring & Drift Alerts
```

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| Distributed Processing | Apache Spark ML |
| Model Training | XGBoost4J-Spark |
| GPU Acceleration | NVIDIA RAPIDS Accelerator |
| Cloud | AWS S3, SageMaker, CloudWatch |
| MLOps | SageMaker Pipelines, CI/CD |
| Monitoring | Evidently AI |
| Language | Python, PySpark |

---

## 🧠 Model Training Approach

- Spark ML pipelines for distributed feature engineering
- XGBoost4J-Spark for distributed gradient boosting
- RAPIDS Accelerator to offload training to NVIDIA GPUs
- Class imbalance handled with weighted loss and threshold tuning
- Hyperparameter tuning via Spark cross-validation

This setup significantly reduced training time while improving predictive performance.

---

## 🔁 MLOps & Productionization

This project goes beyond model training.

### CI/CD with SageMaker Pipelines

- Automated retraining on new data
- Model validation and conditional deployment
- Versioned model artifacts

### Monitoring with Evidently AI

- Data drift detection
- Prediction drift tracking
- CloudWatch alerts when drift exceeds thresholds

### Automated Alerts

If model performance degrades or data distribution shifts, the retraining pipeline is triggered automatically.

---

## 📂 Repository Structure

```
├── spark_jobs/          # ETL & feature engineering scripts
├── training/            # GPU XGBoost training code
├── pipelines/           # SageMaker pipeline definitions
├── monitoring/          # Evidently drift reports
├── images/              # ROC curve and diagrams
└── README.md
```

---

## 🚀 How to Run

### 1. Run Spark ETL

```
spark-submit spark_jobs/etl.py
```

### 2. Train GPU-Accelerated XGBoost

```
spark-submit training/train_xgboost_gpu.py
```

### 3. Trigger SageMaker Pipeline

```
python pipelines/run_pipeline.py
```

---

## 🎯 Business Impact

This system enables logistics teams to:

- Predict delays before dispatch
- Optimize routing decisions
- Reduce late deliveries
- Improve SLA compliance
- Monitor model health in real time
