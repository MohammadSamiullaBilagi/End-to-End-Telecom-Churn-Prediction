# 📡 Telecom Customer Churn Prediction - End-to-End MLOps

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AWS](https://img.shields.io/badge/AWS-EC2%20|%20S3%20|%20ECR-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black)

## 📋 Project Overview
This project is an end-to-end Machine Learning solution designed to predict customer churn in the telecommunications industry. The solution is built with a modular architecture, covering the entire lifecycle from Data Ingestion to Deployment. 

It implements a robust MLOps pipeline including Data Validation (with drift detection), Transformation, Model Training, Experiment Tracking, and automated deployment to AWS EC2 using Docker and GitHub Actions.

## 🔑 Key Features
*   **Modular Architecture:** Codebase organized into Config, Entity, and Components for maintainability.
*   **Data Pipeline:** Automated ETL process (Extract, Transform, Load) using **MongoDB Atlas**.
*   **Data Integrity:** Implemented `schema.yaml` validation and **Data Drift** detection to monitor distribution changes over time.
*   **Experiment Tracking:** integrated **MLflow** and **DagsHub** to track model parameters, metrics, and artifacts.
*   **CI/CD Deployment:** Fully automated deployment pipeline using **GitHub Actions**, **AWS ECR**, and **AWS EC2**.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Database:** MongoDB Atlas
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **MLOps:** MLflow, DagsHub
*   **Containerization:** Docker
*   **Cloud (AWS):** S3 (Artifacts), ECR (Container Registry), EC2 (Virtual Machine)
*   **CI/CD:** GitHub Actions

## 📊 Model Performance
The best performing model (Optimized Random Forest/Gradient Boosting) achieved the following metrics on the test set:

| Metric | Score |
| :--- | :--- |
| **AUC Score** | 0.8642 |
| **Recall** | 0.7962 |
| **F1 Score** | 0.6527 |
| **Precision** | 0.5531 |

*Note: High Recall (0.80) was prioritized to minimize False Negatives, ensuring the business identifies ~80% of at-risk customers.*

## 🏗️ Project Architecture & Workflow

The pipeline is divided into the following stages:

1.  **Data Ingestion:** 
    *   Extracts data from MongoDB Atlas.
    *   Splits data into Train/Test sets.
    *   Stores raw data in Artifacts.
2.  **Data Validation:** 
    *   Validates data against `schema.yaml`.
    *   Checks for Data Drift and generates `report.yaml`.
3.  **Data Transformation:**
    *   Handles missing values (KNN Imputer for numerical, Simple Imputer for categorical).
    *   Encodes features (OneHotEncoding).
    *   Saves `preprocessor.pkl`.
4.  **Model Trainer:**
    *   Trains multiple models (Random Forest, Decision Tree, XGBoost, etc.).
    *   Performs Hyperparameter Tuning.
    *   Logs experiments to MLflow.
    *   Saves the best model as `model.pkl`.
5.  **Deployment:**
    *   Docker image built and pushed to AWS ECR.
    *   Application deployed on AWS EC2.

## 📂 Project Structure
```text
├── .github/workflows   # CI/CD YAML files
├── src
│   ├── components      # Ingestion, Validation, Transformation, Trainer
│   ├── config          # Configuration management
│   ├── constants       # Hardcoded values
│   ├── entity          # Dataclasses for inputs/outputs
│   ├── pipeline        # Training and Prediction pipelines
│   ├── utils           # Utility functions
├── artifacts           # Stores generated files (CSV, PKL, YAML)
├── config              # config.yaml
├── app.py              # Flask/FastAPI app entry point
├── main.py             # Main pipeline executor
├── push_data.py        # Script to push local CSV to MongoDB
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── README.md
