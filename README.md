# fraud-detection-app

# Real-Time E-Commerce Fraud Detection Engine

An end-to-end Machine Learning web application designed to evaluate and flag fraudulent transactions in real time. The project combines data cleaning, business analytics, interactive Power BI dashboards, machine learning, and Streamlit deployment.

---

## 🔄 Project Workflow

The project follows an end-to-end data workflow:

**Data Cleaning → Business Analytics → Machine Learning → Deployment**

### 1. Data Cleaning & Preparation — Python
The raw e-commerce transaction data was cleaned and prepared for analysis and machine learning, including handling data quality issues, duplicates, outliers, data types, and standardizing the dataset.

### 2. Business Analytics & Interactive Dashboard — Power BI
The cleaned dataset was analyzed using Power BI to identify important fraud patterns and business insights.

The dashboard focuses on:
- Fraud rate and fraud volume
- Financial impact of fraudulent transactions
- Transaction value patterns
- Account age and fraud risk
- Transaction timing
- Payment methods
- Product categories
- Device usage
- High-risk transaction combinations

Interactive slicers are included to allow dynamic exploration of the data.

### 3. Machine Learning
Multiple classification models were trained and evaluated under severe class imbalance conditions. XGBoost was selected as the final model based on its overall performance and fraud detection capability.

### 4. Deployment — Streamlit
The final machine learning pipeline was deployed as an interactive Streamlit web application for real-time fraud prediction.

---

## 🔬 Model Experimentation & Comparison

During model development in Google Colab, multiple classification algorithms were trained, tuned, and evaluated under severe class imbalance conditions to identify the best architecture for production:

| Model | Accuracy | Fraud Recall (Class 1) | ROC-AUC | Outcome / Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | ~95.0% | < 10.0% | ~0.72 | Suffered from the "majority class trap"—high accuracy by almost entirely missing fraud cases. |
| **Logistic Regression** | ~75.4% | ~66.4% | ~0.78 | Captured fraud effectively, but generated high false-positive rates (flagged over 1,100 legitimate transactions). |
| **XGBoost (Tuned)** | **83.14%** | **62.70%** | **0.8072** | **Best Overall:** Achieved the highest ROC-AUC and PR-AUC, catching 62.7% of fraud while cutting false alarms by ~395 cases. |

### Why XGBoost Was Selected:

1. **Handling Complex Feature Interactions:** XGBoost effectively captured non-linear risk signals such as high-dollar transactions on accounts under 30 days old combined with late-night hours.

2. **Optimal Business Balance:** It provided the strongest harmonic trade-off between loss prevention and customer experience.

3. **Optimized Decision Threshold:** By operating at a tuned threshold of **0.7885**, the pipeline maximizes the positive class F1-score specifically for production deployment.

---

## 📊 Business Analytics & Dashboard

The Power BI dashboard transforms the cleaned transaction data into actionable business insights.

The analysis explores:

- Overall fraud rate
- Fraudulent transaction value and financial exposure
- Transaction amount patterns
- Account age risk
- Fraud patterns across transaction hours and time periods
- Payment method behavior
- Product category patterns
- Device-related risk
- High-risk combinations between account age and transaction value

The dashboard uses interactive slicers and visual storytelling to help identify patterns associated with fraudulent transactions.

---

## 🌐 Live Demo

* **Interactive Web Application:** https://fraud-detection-app-g5yqmupy8osae8sspz4nkx.streamlit.app/
* **GitHub Repository:** https://github.com/MARDEV691/fraud-detection-app

---

## 📌 Project Overview

E-commerce fraud prevention presents a severe class-imbalance challenge: legitimate transactions heavily outnumber fraudulent ones.

This project addresses the problem through a complete data-driven workflow:

**Clean reliable data → Discover fraud patterns → Build a predictive model → Deploy the solution**

The Power BI analysis provides business insights into fraud behavior, while the machine learning model transforms these insights into a predictive fraud detection solution.

---

## 📈 Key Performance Metrics

| Evaluation Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8072** | High discriminatory power across varied decision thresholds |
| **Overall Accuracy** | **83.14%** | Reliable baseline performance across both classes |
| **Fraud Recall (Class 1)** | **62.70%** | Successfully catches nearly two-thirds of fraudulent transactions |
| **PR-AUC** | **0.3534** | Evaluates performance under severe positive class sparsity |
| **Optimal F1 Threshold** | **0.7885** | Optimized threshold minimizing customer checkout friction |

---

## ⚙️ Feature Engineering & Risk Signals

The model evaluates transactional and behavioral risk features:

* **`Amount_to_AccountAge_Ratio`**: Detects velocity spikes where newly created accounts make large purchases.
* **`Is_Late_Night`**: Flags higher-risk purchase timing (10:00 PM – 5:59 AM).
* **`Address_Mismatch`**: Identifies discrepancy between shipping and billing addresses.
* **`Unit_Price`**: Normalized transaction cost per item quantity.
* **`Device_Frequency`**: Device risk categorization across desktop, mobile, and tablet channels.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Power BI
- DAX
- Scikit-learn
- XGBoost
- Streamlit
- Google Colab
- GitHub

---

## 📁 Repository Structure

```text
fraud-detection-app/
├── data/                       # Dataset files
├── data-cleaning/              # Data cleaning notebooks/scripts
├── dashboard/                  # Power BI dashboard
├── machine-learning/           # ML notebooks and experiments
├── app.py                      # Interactive Streamlit application
├── requirements.txt             # Project dependencies
├── xgboost_fraud_model.json    # Serialized XGBoost model weights
├── model_config.json            # Model configuration & optimal decision threshold
├── model_features.json          # Exact feature schema expected by the model
├── scaler_params.json           # Fitted standardization parameters
├── presentation/               # Final project presentation
└── README.md                   # Project documentation
