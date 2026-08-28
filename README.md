# fraud-detection-app
#Real-Time E-Commerce Fraud Detection Engine

An end-to-end Machine Learning web application designed to evaluate and flag fraudulent transactions in real time. Built with an optimized **XGBoost Classifier**, scikit-learn preprocessing pipelines, and deployed interactively using **Streamlit Community Cloud**.
---


## 🔬 Model Experimentation & Comparison

During model development in Google Colab, multiple classification algorithms were trained, tuned, and evaluated under severe class imbalance conditions to identify the best architecture for production:

| Model | Accuracy | Fraud Recall (Class 1) | ROC-AUC | Outcome / Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | ~95.0% | < 10.0% | ~0.72 | Suffered from the "majority class trap"—high accuracy by almost entirely missing fraud cases. |
| **Logistic Regression** | ~75.4% | ~66.4% | ~0.78 | Captured fraud effectively, but generated high false-positive rates (flagged over 1,100 legitimate transactions). |
| **XGBoost (Tuned)** | **83.14%** | **62.70%** | **0.8072** | **Best Overall:** Achieved the highest ROC-AUC and PR-AUC, catching 62.7% of fraud while cutting false alarms by ~395 cases. |

### Why XGBoost Was Selected:
1. **Handling Complex Feature Interactions:** XGBoost effectively captured non-linear risk signals (e.g., high-dollar transactions on accounts under 30 days old combined with late-night hours).
2. **Optimal Business Balance:** It provided the strongest harmonic trade-off between loss prevention (high fraud recall) and customer experience (low false-positive friction).
3. **Optimized Decision Threshold:** By operating at a tuned threshold ($0.7885$), the pipeline maximizes the positive class F1-score specifically for production deployment.

---


 Live Demo
* **Interactive Web Application:** (https://fraud-detection-app-g5yqmupy8osae8sspz4nkx.streamlit.app/)
* **GitHub Repository:** [MARDEV691/fraud-detection-app](https://github.com/MARDEV691/fraud-detection-app)

---

 Project Overview
E-commerce fraud prevention presents a severe class-imbalance challenge: legitimate transactions heavily outnumber fraudulent ones. 

A naive baseline model predicting all transactions as legitimate can achieve high overall accuracy while failing to capture actual fraud. This project addresses that trade-off by engineering behavior-driven risk signals, tuning classification thresholds, and deploying an operational machine learning pipeline that catches fraud while minimizing checkout friction for legitimate shoppers.

---

Key Performance Metrics

| Evaluation Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8072** | High discriminatory power across varied decision thresholds |
| **Overall Accuracy** | **83.14%** | Reliable baseline performance across both classes |
| **Fraud Recall (Class 1)** | **62.70%** | Successfully catches nearly two-thirds of fraudulent transactions |
| **PR-AUC** | **0.3534** | Evaluates performance under severe positive class sparsity |
| **Optimal F1 Threshold** | **0.7885** | Optimized threshold minimizing customer checkout friction |

---

Feature Engineering & Risk Signals
The model evaluates transactional and behavioral risk features:
* **`Amount_to_AccountAge_Ratio`**: Detects velocity spikes where newly created accounts make large purchases.
* **`Is_Late_Night`**: Flags higher-risk purchase timing (10:00 PM – 5:59 AM).
* **`Address_Mismatch`**: Identifies discrepancy between shipping and billing addresses.
* **`Unit_Price`**: Normalized transaction cost per item quantity.
* **`Device_Frequency`**: Device risk categorization across desktop, mobile, and tablet channels.

---

 Repository Structure

```text
fraud-detection-app/
├── app.py                      # Interactive Streamlit application interface
├── requirements.txt            # Project dependencies for cloud deployment
├── xgboost_fraud_model.json    # Serialized XGBoost model weights
├── model_config.json           # Model configuration & optimal decision threshold
├── model_features.json         # Exact feature schema expected by the model
├── scaler_params.json          # Fitted standardization parameters (means & scales)
└── README.md                   # Project documentation & summary
