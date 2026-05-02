# Predictive Maintenance System for Foundry Machine

## Project Overview

This project presents a complete **AI-based Predictive Maintenance System (PdMS)** developed during a 20-week industrial internship at **BVM FSM, CoE DM (in collaboration with IIT Delhi – AIA Foundation)**. 

The system analyzes real industrial sensor and alarm data from a foundry machine to:

* Predict machine failures in advance
* Classify fault types
* Estimate Remaining Useful Life (RUL)
* Detect anomalies in machine behavior

It enables a transition from **reactive maintenance → proactive maintenance**.

---

## Key Features

* Binary Failure Prediction (Fail / No Fail)
* Fault Type Classification
* Remaining Useful Life (RUL) Estimation
* Anomaly Detection using Isolation Forest
* Real-time Streamlit Dashboard
* SHAP-based Model Explainability
* Supports:

  * CSV Upload Mode
  * Manual Sensor Input Mode

---

## Dataset Details

* **FM_data.csv** → 2,665 rows, 17 sensor parameters
* **FM_alarms.csv** → 156,248 rows, 172 alarm columns 

### Sensor Parameters Include:

* Cycle time
* Hydraulic pressure
* Oil temperature
* Cylinder positions
* Sand weight
* Squeeze pressure

---

## Feature Engineering

* Total Features: **60**
* Rolling statistics (mean, std, min, max)
* Lag features (1, 3, 5 cycles)
* Historical fault rates
* Health Index (0–100 scoring system)

---

## Machine Learning Models

### 1. Failure Prediction

* Model: Random Forest
* Accuracy: **95.3%**
* F1 Score: **96.6%**
* ROC-AUC: **98.4%** 

### 2. Fault Classification

* Model: XGBoost / Random Forest
* Accuracy: up to **96%**

### 3. RUL Prediction

* Model: Random Forest Regressor

### 4. Anomaly Detection

* Model: Isolation Forest

---

## Explainability

* SHAP (SHapley Additive Explanations) used
* Identifies most critical features influencing failure
* Example key predictors:

  * Fault history (HPU Oil Level Low)
  * Arm position faults
  * Oil temperature trends

---

## Streamlit Dashboard

### Features:

* Health Index Gauge (0–100)
* Failure Probability (%)
* RUL Estimation
* Alert Levels:

  * NORMAL
  * WARNING
  * CRITICAL
* Radar Charts & Visualizations
* Maintenance Recommendations

---

## System Architecture

1. Data Ingestion (Sensor + Alarm Data)
2. Data Cleaning & Preprocessing
3. Feature Engineering (60 Features)
4. Model Execution:

   * Failure Prediction
   * Fault Classification
   * RUL Prediction
   * Anomaly Detection
5. Output via Streamlit Dashboard

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Plotly / Matplotlib

---

## How to Run

```bash
git clone https://github.com/your-username/predictive-maintenance-system.git
cd predictive-maintenance-system
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

---

## Deployment

Deployed using Streamlit Cloud

 Live App: *(Add your link here)*

---

## Limitations

* Trained on a single machine dataset
* Limited time range (7 days data)
* No real-time PLC integration
* RUL based on relative cycle estimation

---

## Future Enhancements

* Real-time IoT (PLC integration using MQTT/OPC-UA)
* Multi-machine monitoring
* Automated alerts (Email/SMS)
* Advanced deep learning models (ANN/CNN)
* Continuous model retraining

---

## Author

**Kamya Patel**
B.Tech – Computer Science and Design

---

## License

© 2026 Kamya Patel. All Rights Reserved.

This project is proprietary and confidential. It has been developed using real industrial data and is intended strictly for academic and demonstration purposes.

You are NOT allowed to:

* Copy, modify, or distribute this project
* Use this project or its code for commercial purposes
* Reproduce this system in any form without explicit permission

You MAY:

* View the project for learning and reference purposes only

Any unauthorized use, reproduction, or distribution of this project is strictly prohibited and may result in legal action.

For permissions, contact: kamyapatel75@gmail.com

