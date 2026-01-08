# 🌾 Live Crop Yield Risk Dashboard  
### Uncertainty-Aware & Risk-Sensitive Crop Yield Forecasting

---

## 🚀 Project Overview

This repository contains a **full-stack machine learning web application** that delivers **uncertainty-aware crop yield forecasts**, **risk metrics**, and **causal insights** via a **live interactive Plotly Dash dashboard** integrated with **Flask**.

Unlike traditional point-prediction systems, this project models:
- **Prediction intervals (Q10 / Q50 / Q90)**
- **Uncertainty & risk (PICP, CRPS, VaR, CVaR)**
- **Extreme events (EVT: GEV & POT)**
- **Region-wise causal effects (ATE)**

---

## ✨ Key Features

### 🔹 Machine Learning
- Quantile LightGBM regression
- Predictive distribution instead of a single value
- Robust evaluation metrics

### 🔹 Uncertainty & Risk Metrics
- RMSE, R²  
- PICP (Prediction Interval Coverage Probability)  
- CRPS (Continuous Ranked Probability Score)  
- Sharpness & Coverage-Width Tradeoff (CWT)  
- VaR(99%) & CVaR(99%)

### 🔹 Extreme Value Theory (EVT)
- Generalized Extreme Value (GEV)
- Peaks-Over-Threshold (POT)

### 🔹 Causal Inference
- Region-wise Average Treatment Effect (ATE)
- Doubly Robust estimation

### 🔹 Live Interactive Dashboard
- Built with Plotly Dash
- Zoom, pan, hover, toggle legends
- Auto-updates after dataset upload
- Metrics + plots shown together

---

## 📊 Dashboard Visualizations

The dashboard includes:

1. **Calibration Plot** – Predicted vs Observed Yield  
2. **Prediction Interval Coverage vs Width**  
3. **Residual / CRPS Distribution**  
4. **Tail Fit Comparison (GEV vs POT)**  
5. **VaR & CVaR Bar Plot**  
6. **Region-wise Average Treatment Effect (ATE)**  
7. **Metrics Summary Cards**  
   - RMSE, R², PICP, CRPS  
   - Sharpness, CWT  
   - VaR, CVaR  

---

## 🧠 Tech Stack

| Layer | Technology |
|-----|-----------|
| Backend | Flask |
| Dashboard | Plotly Dash |
| ML | LightGBM |
| Statistics | EVT (GEV, POT), CRPS |
| Causal | Doubly Robust Estimation |
| Visualization | Plotly |
| Deployment | Gunicorn, Render |

---

## 📁 Project Structure
```bash
crop_yield_webapp/
│
├── app.py # Flask entry point
├── dashboard_dash.py # Live Dash dashboard
├── Procfile # Deployment config
├── requirements.txt
│
├── model/
│ ├── init.py
│ └── train_predict.py # ML + risk + causal logic
│
├── templates/
│ └── index.html # Dataset upload page
│
├── static/
│ └── results/
│ └── predictions.csv # Auto-generated predictions
│
├── uploads/
│ └── data.csv
│
└── README.md

```

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/crop-yield-risk-dashboard.git
cd crop-yield-risk-dashboard
## 🚀 Getting Started
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
python app.py
```
