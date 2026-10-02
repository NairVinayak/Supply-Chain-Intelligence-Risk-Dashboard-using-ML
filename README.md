# 📦 Supply Chain Intelligence & Risk Dashboard using Machine Learning

An end-to-end analytics and machine learning project analyzing a 180K+ record e-commerce supply chain dataset to diagnose delivery delays, forecast demand, and predict shipment risk — deployed as a live interactive dashboard.

## 🔗 Live Demo
**[View Live Dashboard →](https://supply-chain-intelligence-risk-dashboard-using-ml-lipkmzhndkkf.streamlit.app/)**

## 📊 Project Overview
This project analyzes the DataCo Smart Supply Chain dataset (180,516 orders, $36.7M in sales, 23 regions, 4 shipping modes) to answer: **why are deliveries late, and what can be done about it?**

The analysis combines descriptive analytics, predictive machine learning, and model explainability to trace delivery delays to their root cause — and validates that finding using four independent methods.

## 🔍 Key Findings
- **Only 45% of orders are delivered on time** overall, consistent across nearly 3 years of data
- **Root cause:** First Class and Second Class shipping have unrealistic scheduled delivery windows (95% and 77% late respectively) — not inconsistent carrier performance, and not offset by any profit advantage
- **Western Europe and Central America** are the highest dollar-weighted risk regions — combining for $11.5M in sales (31% of revenue) while underperforming on reliability
- This root cause was independently confirmed by **four separate methods**: EDA, Random Forest feature importance, SHAP explainability, and K-means regional clustering

## 🧠 Machine Learning Components
| Model | Purpose | Result |
|---|---|---|
| Prophet | Demand forecasting | 8.13% MAPE |
| Random Forest | Delay risk classification | 0.73 ROC-AUC |
| SHAP | Model explainability | Confirmed shipping mode + scheduled days drive ~96% of predictions |
| K-means | Regional risk segmentation | 4 distinct risk-performance tiers identified |

## 🛠️ Tech Stack
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Forecasting:** Prophet
- **Machine Learning:** Scikit-learn (Random Forest, K-means)
- **Explainability:** SHAP
- **Dashboard:** Streamlit
- **Environment:** Google Colab

## 📁 Repository Structure
├── app.py # Streamlit dashboard application
├── requirements.txt # Python dependencies
├── models/ # Trained ML models (.pkl)
├── Data/ # Cleaned dataset and region cluster results
└── README.md


## 📈 Dataset
[DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle) — 180,516 orders across 53 original columns, spanning Jan 2015–Sep 2017.

## 🚀 Features
- Interactive KPI dashboard (on-time rate, sales, shipping delay)
- Regional risk visualization (cluster map + breakdown table)
- Live delay-risk predictor — input order details, get a real-time ML prediction

## Author
Vinayak Nair
