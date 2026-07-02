# 🏡 California House Prediction API

A clean, production-ready FastAPI application that serves a machine learning model to predict house prices in California. Built with a Scikit-Learn Random Forest Regressor, the API supports single-record JSON predictions as well as bulk data processing through CSV uploads.

---

## ✨ Features
- **Single Real-time Predictions:** Instantly computes individual property evaluations via a structured JSON payload.
- **Bulk Batch Processing:** Upload an entire `.csv` dataset and automatically receive a downloadable spreadsheet appended with clean, formatted predictions.
- **Strict Data Validation:** Uses Pydantic to enforce spatial data integrity rules (e.g., California latitude/longitude boundaries).
- **Auto-Generated UI Docs:** Built-in interactive API endpoint testing via FastAPI's native documentation interface.

---

## 🛠️ Technical Architecture & Stack
- **Framework:** FastAPI & Uvicorn
- **Data Science:** Pandas & NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Serialization:** Joblib

---

## 🚀 Local Deployment Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/rishi653-web/House_Prediction_API.git](https://github.com/rishi653-web/House_Prediction_API.git)
cd House_Prediction_API