# 🚚 Dynamic Last-Mile Delivery ETA & Route Delay Prediction Engine

An end-to-end Machine Learning web application designed to replace legacy, rigid last-mile distance formulas with dynamic, data-driven ETAs and SLA breach risk predictions. Built to handle urban logistics volatility, traffic density, weather severity, and agent performance metrics.

🔗 **Live App:** [Streamlit Web Application](https://anil-kanasageri88-delivery-eta-prediction-y78yse5wkqdhdj3dpz3h.streamlit.app/)

---

## 📌 Project Overview
In modern e-commerce and quick-commerce, traditional static estimation models fail when real-world disruptions (such as traffic jams, peak ordering hours, and agent variability) occur. Inaccurate delivery estimates lead to customer dissatisfaction and costly SLA breaches. 

This project processes historical delivery datasets (**40,234 rows** from `amazon_delivery.csv`), performs rigorous feature engineering, and deploys an optimized **LightGBM** machine learning pipeline via an interactive Streamlit dashboard.

---

## 🛠️ Technical Stack & Architecture
* **Language & Core Libraries:** Python, Pandas, NumPy, Scikit-learn
* **Machine Learning:** LightGBM (Gradient Boosting Regressor)
* **Web Deployment:** Streamlit
* **Geospatial & Feature Engineering:** Haversine formula for spatial distance calculations, temporal features (`Order_Hour`), and categorical encoding for traffic and weather conditions.

---

## 📊 Model Performance & Benchmarking
After comparing multiple regression algorithms (Linear Regression, SVR, and Random Forest), **LightGBM** emerged as the winning model, capturing non-linear urban delivery dynamics without overfitting:
* **$R^2$ Score:** `0.8154`
* **Mean Absolute Error (MAE):** `17.31 minutes`
* **Root Mean Squared Error (RMSE):** `22.30 minutes`
* **Accuracy:** `84.04%`

---

## 🚀 Key Features of the Streamlit App
* **Interactive Parameter Tuning:** Adjust sliders and dropdowns for Agent Age, Rating, Distance (KM), Order Hour, Preparation Time, Weather, Traffic, Vehicle Type, and Area.
* **Built-in Sanity Checks & Input Validation:** Prevents unrealistic inputs (such as extreme last-mile distances >50 km) to maintain real-world operational realism.
* **Instant Predictions:** Generates real-time Estimated Delivery Time (ETA) outputs seamlessly.

---

## 📂 Repository Structure
```text
├── app.py                  # Main Streamlit application
├── model/                  # Serialized LightGBM models (.joblib)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
