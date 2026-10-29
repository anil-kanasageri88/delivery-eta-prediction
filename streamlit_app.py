import streamlit as st
import joblib
import pandas as pd

# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load('lgbm_eta_model.pkl')
    feature_columns = joblib.load('model_features.pkl')
    return model, feature_columns

model, feature_columns = load_model()

st.title("🚚 Last-Mile Delivery ETA Prediction Engine")
st.write("Enter the order, agent, and environmental details to predict the estimated delivery time.")

# --- Input Widgets ---
col1, col2 = st.columns(2)

with col1:
    agent_age = st.slider("Agent Age", 18, 50, 28)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.7, 0.1)
    distance_km = st.number_input("Distance (KM)", min_value=0.1, value=8.5)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
    prep_time = st.number_input("Prep Time (Minutes)", min_value=1.0, value=12.0)

with col2:
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Stormy", "Sandstorms", "Windy", "Fog"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
    vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "van", "bicycle"])
    area = st.selectbox("Area", ["Urban", "Metropolitian", "Semi-Urban", "Other"])
    category = st.selectbox("Category", ["Electronics", "Clothing", "Sports", "Cosmetics", "Toys", "Grocery", "Books", "Home"])

# --- Prediction Logic ---
if st.button("Predict Estimated Delivery Time", type="primary"):
    input_data = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Distance_KM': distance_km,
        'Prep_Time_Minutes': prep_time,
        'Order_Hour': order_hour
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    def set_feature(prefix, val):
        for col in feature_columns:
            if col.strip().lower() == f"{prefix}_{val}".strip().lower():
                input_df.loc[0, col] = 1

    set_feature('Weather', weather)
    set_feature('Traffic', traffic)
    set_feature('Vehicle', vehicle)
    set_feature('Area', area)
    set_feature('Category', category)

    prediction = model.predict(input_df)[0]

    # Display Prediction Result
    st.success(f"Estimated Delivery Time (ETA): **{prediction:.2f} minutes**")

st.write("---")

# --- Model Performance & Business Metrics Dashboard ---
st.subheader("📊 Model Performance & Business Impact")
m1, m2, m3 = st.columns(3)

m1.metric("Model MAE", "17.01 mins", "Error tolerance")
m2.metric("R² Score", "0.8222", "Variance explained")
m3.metric("SLA Breach Rate", "10.52%", "-4.48% vs Target (<15%)")

# --- Full Model Performance Comparison Table ---
st.write("---")
st.subheader("🔬 All Models Accuracy Comparison")
st.write("Comparing LightGBM against baseline and traditional regression models:")

comparison_data = {
    "Model": ["LightGBM", "Random Forest", "Linear Regression", "SVR"],
    "MAE (Minutes)": [17.01, 17.41, 26.10, 39.71],
    "RMSE (Minutes)": [21.76, 22.76, 33.15, 49.91],
    "R² Score": [0.8222, 0.8055, 0.5876, 0.0649]
}

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(comparison_df.style.highlight_min(subset=["MAE (Minutes)", "RMSE (Minutes)"], color="lightgreen")
                                  .highlight_max(subset=["R² Score"], color="lightgreen"), 
             use_container_width=True)
