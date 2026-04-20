import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_squared_error

st.title("⚡ Smart Energy System")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)

st.subheader("Dataset Preview")
st.write(data.head())

data = data.select_dtypes(include=[np.number]).dropna()

if data.shape[1] < 2:
    st.error("Need at least 2 numeric columns")
else:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-6)

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    preds = model.predict(X)

    # Metrics
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)

    st.subheader("Model Performance")
    st.write("R²:", round(r2, 3))
    st.write("MSE:", round(mse, 2))

    # Plot
    st.line_chart(pd.DataFrame({
        "Actual": y,
        "Predicted": preds
    }))

    # Anomaly detection
    st.subheader("Anomaly Detection")

    iso = IsolationForest(contamination=0.03)
    labels = iso.fit_predict(X)

    st.write("Anomalies:", np.sum(labels == -1))

    # Optimization
    st.subheader("Energy Optimization")

    optimized = preds * 0.85

    st.line_chart(pd.DataFrame({
        "Predicted": preds,
        "Optimized": optimized
    }))
if st.button("Run Model"):
