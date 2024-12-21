import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# Title and description
st.title("AML Fraud Detection - Explainable AI")
st.write("This application predicts whether a transaction is fraudulent and provides explanations using SHAP values.")

# Load the model and feature names
try:
    model = joblib.load("/Users/kavitasingh/Projects/XAI/models/logistic_regression_2.pkl")  # Load only the model
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature names (manually)
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount']

# Sidebar for user input
st.sidebar.header("Input Transaction Details")
time = st.sidebar.number_input("Transaction Time", min_value=0, value=50000, step=1000)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)

# PCA-transformed features
pca_features = [st.sidebar.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

# Prepare the input DataFrame with the correct feature order
input_data = [time] + pca_features + [amount]
input_df = pd.DataFrame([input_data], columns=features)

# Predict fraud and provide explanations
if st.sidebar.button("Predict Fraud"):
    try:
        # Make predictions
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        # Display prediction results
        st.subheader("Prediction Result")
        st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Non-Fraud'}")
        st.write(f"Probability of Fraud: {probability:.2f}")

        # SHAP Analysis
        st.subheader("SHAP Analysis")
        st.write("Using SHAP to explain feature contributions:")

        # Use a representative dataset (replace input_df with a broader dataset for better insights)
        representative_data = input_df.copy()  # Replace with training/testing data if available

        # Create SHAP explainer
        explainer = shap.LinearExplainer(model, representative_data)
        shap_values = explainer.shap_values(representative_data)

        # SHAP Summary Plot
        st.subheader("SHAP Summary Plot")
        st.write("Feature importance across the dataset:")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, representative_data, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction or explanation: {e}")
