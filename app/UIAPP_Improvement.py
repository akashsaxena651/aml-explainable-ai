import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# Title and description
st.title("AML Fraud Detection - Explainable AI")
st.write("This application predicts whether a transaction is fraudulent and provides explanations using SHAP values.")

# Load the model
try:
    model = joblib.load("/Users/kavitasingh/Projects/XAI/models/logistic_regression_2.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature names
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount']

# Sidebar for user input
st.sidebar.header("Upload Transaction Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Transaction Data")
    st.write(data)

    # Check if required columns are present
    required_columns = features  # Use the defined features directly
    if all(col in data.columns for col in required_columns):
        # Prepare input DataFrame for predictions
        input_df = data[required_columns]

        # Ensure the order of columns matches the model's expected order
        input_df = input_df[features]  # Reorder to match the model's expected feature order

        # Predict fraud for each transaction
        if st.sidebar.button("Predict Fraud"):
            try:
                # Make predictions
                predictions = model.predict(input_df)
                probabilities = model.predict_proba(input_df)[:, 1]

                # Add predictions and probabilities to the DataFrame
                data['Prediction'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]
                data['Probability'] = probabilities

                # Display prediction results
                st.subheader("Prediction Results")
                st.write(data[['Time', 'Amount', 'Prediction', 'Probability']])

                # SHAP Analysis
                st.subheader("SHAP Analysis")
                st.write("Using SHAP to explain feature contributions:")

                # Create SHAP explainer
                explainer = shap.LinearExplainer(model, input_df)
                shap_values = explainer.shap_values(input_df)

                # SHAP Summary Plot
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, input_df, show=False)
                st.pyplot(fig)

                # Detailed account information
                st.subheader("Detailed Account Information")
                for index, row in data.iterrows():
                    st.write(f"Transaction {index + 1}:")
                    st.write(f"Transaction Amount: {row['Amount']}")
                    st.write(f"Prediction: {row['Prediction']}")
                    st.write(f"Probability of Fraud: {row['Probability']:.2f}")
                    st.write("---")

            except Exception as e:
                st.error(f"Error during prediction or explanation: {e}")
    else:
        st.error("Uploaded CSV does not contain the required columns.")
else:
    st.info("Please upload a CSV file containing transaction data.")