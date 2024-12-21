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

@st.cache_data
def load_data(file):
    """Load data in chunks and concatenate into a single DataFrame."""
    chunk_size = 100000  # Adjust based on your memory capacity
    data_chunks = []
    
    with st.spinner("Loading data..."):
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            data_chunks.append(chunk)
    
    return pd.concat(data_chunks, ignore_index=True)

if uploaded_file is not None:
    # Load the data
    data = load_data(uploaded_file)

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

                # Generate alerts for fraudulent transactions
                st.subheader("Fraud Alerts")
                for index, row in data.iterrows():
                    if row['Prediction'] == 'Fraud':
                        # Create a clickable link for detailed information using secrets
                        transaction_url = f"{st.secrets['general']['url']}/transaction/{index}"
                        st.markdown(f"[Alert: Transaction {index + 1}]({transaction_url})")
                        st.write(f"Transaction Amount: {row['Amount']}")
                        st.write(f"Probability of Fraud: {row['Probability']:.2f}")
                        st.write("---")

            except Exception as e:
                st.error(f"Error during prediction or explanation: {e}")
    else:
        st.error("Uploaded CSV does not contain the required columns.")
else:
    st.info("Please upload a CSV file containing transaction data.")