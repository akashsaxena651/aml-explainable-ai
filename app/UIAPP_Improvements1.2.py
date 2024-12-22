import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
import altair as alt

# Title and description
st.title("AML Fraud Detection - Explainable AI")
st.write("This application predicts whether a transaction is fraudulent and provides explanations using SHAP values.")

# Load the model
try:
    model = joblib.load("models/logistic_regression_2.pkl")
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
                # Create SHAP explainer
                explainer = shap.LinearExplainer(model, input_df)
                shap_values = explainer.shap_values(input_df)

                # SHAP Analysis
                st.subheader("SHAP Analysis")
                st.write("Using SHAP to explain feature contributions:")
                
                # Create a summary plot with hover tooltips
                shap_values_df = pd.DataFrame(shap_values, columns=features)
                shap_values_df['index'] = shap_values_df.index

                # Melt the DataFrame for easier plotting
                shap_values_melted = shap_values_df.melt(id_vars=['index'], var_name='Feature', value_name='SHAP Value')

                # Add feature values to the melted DataFrame
                feature_values_df = input_df.copy()
                feature_values_df['index'] = feature_values_df.index
                feature_values_melted = feature_values_df.melt(id_vars=['index'], var_name='Feature', value_name='Feature Value')

                # Merge SHAP values and feature values
                shap_values_melted = shap_values_melted.merge(feature_values_melted, on=['index', 'Feature'])

                # Plot using Altair for interactivity

                chart = alt.Chart(shap_values_melted).mark_circle(size=60).encode(
                    x='SHAP Value',
                    y=alt.Y('Feature', sort=alt.EncodingSortField(field='SHAP Value', op='mean', order='descending')),
                    color=alt.Color('Feature Value', scale=alt.Scale(scheme='redblue')),
                    tooltip=['Feature', 'SHAP Value', 'Feature Value']
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                # Summary table for high probability fraud alerts
                st.subheader("High Probability Fraud Alerts Summary")
                high_prob_fraud = data[data['Probability'] > 0.8]
                high_prob_fraud['Alert Number'] = high_prob_fraud.index + 1
                high_prob_fraud['Danger Icon'] = high_prob_fraud['Probability'].apply(lambda x: "⚠️" if x == 1 else "")
                if not high_prob_fraud.empty:
                    st.write(high_prob_fraud[['Alert Number', 'Time', 'Amount', 'Probability', 'Danger Icon']])
                else:
                    st.write("No high probability fraud alerts found.")
                

                # Generate alerts for fraudulent transactions
                st.subheader("Fraud Alerts")
                for index, row in data.iterrows():
                    if row['Prediction'] == 'Fraud':
                        # Create a dropdown for detailed information
                        with st.expander(f"Alert: Transaction {index + 1}"):
                            st.write(f"Transaction Amount: {row['Amount']}")
                            st.write(f"Probability of Fraud: {row['Probability']:.2f}")
                            st.write("Detailed SHAP values:")
                            shap_values_for_row = shap_values[index]
                            shap_df = pd.DataFrame({
                                'Feature': features,
                                'SHAP Value': shap_values_for_row
                            }).sort_values(by='SHAP Value', key=abs, ascending=False)
                            
                            st.write(shap_df)
                            
                            # Show danger icon if probability is high
                            if row['Probability'] > 0.8:
                                st.warning("⚠️ High probability of fraud detected!")

                # Instructions for the user
                st.info("Click on the Alert for more details.")
                # Make predictions
                predictions = model.predict(input_df)
                probabilities = model.predict_proba(input_df)[:, 1]

                # Add predictions and probabilities to the DataFrame
                data['Prediction'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]
                data['Probability'] = probabilities

            except Exception as e:
                st.error(f"Error during prediction or explanation: {e}")
    else:
        st.error("Uploaded CSV does not contain the required columns.")
else:
    st.info("Please upload a CSV file containing transaction data.")
