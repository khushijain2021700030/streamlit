import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Define the function to process new data and classify anomalies
def classify_anomalies(new_data, model_iso, model_lof):
    # Convert relevant columns to numeric and handle errors
    new_data = new_data.apply(pd.to_numeric, errors='coerce')

    # Drop rows with any NaN values
    new_data = new_data.dropna()

    # Scale Amount and Time columns
    scaler = StandardScaler()
    new_data[['Scaled_Amount', 'Scaled_Time']] = scaler.fit_transform(new_data[['Amount', 'Time']])
    
    # Drop the original columns (optional)
    new_data = new_data.drop(columns=['Amount', 'Time'], errors='ignore')

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(new_data)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Classify anomalies
    pca_df['Predicted_ISO'] = model_iso.predict(new_data)
    pca_df['Predicted_LOF'] = model_lof.predict(new_data)

    # Convert -1 to 1 (anomalies) and 1 to 0 (normal)
    pca_df['Predicted_ISO'] = pca_df['Predicted_ISO'].apply(lambda x: 1 if x == -1 else 0)
    pca_df['Predicted_LOF'] = pca_df['Predicted_LOF'].apply(lambda x: 1 if x == -1 else 0)
    
    return pca_df

# Load pre-trained models (ensure these files exist and are in the correct path)
import joblib
model_iso = joblib.load("iso_forest_model.pkl")
model_lof = joblib.load("lof_model.pkl")

# Streamlit app
st.title('Credit Card Transaction Anomaly Detection')

# Upload file
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, sheet_name='creditcard_test')

    # Display the data
    st.write("Data Preview:")
    st.write(data.head())
    
    # Classify anomalies
    classified_data = classify_anomalies(data, model_iso, model_lof)
    
    # Display classified data
    st.write("Classified Data:")
    st.write(classified_data)
    
    # Visualize the results
    st.subheader('PCA Visualization with Anomalies')
    fig, ax = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', hue='Predicted_ISO', data=classified_data, palette=['blue', 'red'], ax=ax)
    ax.set_title('PCA of Credit Card Transactions with Isolation Forest Anomalies')
    st.pyplot(fig)
    
    st.subheader('Local Outlier Factor (LOF) Visualization')
    fig, ax = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', hue='Predicted_LOF', data=classified_data, palette=['blue', 'red'], ax=ax)
    ax.set_title('PCA of Credit Card Transactions with LOF Anomalies')
    st.pyplot(fig)

