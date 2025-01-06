import pandas as pd
import scipy.stats as stats

# Load Dataset
df = pd.read_csv('diagnostic_test_data.csv')

# Calculate Probability of Success
p = (df['Actual_Status'].isin(['True Positive', 'True Negative']).sum()) / len(df)

# Binomial Prediction
n = 30  # Number of patients for new prediction
k = 28  # Desired number of accurate predictions
prob = stats.binom.pmf(k, n, p)

print(f"Probability of exactly {k} accurate diagnoses out of {n} is {prob}")

"""Uses streamlit"""

import streamlit as st

st.title("Diagnostic Test Success Probability Predictor")

n = st.number_input("Number of Patients", min_value=1, value=30)
p = st.slider("Probability of Test Accuracy", 0.0, 1.0, 0.9)
k = st.number_input("Desired Accurate Diagnoses", min_value=0, value=28)

prob = stats.binom.pmf(k, n, p)
st.write(f"Probability of exactly {k} accurate diagnoses out of {n} is {prob:.4f}")
