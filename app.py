import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Model and Scaler using relative paths
@st.cache_resource
def load_model():
    # Use relative paths; the files must be in the same folder as app.py (or adjust accordingly)
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # List of features used during training
    selected_features = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'total_visits',
        'number_diagnoses',
        'gender_Female',
        'polypharmacy',
        'age_group',
        'chronic_condition'
    ]
    return model, scaler, selected_features

model, scaler, selected_features = load_model()

# 2. Streamlit App Title and Description
st.title('Hospital Readmission Predictor')
st.write('This application predicts the likelihood of hospital readmission within 30 days based on patient data.')
st.write('A prototype developed by Rohit, Vishnu, Mary and Akmal from 4CAI03.')

st.header('Enter Patient Information')

# 3. Create Input Fields
time_in_hospital = st.slider('Time in Hospital (days)', min_value=1, max_value=14, value=4)
num_lab_procedures = st.slider('Number of Lab Procedures', min_value=1, max_value=126, value=42)
num_procedures = st.slider('Number of Procedures', min_value=0, max_value=6, value=1)
num_medications = st.slider('Number of Medications', min_value=1, max_value=81, value=16)
total_visits = st.slider('Total Visits (outpatient + emergency + inpatient)', min_value=0, max_value=100, value=5)
number_diagnoses = st.slider('Number of Diagnoses', min_value=1, max_value=16, value=8)

gender = st.selectbox('Gender', ['Male', 'Female'])
gender_Female = 1 if gender == 'Female' else 0

age_group_label = st.selectbox('Age Group', ['40-50', '50-60', '60-70', '70-80', '80-90'])
age_group_mapping = {'40-50': 1, '50-60': 2, '60-70': 3, '70-80': 4, '80-90': 5}
age_group = age_group_mapping[age_group_label]

chronic_condition_flag = st.checkbox('Chronic Condition (heart failure, diabetes, hypertension, etc.)', value=False)
chronic_condition = 1 if chronic_condition_flag else 0

# Derived feature: Polypharmacy Indicator (1 if more than 10 medications)
polypharmacy = 1 if num_medications > 10 else 0

# 4. Organize the Input Data as a DataFrame
input_data = pd.DataFrame({
    'time_in_hospital': [time_in_hospital],
    'num_lab_procedures': [num_lab_procedures],
    'num_procedures': [num_procedures],
    'num_medications': [num_medications],
    'total_visits': [total_visits],
    'number_diagnoses': [number_diagnoses],
    'gender_Female': [gender_Female],
    'polypharmacy': [polypharmacy],
    'age_group': [age_group],
    'chronic_condition': [chronic_condition]
})

st.write("### Input Data Preview")
st.write(input_data)

# 5. Scale Numeric Features Before Prediction
numeric_cols = [
    'time_in_hospital', 
    'num_lab_procedures', 
    'num_procedures', 
    'num_medications', 
    'total_visits', 
    'number_diagnoses'
]
input_data_scaled = input_data.copy()
input_data_scaled[numeric_cols] = scaler.transform(input_data[numeric_cols])

# 6. Make Prediction on Button Click
if st.button('Predict Readmission Risk'):
    prediction = model.predict(input_data_scaled)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data_scaled)[0][1]
    
    if prediction == 1:
        st.error('High Risk: Patient is likely to be readmitted within 30 days.')
    else:
        st.success('Low Risk: Patient is unlikely to be readmitted within 30 days.')
    
    if probability is not None:
        st.write(f"Readmission Probability: {probability:.2f}")
