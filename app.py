import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load model and scaler using relative paths (files must be in the same repo folder)
@st.cache_resource
def load_model():
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

# Set the app title and description
st.title('Hospital Admission Predictor')
st.write('This application predicts the likelihood of hospital admission within 30 days based on patient data.')
st.write('This prototype is developed by Rohit, Vishnu, Mary and Akmal.')
# Create input fields for user data
st.header('Enter Patient Information')

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

# Organize input data into a DataFrame
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

# Scale numeric features before making a prediction
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

if st.button('Predict Admission Risk'):
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data_scaled)[0][1]
    
    # Display the prediction outcome
    if prediction == 1:
        st.error('High Risk: Patient is likely to be admitted within 30 days.')
    else:
        st.success('Low Risk: Patient is unlikely to be admitted within 30 days.')
    
    if probability is not None:
        st.write(f"Admission Probability: {probability:.2f}")
        # Create an interactive gauge chart using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Admission Probability"},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 0.5], 'color': "lightgreen"},
                       {'range': [0.5, 1], 'color': "lightcoral"}
                   ]}
        ))
        st.plotly_chart(fig)
    
    # Generate a downloadable CSV report of the input data and prediction result
    report_data = input_data.copy()
    report_data['Prediction'] = prediction
    report_data['Probability'] = probability if probability is not None else 'N/A'
    csv = report_data.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Report", data=csv, file_name='prediction_report.csv', mime='text/csv')
