import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from fpdf import FPDF  # pip install fpdf
from io import BytesIO

# Load model and scaler using relative paths
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

st.title('Hospital Admission Predictor')
st.write('This application predicts the likelihood of hospital admission within 30 days based on patient data.')
st.write('A prototype developed by Rohit, Vishnu, Mary and Akmal from 4CAI03.')

st.header('Enter Patient Information')

# Input widgets
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

# Derived feature: Polypharmacy Indicator (1 if num_medications > 10)
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

# Scale numeric features before prediction
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

# Function to generate a PDF report using FPDF
def generate_pdf(report_data, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Hospital Admission Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for key, value in report_data.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    if probability is not None:
        pdf.cell(0, 10, f"Admission Probability: {probability:.2f}", ln=True)
    else:
        pdf.cell(0, 10, "Admission Probability: N/A", ln=True)
    # Generate PDF as bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

if st.button('Predict Admission Risk'):
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data_scaled)[0][1]
    
    # Display prediction result
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
    
    # Prepare report data as a dictionary (from the single row of input_data)
    report_dict = input_data.iloc[0].to_dict()
    report_dict["Prediction"] = 'High Risk' if prediction == 1 else 'Low Risk'
    report_dict["Probability"] = f"{probability:.2f}" if probability is not None else "N/A"
    
    # Generate PDF report
    pdf_bytes = generate_pdf(report_dict, prediction, probability)
    
    # Provide a download button for the PDF report
    st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="prediction_report.pdf", mime="application/pdf")
