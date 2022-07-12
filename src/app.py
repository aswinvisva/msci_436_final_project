import pickle

import streamlit as st

from load_data import DataLoader

input_data = {}
data_loader = DataLoader()

age = st.slider('Input the Age', min_value=1, max_value=100)

input_data["Age"] = age

sex = st.selectbox(
     'What is your sex?',
     ('M', 'F'))

input_data["Sex"] = sex

chest_pain_type = st.selectbox(
     'What is your chest pain type?',
     ('ASY', 'NAP', 'ATA', 'TA'))

input_data["ChestPainType"] = chest_pain_type

resting_bp = st.slider('Resting BP', min_value=0, max_value=200)

input_data["RestingBP"] = resting_bp

cholesterol = st.slider('Cholesterol', min_value=0, max_value=700)

input_data["Cholesterol"] = cholesterol

fasting_bs = st.selectbox(
     'Fasting BS',
     (0, 1))

input_data["FastingBS"] = fasting_bs

resting_ecg = st.selectbox(
     'Resting ECG',
     ('Normal', 'ST', 'LVH'))

input_data["RestingECG"] = resting_ecg

max_hr = st.slider('MaxHR', min_value=50, max_value=250)

input_data["MaxHR"] = max_hr

exercise_angina = st.selectbox(
     'Exercising Angina',
     ('Y', 'N'))

input_data["ExerciseAngina"] = exercise_angina

old_peak = st.slider('Oldpeak', min_value=-10.0, max_value=10.0, step=0.1)

input_data["Oldpeak"] = old_peak

st_slope = st.selectbox(
     'ST_Slope',
     ('Up', 'Flat', 'Down'))

input_data["ST_Slope"] = st_slope

row = data_loader.transform_row(input_data)

with open('model.pkl', 'rb') as handle:
        clf = pickle.load(handle)

y_pred=clf.predict(row)[0]

if st.button('Diagnose'):
     st.write('Model Prediction: %.3g probability of heart disease' % y_pred)

