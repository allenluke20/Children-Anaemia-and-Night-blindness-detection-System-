import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import time


st.write("**Input the child's health metrics across the categories below to evaluate the risks predictions for Anaemia and Nightblindness in real time.**")
# Optimally cache and load the trained pipeline artifacts
@st.cache_resource
def load_pipeline():
    model = joblib.load("saved model\\model.pkl")
    classes = joblib.load("saved model\\classes.pkl")
    return model, classes

try:
    model, classes = load_pipeline()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'model.pkl' and 'classes.pkl' exist in this directory.")
    st.stop()

# Exact schema ordering expected by your scikit-learn ColumnTransformer pipeline
FEATURE_COLUMNS = ['age', 'gender', 'bmi', 'diet_type', 'hemoglobin_g_dl','has_night_blindness', 'has_fatigue', 'has_pale_skin', 'has_dry_skin', 'other_diagnosis']

# Structure 3-column input form layout to manage 29 variables cleanly
col1, col2 = st.columns(2)

with col1:
    st.subheader(":grey[Demographics, Lifestyle & Clinical Metrics]")
    age = st.number_input("Age", min_value=1, max_value=17)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=35.0, value=24.5)
    diet_type = st.selectbox("Diet Type", ['omnivorous', 'vegetarian'])
    hemoglobin = st.number_input("Hemoglobin (g/dL)", value=13.5)


with col2:
    st.subheader(":grey[Symptoms Checklist]")
    symptoms = {
        "has_night_blindness": st.checkbox("Night Blindness"),
        "has_fatigue": st.checkbox("Chronic Fatigue"),
        "has_pale_skin": st.checkbox("Pale Skin Complexion"),
        "has_dry_skin": st.checkbox("Severely Dry Skin")
    }

    st.subheader(":grey[Medical History]")
    other = st.selectbox("Other diagnosis", ["No other disease", 'Malaria', 'Sickle cell'])

# Pack continuous variables and text strings directly into input dictionary
input_data = {
    "age": age, "gender": gender, "bmi": bmi,
    "diet_type": diet_type,"hemoglobin_g_dl": hemoglobin,
}

# Cast boolean symptom checks into binary format inputs (0 or 1)
for key, checked in symptoms.items():
    input_data[key] = 1 if checked else 0

input_data['other_diagnosis'] = other

# Convert input dictionary into structured DataFrame mapped to the model's expected column sequence
input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

st.session_state.gender = gender
st.session_state.age = age

st.markdown("---")

if st.button("Execute Prediction", type="primary", use_container_width=False):
    with st.spinner("Processing through pipeline components..."):
        try:
            # Model pipeline handles scaling and encoding natively 
            pred_idx = model.predict(input_df)[0]
            confidence = max(model.predict_proba(input_df)[0])
            hemoglobin = input_df["hemoglobin_g_dl"].iloc[0]                        
            diagnosis = classes[pred_idx]
            st.session_state.outputs = diagnosis
            st.session_state.conf = confidence
            st.session_state.hem = hemoglobin
            
            time.sleep(3)
            st.success("### Inference Complete")
            time.sleep(1)
            st.switch_page("pages//recommend.py")
            
        except Exception as e:
            st.error(f"Prediction Pipeline Failed: {e}")


