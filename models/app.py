import streamlit as st
import pandas as pd
import joblib

# Configure the browser tab layout
st.set_page_config(page_title="Deficiency Diagnostic App", layout="wide")
st.title("Vitamin Deficiency Disease Classifier")
st.write("Modify the patient health metrics across the categories below to evaluate pipeline predictions in real time.")

# Optimally cache and load the trained pipeline artifacts
@st.cache_resource
def load_pipeline():
    model = joblib.load("model.pkl")
    classes = joblib.load("classes.pkl")
    return model, classes

try:
    model, classes = load_pipeline()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'model.pkl' and 'classes.pkl' exist in this directory.")
    st.stop()

# Exact schema ordering expected by your scikit-learn ColumnTransformer pipeline
FEATURE_COLUMNS = [
    "age", "gender", "bmi", "smoking_status", "alcohol_consumption", 
    "exercise_level", "diet_type", "sun_exposure", "vitamin_a_percent_rda", 
    "vitamin_c_percent_rda", "vitamin_d_percent_rda", "vitamin_e_percent_rda", 
    "vitamin_b12_percent_rda", "folate_percent_rda", "calcium_percent_rda", 
    "iron_percent_rda", "hemoglobin_g_dl", "serum_vitamin_d_ng_ml", 
    "serum_vitamin_b12_pg_ml", "serum_folate_ng_ml", "has_night_blindness", 
    "has_fatigue", "has_bleeding_gums", "has_bone_pain", "has_muscle_weakness", 
    "has_numbness_tingling", "has_memory_problems", "has_pale_skin", "has_dry_skin"
]

# Structure 3-column input form layout to manage 29 variables cleanly
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Lifestyle")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["Never", "Light", "Moderate", "Heavy"])
    exercise_level = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"])
    diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Pescatarian", "Vegan"])
    sun_exposure = st.selectbox("Sun Exposure Level", ["Low", "Moderate", "High"])

with col2:
    st.subheader("Biomarkers & Vitamin RDAs")
    vit_a = st.number_input("Vitamin A (% RDA)", value=100.0)
    vit_c = st.number_input("Vitamin C (% RDA)", value=100.0)
    vit_d = st.number_input("Vitamin D (% RDA)", value=100.0)
    vit_e = st.number_input("Vitamin E (% RDA)", value=100.0)
    vit_b12 = st.number_input("Vitamin B12 (% RDA)", value=100.0)
    folate = st.number_input("Folate (% RDA)", value=100.0)
    calcium = st.number_input("Calcium (% RDA)", value=100.0)
    iron = st.number_input("Iron (% RDA)", value=100.0)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", value=13.5)
    serum_d = st.number_input("Serum Vitamin D (ng/mL)", value=30.0)
    serum_b12 = st.number_input("Serum Vitamin B12 (pg/mL)", value=400.0)
    serum_folate = st.number_input("Serum Folate (ng/mL)", value=10.0)

with col3:
    st.subheader("Presenting Symptoms Checklist")
    symptoms = {
        "has_night_blindness": st.checkbox("Night Blindness"),
        "has_fatigue": st.checkbox("Chronic Fatigue"),
        "has_bleeding_gums": st.checkbox("Bleeding Gums"),
        "has_bone_pain": st.checkbox("Bone Pain"),
        "has_muscle_weakness": st.checkbox("Muscle Weakness"),
        "has_numbness_tingling": st.checkbox("Numbness/Tingling"),
        "has_memory_problems": st.checkbox("Memory Fluctuations"),
        "has_pale_skin": st.checkbox("Pale Skin Complexion"),
        "has_dry_skin": st.checkbox("Severely Dry Skin")
    }

# Pack continuous variables and text strings directly into input dictionary
input_data = {
    "age": age, "gender": gender, "bmi": bmi, "smoking_status": smoking_status,
    "alcohol_consumption": alcohol_consumption, "exercise_level": exercise_level,
    "diet_type": diet_type, "sun_exposure": sun_exposure,
    "vitamin_a_percent_rda": vit_a, "vitamin_c_percent_rda": vit_c, "vitamin_d_percent_rda": vit_d,
    "vitamin_e_percent_rda": vit_e, "vitamin_b12_percent_rda": vit_b12, "folate_percent_rda": folate,
    "calcium_percent_rda": calcium, "iron_percent_rda": iron,
    "hemoglobin_g_dl": hemoglobin, "serum_vitamin_d_ng_ml": serum_d,
    "serum_vitamin_b12_pg_ml": serum_b12, "serum_folate_ng_ml": serum_folate
}

# Cast boolean symptom checks into binary format inputs (0 or 1)
for key, checked in symptoms.items():
    input_data[key] = 1 if checked else 0

# Convert input dictionary into structured DataFrame mapped to the model's expected column sequence
input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

st.markdown("---")

# Inference execution flow trigger button
if st.button("Execute Pipeline Prediction", type="primary", use_container_width=True):
    with st.spinner("Processing through ML pipeline components..."):
        try:
            # Model pipeline handles scaling and encoding natively 
            pred_idx = model.predict(input_df)[0]
            diagnosis = classes[pred_idx]
            
            # Display target predictions output panels
            st.success("### Pipeline Inference Complete")
            st.metric(label="Predicted Diagnosis Result", value=str(diagnosis))
            
        except Exception as e:
            st.error(f"Prediction Pipeline Failed: {e}")
