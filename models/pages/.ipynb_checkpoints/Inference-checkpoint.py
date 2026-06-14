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


def show_clinical_recommendation(prediction, confidence, hemoglobin=None, age=None):
    """
    Display clinical recommendations based on the model prediction.

    Parameters:
    -----------
    prediction : str
        Predicted class: "Anaemia", "Healthy", or "Night_blindness"
    confidence : float
        Model confidence score (0-1)
    hemoglobin : float, optional
        Patient hemoglobin value in g/dL
    age : int, optional
        Patient age in years
    """

    # Map prediction labels to display names
    label_map = {
        "Anaemia": "Anaemia",
        "Healthy": "Healthy",
        "Night_blindness": "Night Blindness"
    }

    display_label = label_map.get(prediction, prediction)

    # Color scheme matching the PPT theme
    if prediction == "Anaemia":
        color = "#E67E73"  # Coral
        bg_color = "#FFF5F5"
        border_color = "#E67E73"
        icon = "🩸"
        severity = "HIGH PRIORITY"
    elif prediction == "Night_blindness":
        color = "#F39C12"  # Orange
        bg_color = "#FFF9F0"
        border_color = "#F39C12"
        icon = "👁️"
        severity = "MODERATE PRIORITY"
    else:
        color = "#0D7377"  # Teal
        bg_color = "#F5FAFA"
        border_color = "#0D7377"
        icon = "✅"
        severity = "ROUTINE"

    # --- PREDICTION HEADER ---
    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border-left: 6px solid {border_color};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        ">
            <h2 style="color: {color}; margin: 0;">
                {icon} Prediction: {display_label}
            </h2>
            <p style="font-size: 18px; color: #333; margin: 8px 0 0 0;">
                Model Confidence: <b>{confidence:.1%}</b> | Severity: <b>{severity}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- CLINICAL RECOMMENDATIONS ---
    if prediction == "Anaemia":
        st.markdown("""
            <div style="background-color: #FFF5F5; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #C0392B; margin-top: 0;">🩸 Clinical Actions for Anaemia</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Immediate iron-folate supplementation</b> — WHO standard pediatric dosing (2 mg/kg elemental iron daily)</li>
                    <li><b>Counsel caregiver on iron-rich foods</b> — leafy greens (mchicha), beans, lentils, liver, red meat</li>
                    <li><b>Schedule follow-up hemoglobin test</b> — re-check in 8 weeks to monitor response</li>
                    <li><b>Flag for malaria co-treatment</b> — current malaria episode may worsen anemia; ensure full ACT course</li>
                    <li><b>Refer to district hospital urgently if:</b>
                        <ul>
                            <li>Hemoglobin < 8 g/dL (severe anemia threshold)</li>
                            <li>Signs of heart failure (tachycardia, gallop rhythm, hepatomegaly)</li>
                            <li>Pallor + respiratory distress</li>
                        </ul>
                    </li>
                    <li><b>Deworming</b> — administer albendazole if not given in last 6 months (helminths drive iron deficiency)</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        # Warning box for severe thresholds
        if hemoglobin and hemoglobin < 8.0:
            st.error("🚨 **SEVERE ANAEMIA ALERT**: Hemoglobin below 8 g/dL. Immediate referral to district hospital required.")
        elif hemoglobin and hemoglobin < 11.5:
            st.warning("⚠️ Hemoglobin below WHO normal threshold for children (11.5 g/dL). Supplementation and monitoring required.")

    elif prediction == "Night_blindness":
        st.markdown("""
            <div style="background-color: #FFF9F0; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #D35400; margin-top: 0;">👁️ Clinical Actions for Vitamin A Deficiency / Night Blindness</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Immediate high-dose Vitamin A</b> — WHO protocol: 200,000 IU oral (or 100,000 IU if age < 12 months)</li>
                    <li><b>Dark adaptation test referral</b> — schedule at district eye clinic within 2 weeks</li>
                    <li><b>Counsel on Vitamin A-rich diet</b> —
                        <ul>
                            <li>Orange vegetables: pumpkin, carrots, sweet potato</li>
                            <li>Dark leafy greens: spinach, amaranth (mchicha)</li>
                            <li>Animal sources: eggs, liver, fish, palm oil</li>
                        </ul>
                    </li>
                    <li><b>Deworming</b> — administer albendazole if not done in last 6 months (helminths worsen VAD absorption)</li>
                    <li><b>Re-screen in 3 months</b> — check for symptom resolution and serum retinol if available</li>
                    <li><b>Refer to ophthalmology if:</b>
                        <ul>
                            <li>No improvement after 3 months of supplementation</li>
                            <li>Signs of corneal xerosis or Bitot spots (advanced VAD)</li>
                            <li>Bilateral blindness or visual impairment</li>
                        </ul>
                    </li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        if age and age < 5:
            st.info("ℹ️ Patient is under 5 years — Vitamin A deficiency carries high risk of permanent visual impairment. Prioritize ophthalmology referral.")

    else:  # Healthy
        st.markdown("""
            <div style="background-color: #F5FAFA; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #0D7377; margin-top: 0;">✅ Routine Health Maintenance</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Continue balanced diet</b> — ensure regular intake of iron and Vitamin A-rich foods</li>
                    <li><b>Routine growth monitoring</b> — track BMI-for-age at monthly CHW visits</li>
                    <li><b>Standard immunization schedule</b> — verify up-to-date vaccines per EPI calendar</li>
                    <li><b>Deworming every 6 months</b> — albendazole as per national guidelines</li>
                    <li><b>Next screening</b> — schedule routine check-up in 6 months or sooner if symptoms develop</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        st.success("🎉 No immediate clinical intervention required. Continue standard preventive care.")

    # --- FOOTER DISCLAIMER ---
    st.markdown("""
        <div style="
            background-color: #F8F9FA;
            border-radius: 6px;
            padding: 12px;
            margin-top: 20px;
            font-size: 13px;
            color: #6C757D;
            text-align: center;
        ">
            ⚠️ <b>Disclaimer:</b> This AI tool supports — but does not replace — clinical""") 


if st.button("Execute Prediction", type="primary", use_container_width=False):
    with st.spinner("Processing through pipeline components..."):
        try:
            # Model pipeline handles scaling and encoding natively 
            pred_idx = model.predict(input_df)[0]
            confidence = max(model.predict_proba(input_df)[0])
            hemoglobin = input_df["hemoglobin_g_dl"].iloc[0]
            age = input_df["age"].iloc[0]                         
            diagnosis = classes[pred_idx]
            st.session_state.outputs = diagnosis
            show_clinical_recommendation(diagnosis, confidence, hemoglobin, age)
            
            time.sleep(3)
            st.success("### Inference Complete")
            time.sleep(1)
            st.switch_page("pages//recommend.py")
            
        except Exception as e:
            st.error(f"Prediction Pipeline Failed: {e}")


