import streamlit as st

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
        severity = "HIGH PRIORITY"
    elif prediction == "Night_blindness":
        color = "#F39C12"  # Orange
        bg_color = "#FFF9F0"
        border_color = "#F39C12"
        severity = "MODERATE PRIORITY"
    else:
        color = "#0D7377"  # Teal
        bg_color = "#F5FAFA"
        border_color = "#0D7377"
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
                Prediction: {display_label}
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
                <h4 style="color: #C0392B; margin-top: 0;"> Clinical Actions for Anaemia</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Immediate iron-folate supplementation</b> — WHO standard pediatric dosing (2 mg/kg elemental iron daily)</li>
                    <li><b>Caregiver should increase iron-rich foods to the child</b> —
                        <ul>
                            <li>Leafy Green Vegetables: leafy greens (mchicha), Pumpkin Leaves (Maboga)</li> 
                            <li>Legumes and Pulses: Beans (Maharagwe), Lentils (Lenda / Dengu), Peanuts (Karanga)</li> 
                            <li>Animal Products: Eggs (mayai), Liver (maini)</li> 
                        </ul>
                    <li><b>Schedule follow-up hemoglobin test</b> — re-check in 8 weeks to monitor response</li>
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

        if hemoglobin and hemoglobin < 8.0:
            st.error("🚨 **SEVERE ANAEMIA ALERT**: Hemoglobin below 8 g/dL. Immediate referral to district hospital required.")
        elif hemoglobin and hemoglobin < 11.5:
            st.warning("⚠️ Hemoglobin below WHO normal threshold for children (11.5 g/dL). Supplementation and monitoring required.")
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
            ⚠️ <b>Disclaimer:</b> This AI tool supports — but does not replace — clinical judgment. 
            Confirm all high-risk predictions with laboratory testing and physician review.
        </div>
    """, unsafe_allow_html=True)

    elif prediction == "Night_blindness":
        st.markdown("""
            <div style="background-color: #FFF9F0; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #D35400; margin-top: 0;"> Clinical Actions for Vitamin A Deficiency / Night Blindness</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Immediate high-dose Vitamin A</b> — WHO protocol: 200,000 IU oral (or 100,000 IU if age < 12 months)</li>
                    <li><b>Dark adaptation test referral</b> — schedule at district eye clinic within 2 weeks</li>
                    <li><b>Caregiver should provide a proper Vitamin A-rich diet to the child</b> —
                        <ul>
                            <li>Orange vegetables: Pumpkin (Maboga), Carrots, Sweet potato (Viazi vitamu)</li>
                            <li>Dark leafy greens: Spinach, Amaranth (Mchicha)</li>
                            <li>Animal sources: Eggs (Mayai), Liver (Maini), Fish (Samaki), Palm oil (Mafuta ya mawese)</li>
                        </ul>
                    </li>
                    <li><b>Deworming</b> — administer albendazole if not done in last 6 months (helminths worsen Vitamin A absorption)</li>
                    <li><b>Re-screen in 3 months</b> — check for symptom resolution and serum retinol if available</li>
                    <li><b>Refer to ophthalmology if:</b>
                        <ul>
                            <li>No improvement after 3 months of supplementation</li>
                            <li>Bilateral blindness or visual impairment</li>
                        </ul>
                    </li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        if age and age < 5:
            st.info("ℹ️ Patient is under 5 years — Vitamin A deficiency carries high risk of permanent visual impairment. Prioritize ophthalmology referral.")
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
            ⚠️ <b>Disclaimer:</b> This AI tool supports — but does not replace — clinical judgment. 
            Confirm all high-risk predictions with laboratory testing and physician review.
        </div>
    """, unsafe_allow_html=True)

    else:  # Healthy
        st.markdown("""
            <div style="background-color: #F5FAFA; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #0D7377; margin-top: 0;"> Routine Health Maintenance</h4>
                <ol style="font-size: 16px; line-height: 1.8; color: #333;">
                    <li><b>Continue balanced diet</b> — ensure regular intake of iron and Vitamin A-rich foods</li>
                    <li><b>Routine growth monitoring</b> — track BMI-for-age monthly</li>
                    <li><b>Standard immunization schedule</b> — verify up-to-date vaccines per EPI calendar</li>
                    <li><b>Deworming every 6 months</b> — albendazole as per doctor's prescriptions</li>
                    <li><b>Next screening</b> — schedule routine check-up in 6 months or sooner if symptoms develop</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        st.success(" No immediate clinical intervention required. Continue standard preventive care.")

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
            ⚠️ <b>Disclaimer:</b> This AI tool supports — but does not replace — clinical judgment. 
            Confirm all high-risk predictions with laboratory testing and physician review.
        </div>
    """, unsafe_allow_html=True)

prediction = st.session_state.outputs
confidence = st.session_state.conf
hemoglobin = st.session_state.hem
age = st.session_state.age
show_clinical_recommendation(prediction, confidence, hemoglobin, age)