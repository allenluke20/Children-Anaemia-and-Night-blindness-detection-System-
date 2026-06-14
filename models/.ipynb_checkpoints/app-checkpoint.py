import streamlit as st
st.set_page_config(page_title="Deficiency Diagnostic App", layout="wide")
home_page = st.Page("pages//Home.py", title="Home")
inference = st.Page("pages//Inference.py", title="Diagnosis")
recommend = st.Page("pages//recommend.py", title="Predictions & Recommendations")
pg = st.navigation([home_page, inference, recommend])
pg.run()