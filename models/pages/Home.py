import streamlit as st
col1, col2 = st.columns([3,1])
with col1:
    st.image("CAND.png", width='stretch')   

if "user_name" not in st.session_state:
    st.session_state.user_name = None

@st.dialog("Patient Identifier")
def name_prompt_popup():
    st.write("Please enter the child's name.")
    
    # Input widget inside the pop-up
    entered_name = st.text_input("Full Name:", placeholder="e.g. Cristiano Junior")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit"):
            if entered_name.strip():
                st.session_state.user_name = entered_name.strip()
                st.rerun() 
            else:
                st.error("Name field cannot be empty!")
                
    with col2:
        if st.button("Cancel"):
            st.rerun()

with col2:
    st.title(":blue[NutriScan]")
    st.expander('About NutriScan')
    if st.session_state.user_name is None:
        st.info("Hello! Please sign in to continue.")
    
        if st.button("Sign in"):
            name_prompt_popup()
    else:
        st.success(f" Signed in as: **{st.session_state.user_name}**")
    
        col3, col4 = st.columns([1,1])
        with col3:
            if st.button("Change Name"):
                st.session_state.user_name = None
                st.rerun()
        with col4:
            if st.button("Continue"):
                st.switch_page("pages//Inference.py")