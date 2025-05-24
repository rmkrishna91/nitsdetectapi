import streamlit as st
import requests

st.set_page_config(page_title="AI-Generated Text Detector")

# Initialize session state variables
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "output_prob" not in st.session_state:
    st.session_state.output_prob = 0.0

st.title("AI-Generated Text Detection")

# Text area for input
text_input = st.text_area("Enter text to check if it's AI-generated:", height=200, value=st.session_state.text_input)

# Columns for Submit and Clear buttons

col1, col2 = st.columns([0.3, 2.2])

with col1:
    if st.button("Submit"):
        st.session_state.text_input = text_input  # Save current input
        if text_input.strip():
            
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",  # Update this if hosted externally
                    json={"text": text_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.output_prob = float(result["probability"])
                else:
                    st.error(f"FastAPI Error: {response.status_code}")
                    st.session_state.output_prob = 0.0
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                st.session_state.output_prob = 0.0
        else:
            st.warning("Please enter some text before submitting.")

with col2:
    if st.button("Clear"):
        st.session_state.text_input = ""
        st.session_state.output_prob = 0.0
        # st.experimental_rerun()  # Reset interface

# Display results if prediction exists
if st.session_state.output_prob > 0:
    label = "AI-Generated" if st.session_state.output_prob >= 0.5 else "Human-Written"
    st.success(f"Prediction: **{label}**")
    st.write(f"**Confidence: {st.session_state.output_prob:.2%}**")
    st.progress(st.session_state.output_prob)
