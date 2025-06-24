import streamlit as st
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai

# 🔐 Gemini setup
genai.configure(api_key="AIzaSyATtxQh_spw-yMHzIA-gyWdUxp5bfqry0s")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# 📦 Load model and metadata
ml_model = pickle.load(open("best_cvd_model.pkl", "rb"))
metadata = pickle.load(open("cvd_metadata.pkl", "rb"))

# 🩺 Page setup
st.set_page_config(page_title="CVD Prediction App", page_icon="❤️")
st.title("🩺 CVD Risk Prediction & Diet Recommendation")

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "show_diet" not in st.session_state:
    st.session_state.show_diet = False
if "input_data" not in st.session_state:
    st.session_state.input_data = {}
if "cvd" not in st.session_state:
    st.session_state.cvd = None

# --- Form for input ---
with st.form("cvd_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 10, 100, 30)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    with col2:
        gender = st.selectbox("Gender", metadata['Gender'])
        activity = st.selectbox("Activity Level", metadata['Activity Level'])
        disease = st.selectbox("Disease", metadata['Disease'])
        diet_pref = st.selectbox("Dietary Preference", ["Vegetarian", "Non-Vegetarian"])
    submitted = st.form_submit_button("🔍 Predict")

# --- On form submit ---
if submitted:
    gender_num = 0 if gender == "Male" else 1
    input_df = pd.DataFrame([{
        "Ages": age,
        "Gender": gender_num,
        "Height": height,
        "Weight": weight,
        "Activity Level": activity,
        "Disease": disease
    }])
    prediction = ml_model.predict(input_df)[0]

    # Save state
    st.session_state.prediction = prediction
    st.session_state.input_data = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "activity": activity,
        "disease": disease,
        "diet_pref": diet_pref
    }
    st.session_state.show_diet = False  # Reset diet section

# --- Show Prediction Result ---
if st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    data = st.session_state.input_data

    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of CVD")
        if st.button("🍽️ Generate Personalized Diet Plan"):
            st.session_state.show_diet = True
            st.session_state.cvd = "cvd"
    else:
        st.success("✅ Low Risk of CVD")
        if st.button("🍽️ Generate Personalized Diet Plan for Non CVD"):
            st.session_state.show_diet = True
            st.session_state.cvd = "not cvd"

# --- Show Diet Plan if button clicked ---
if st.session_state.show_diet:
    data = st.session_state.input_data
    risk_label = "High" if st.session_state.cvd == "cvd" else "Low"

    prompt = f"""
    I want you to act as a professional nutritionist and fitness expert.

    Based on this profile:
    - Age: {data['age']}
    - Gender: {data['gender']}
    - Height: {data['height']} cm
    - Weight: {data['weight']} kg
    - Activity Level: {data['activity']}
    - Disease: {data['disease']}
    - Dietary Preference: {data['diet_pref']}
    - Health Condition: {risk_label} Risk of CVD

    Please provide a full-day personalized diet and workout plan:
    - Breakfast
    - Lunch
    - Dinner
    - Healthy snacks
    - Simple home-based workout

    Use bullet points. Avoid excess salt, sugar, and red meat.
    """

    with st.spinner("Generating diet plan..."):
        response = gemini_model.generate_content(prompt)

    st.markdown("### 💡 Personalized Diet Plan")
    st.markdown(response.text)
