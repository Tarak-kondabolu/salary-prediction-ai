import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="wide"
)

# ---------------- CUSTOM DARK THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1f1c2c);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.metric-card {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("salary_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("salary_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("", ["Home", "Predict", "Analytics"])

# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":

    st.markdown("<h1 style='text-align:center;'>💼 Salary Prediction AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Predict Your Dream Salary with Machine Learning</h4>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.markdown("<div class='metric-card'><h3>Model</h3><h2>Random Forest</h2></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><h3>R² Score</h3><h2>0.97</h2></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card'><h3>Accuracy</h3><h2>97%</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class='card'>
    This AI-powered system predicts salary based on:
    - Experience
    - Education
    - Role
    - Skill Level
    
    Built using Random Forest Regressor.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Predict":

    st.markdown("<h2>🎯 Predict Your Salary</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        experience = st.slider("Experience (Years)", 0, 40, 2)
        skill_level = st.selectbox("Skill Level", df["skill_level"].unique())

    with col2:
        education = st.selectbox("Education", df["education"].unique())
        role = st.selectbox("Role", df["role"].unique())

    input_data = pd.DataFrame({
        "experience": [experience],
        "education": [education],
        "role": [role],
        "skill_level": [skill_level]
    })

    input_encoded = pd.get_dummies(input_data)

    model_columns = model.feature_names_in_

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    if st.button("💰 Predict Salary"):
        prediction = model.predict(input_encoded)
        st.success(f"Estimated Salary: ${prediction[0]:,.2f}")

# =========================================================
# ANALYTICS PAGE
# =========================================================
elif page == "Analytics":

    st.markdown("<h2>📊 Salary Analytics Dashboard</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            df.groupby("role")["salary"].mean().reset_index(),
            x="role",
            y="salary",
            title="Average Salary by Role",
            template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df.groupby("skill_level")["salary"].mean().reset_index(),
            x="skill_level",
            y="salary",
            title="Salary by Skill Level",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df,
        x="salary",
        title="Salary Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig3, use_container_width=True)
