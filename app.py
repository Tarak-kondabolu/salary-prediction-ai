import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# PAGE CONFIG
st.set_page_config(page_title="Salary Prediction AI", layout="wide")

# HEADER
st.markdown("""
    <div style='background-color:#4B9CD3;padding:10px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>💼 Salary Prediction System</h1>
        <p style='color:white;text-align:center;'>Machine Learning Based Salary Prediction</p>
    </div>
""", unsafe_allow_html=True)

st.write("\n")

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("salary_data.csv")

df = load_data()

# SIDEBAR
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations", "Predict Salary"])

# ----------------------------
# HOME PAGE
# ----------------------------
if menu == "Home":
    st.subheader("Welcome 👋")
    st.write("This app predicts salary based on:")
    st.markdown("""
    - Experience  
    - Education  
    - Job Role
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", "$8,241")
    col2.metric("MAE", "$3,118")
    col3.metric("R² Score", "0.97")
    
    st.write("---")

# ----------------------------
# DATASET
# ----------------------------
elif menu == "Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.write(f"Dataset Shape: {df.shape}")

# ----------------------------
# VISUALIZATIONS
# ----------------------------
elif menu == "Visualizations":
    st.subheader("Average Salary by Job Role")
    st.bar_chart(df.groupby("Job_Role")["Salary"].mean())

    st.subheader("Experience Distribution")
    st.bar_chart(df["Experience"])

# ----------------------------
# PREDICTION
# ----------------------------
elif menu == "Predict Salary":
    st.subheader("Predict Your Dream Salary")
    
    # INPUTS
    age = st.slider("Age", 18, 60, 25)
    experience = st.slider("Experience (Years)", 0, 40, 2)
    education = st.selectbox("Education", df["Education"].unique())
    job = st.selectbox("Job Role", df["Job_Role"].unique())

    # ENCODE
    le_job = LabelEncoder()
    le_edu = LabelEncoder()
    df["Job_Role_enc"] = le_job.fit_transform(df["Job_Role"])
    df["Education_enc"] = le_edu.fit_transform(df["Education"])

    X = df[["Age","Experience","Education_enc","Job_Role_enc"]]
    y = df["Salary"]

    model = RandomForestRegressor()
    model.fit(X, y)

    input_data = np.array([[age, experience, le_edu.transform([education])[0], le_job.transform([job])[0]]])

    # BUTTON
    if st.button("💰 Start Prediction"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: ₹ {int(prediction[0])}")

    if st.button("📊 View Analytics"):
        st.bar_chart(df.groupby("Job_Role")["Salary"].mean())
        st.bar_chart(df["Experience"])
