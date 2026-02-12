import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Salary Prediction AI", layout="wide")

st.title("💼 Salary Prediction System")
st.markdown("A Machine Learning based Salary Prediction Web App")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salary_data.csv")

df = load_data()

# ----------------------------
# SIDEBAR MENU
# ----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Dataset", "Visualizations", "Train Model", "Predict Salary", "About"]
)

# ----------------------------
# HOME
# ----------------------------
if menu == "Home":
    st.header("Welcome 👋")
    st.write("""
    This application predicts salary based on:
    - Age
    - Experience
    - Education
    - Job Role
    
    Built using:
    - Pandas
    - Scikit-Learn
    - Streamlit
    """)

# ----------------------------
# DATASET VIEW
# ----------------------------
elif menu == "Dataset":
    st.header("Dataset Preview")
    st.dataframe(df)
    st.write("Dataset Shape:", df.shape)

# ----------------------------
# VISUALIZATIONS
# ----------------------------
elif menu == "Visualizations":
    st.header("Data Visualizations")

    fig1 = plt.figure()
    df["Experience"].hist()
    plt.title("Experience Distribution")
    plt.xlabel("Years")
    plt.ylabel("Count")
    st.pyplot(fig1)

    fig2 = plt.figure()
    df.groupby("Job_Role")["Salary"].mean().plot(kind="bar")
    plt.title("Average Salary by Job Role")
    st.pyplot(fig2)

# ----------------------------
# MODEL TRAINING
# ----------------------------
elif menu == "Train Model":
    st.header("Model Training")

    df_model = df.copy()

    le_job = LabelEncoder()
    le_edu = LabelEncoder()

    df_model["Job_Role"] = le_job.fit_transform(df_model["Job_Role"])
    df_model["Education"] = le_edu.fit_transform(df_model["Education"])

    X = df_model.drop("Salary", axis=1)
    y = df_model["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.success("Model Trained Successfully!")

    st.write("R2 Score:", r2_score(y_test, y_pred))
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# ----------------------------
# PREDICTION SECTION
# ----------------------------
elif menu == "Predict Salary":
    st.header("Salary Prediction")

    age = st.slider("Age", 18, 60, 25)
    experience = st.slider("Experience (Years)", 0, 40, 1)
    education = st.selectbox("Education", df["Education"].unique())
    job = st.selectbox("Job Role", df["Job_Role"].unique())

    # Encode
    df_model = df.copy()
    le_job = LabelEncoder()
    le_edu = LabelEncoder()

    df_model["Job_Role"] = le_job.fit_transform(df_model["Job_Role"])
    df_model["Education"] = le_edu.fit_transform(df_model["Education"])

    X = df_model.drop("Salary", axis=1)
    y = df_model["Salary"]

    model = RandomForestRegressor()
    model.fit(X, y)

    job_encoded = le_job.transform([job])[0]
    edu_encoded = le_edu.transform([education])[0]

    input_data = np.array([[age, experience, edu_encoded, job_encoded]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: ₹ {int(prediction[0])}")

# ----------------------------
# ABOUT
# ----------------------------
elif menu == "About":
    st.header("About Project")
    st.write("""
    This is a Medium Level Machine Learning Project.

    Features:
    - Interactive Dashboard
    - Data Visualization
    - Model Training
    - Real-time Salary Prediction
    
    Developed using Streamlit.
    """)
