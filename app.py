import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.title("Student Dropout Prediction Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("student_dropout_dataset_v3.csv")

# Drop ID like in notebook
df = df.drop("Student_ID", axis=1)

# Encode categorical columns
categorical = [
    "Gender",
    "Internet_Access",
    "Part_Time_Job",
    "Scholarship",
    "Department",
    "Parental_Education"
]

encoders = {}

for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("logistic_model.pkl")

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# -----------------------------
# TABS
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction"
])

# -----------------------------
# TAB 1
# -----------------------------

with tab1:

    st.header("Executive Summary")

    st.write("""
    This project predicts student dropout risk using academic, behavioral,
    and demographic information.

    Universities can use this model to identify at-risk students early
    and provide support interventions before dropout occurs.

    Multiple models were tested including Logistic Regression,
    Decision Tree, Random Forest, and Neural Networks.
    """)

    st.write("Dataset shape:", df.shape)

# -----------------------------
# TAB 2
# -----------------------------

with tab2:

    st.header("Descriptive Analytics")

    st.subheader("Target Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Student Dropout Distribution")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------
# TAB 3
# -----------------------------

with tab3:

    st.header("Model Performance")

    performance = pd.DataFrame({
        "Model":["Logistic Regression","Decision Tree","Random Forest","MLP"],
        "Accuracy":[0.84,0.78,0.82,0.80]
    })

    st.table(performance)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=performance, ax=ax)
    st.pyplot(fig)

# -----------------------------
# TAB 4
# -----------------------------

with tab4:

    st.header("Explainability & Interactive Prediction")

    # -----------------------------
    # SHAP
    # -----------------------------

    st.subheader("SHAP Feature Importance")

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(fig)

    # -----------------------------
    # USER INPUTS
    # -----------------------------

    st.subheader("Interactive Prediction")

    age = st.slider("Age", 17, 40, 21)
    gender = st.selectbox("Gender", ["Male","Female"])
    income = st.slider("Family Income", 1000, 10000, 4000)
    internet = st.selectbox("Internet Access", ["Yes","No"])
    study = st.slider("Study Hours per Day", 0, 12, 4)
    attendance = st.slider("Attendance Rate", 0, 100, 75)
    delay = st.slider("Assignment Delay Days", 0, 10, 2)
    travel = st.slider("Travel Time Minutes", 0, 120, 30)
    job = st.selectbox("Part Time Job", ["Yes","No"])
    scholarship = st.selectbox("Scholarship", ["Yes","No"])
    stress = st.slider("Stress Index", 0, 10, 5)
    gpa = st.slider("GPA", 0.0, 4.0, 3.0)
    sem_gpa = st.slider("Semester GPA", 0.0, 4.0, 3.0)
    cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)
    semester = st.slider("Semester", 1, 8, 4)
    dept = st.selectbox("Department", df["Department"].unique())
    parent_edu = st.selectbox("Parental Education", df["Parental_Education"].unique())

    # Encode categorical inputs
    gender = encoders["Gender"].transform([gender])[0]
    internet = encoders["Internet_Access"].transform([internet])[0]
    job = encoders["Part_Time_Job"].transform([job])[0]
    scholarship = encoders["Scholarship"].transform([scholarship])[0]
    dept = encoders["Department"].transform([dept])[0]
    parent_edu = encoders["Parental_Education"].transform([parent_edu])[0]

    input_data = pd.DataFrame([[
        age, gender, income, internet, study, attendance,
        delay, travel, job, scholarship, stress,
        gpa, sem_gpa, cgpa, semester, dept, parent_edu
    ]], columns=X.columns)

    if st.button("Predict"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Dropout Risk ({probability:.2f})")
        else:
            st.success(f"✅ Low Dropout Risk ({probability:.2f})")