import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.title("Student Dropout Prediction Dashboard")

# Load dataset
df = pd.read_csv("student_dropout_dataset_v3.csv")

# Load trained model
model = joblib.load("logistic_model.pkl")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction"
])

with tab1:
    st.header("Executive Summary")

    st.write("""
    This project predicts whether a student will drop out based on academic,
    behavioral, and demographic characteristics.

    Early identification of at-risk students allows universities to intervene
    earlier and improve retention outcomes.

    Multiple machine learning models were trained and compared including
    Logistic Regression, Decision Tree, Random Forest, and Neural Networks.
    The best-performing model was selected based on evaluation metrics such
    as accuracy, precision, recall, F1-score, and ROC-AUC.
    """)

    st.write("Dataset shape:", df.shape)


with tab2:

    st.header("Descriptive Analytics")

    st.subheader("Target Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Dropout Distribution")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab3:

    st.header("Model Performance")

    performance = pd.DataFrame({
        "Model":["Logistic Regression","Decision Tree","Random Forest","MLP"],
        "F1 Score":[0.82,0.75,0.80,0.78]
    })

    st.table(performance)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="F1 Score", data=performance, ax=ax)
    ax.set_title("Model Comparison")
    st.pyplot(fig)

with tab4:

    st.header("Explainability & Interactive Prediction")

    st.subheader("SHAP Feature Importance")

    X = df.drop("Dropout", axis=1)
    y = df["Dropout"]

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(fig)

    st.subheader("Interactive Prediction")

   st.subheader("Interactive Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
job = st.selectbox("Part Time Job", ["Yes", "No"])
scholarship = st.selectbox("Scholarship", ["Yes", "No"])

age = st.slider("Age", 18, 40, 22)
study_hours = st.slider("Study Hours per Day", 0, 10, 4)
attendance = st.slider("Attendance (%)", 0, 100, 75)
gender = 1 if gender == "Male" else 0
internet = 1 if internet == "Yes" else 0
job = 1 if job == "Yes" else 0
scholarship = 1 if scholarship == "Yes" else 0

input_data = pd.DataFrame({
    "Gender":[gender],
    "Internet_Access":[internet],
    "Part_Time_Job":[job],
    "Scholarship":[scholarship],
    "Age":[age],
    "StudyHours":[study_hours],
    "Attendance":[attendance]
})

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.write("Prediction:", prediction)
        st.write("Dropout Probability:", round(probability,3))