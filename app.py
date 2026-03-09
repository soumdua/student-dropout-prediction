import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.title("Student Dropout Prediction Dashboard")

# -----------------------
# LOAD DATA
# -----------------------

df = pd.read_csv("student_dropout_dataset_v3.csv")

# drop ID like notebook
if "Student_ID" in df.columns:
    df = df.drop("Student_ID", axis=1)

# -----------------------
# LOAD MODEL
# -----------------------

model = joblib.load("logistic_model.pkl")

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# -----------------------
# TABS
# -----------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction"
])

# -----------------------
# TAB 1
# -----------------------

with tab1:

    st.header("Executive Summary")

    st.write("""
    This project predicts student dropout risk using academic,
    behavioral, and demographic data.

    Universities can identify at-risk students early and intervene
    before dropout occurs.
    """)

    st.write("Dataset shape:", df.shape)


# -----------------------
# TAB 2
# -----------------------

with tab2:

    st.header("Descriptive Analytics")

    # -----------------------------
    # Target Distribution
    # -----------------------------
    st.subheader("Dropout Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Student Dropout")
    ax.set_xlabel("Dropout")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write("This plot shows how many students dropped out versus stayed enrolled.")

    # -----------------------------
    # GPA vs Dropout
    # -----------------------------
    st.subheader("GPA vs Dropout")

    fig, ax = plt.subplots()
    sns.boxplot(x="Dropout", y="GPA", data=df, ax=ax)
    ax.set_title("GPA Distribution by Dropout Status")
    st.pyplot(fig)

    st.write("Students with lower GPA appear more likely to drop out.")

    # -----------------------------
    # Attendance vs Dropout
    # -----------------------------
    st.subheader("Attendance Rate vs Dropout")

    fig, ax = plt.subplots()
    sns.boxplot(x="Dropout", y="Attendance_Rate", data=df, ax=ax)
    ax.set_title("Attendance Rate by Dropout Status")
    st.pyplot(fig)

    st.write("Students with lower attendance rates show higher dropout probability.")

    # -----------------------------
    # Study Hours vs Dropout
    # -----------------------------
    st.subheader("Study Hours per Day vs Dropout")

    fig, ax = plt.subplots()
    sns.boxplot(x="Dropout", y="Study_Hours_per_Day", data=df, ax=ax)
    ax.set_title("Study Hours by Dropout Status")
    st.pyplot(fig)

    st.write("Students studying fewer hours per day tend to drop out more frequently.")

    # -----------------------------
    # Stress Index vs Dropout
    # -----------------------------
    st.subheader("Stress Index vs Dropout")

    fig, ax = plt.subplots()
    sns.boxplot(x="Dropout", y="Stress_Index", data=df, ax=ax)
    ax.set_title("Stress Index by Dropout Status")
    st.pyplot(fig)

    st.write("Higher stress levels may be associated with higher dropout risk.")

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes("number").corr(),
                cmap="coolwarm",
                ax=ax)
    st.pyplot(fig)

    st.write("This heatmap shows relationships between numerical variables in the dataset.")
# -----------------------
# TAB 3
# -----------------------

with tab3:

    st.header("Model Performance")

    performance = pd.DataFrame({
        "Model":["Logistic Regression","Decision Tree","Random Forest","MLP"],
        "F1 Score":[0.82,0.75,0.80,0.78]
    })

    st.table(performance)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="F1 Score", data=performance, ax=ax)
    st.pyplot(fig)


# -----------------------
# TAB 4
# -----------------------

with tab4:

    st.header("Model Explainability (SHAP)")
    st.write("This section shows which features influence dropout prediction.")

    # Encode categorical variables
    df_encoded = pd.get_dummies(df)

    # Features expected by the model
    feature_names = model.feature_names_in_

    # Add missing columns that model expects
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep only model features
    X = df_encoded[feature_names]

    # Convert to numeric values only
    X = X.astype(float)

    try:
        explainer = shap.LinearExplainer(model, X)

        shap_values = explainer(X)

        fig = plt.figure()
        shap.summary_plot(shap_values.values, X, show=False)

        st.pyplot(fig)

    except Exception as e:
        st.error("SHAP visualization failed.")
        st.write(e)