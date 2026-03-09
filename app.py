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

    st.subheader("Target Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


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

    st.header("Explainability & Interactive Prediction")

    st.subheader("SHAP Feature Importance")

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(fig)

    st.subheader("Interactive Prediction")

    # use dataset means as defaults
    user_input = {}

    for col in X.columns:
        user_input[col] = st.number_input(
            col,
            value=float(X[col].mean())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.write("Prediction:", prediction)
        st.write("Dropout Probability:", round(probability,3))