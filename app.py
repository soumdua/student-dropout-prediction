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

    st.markdown("""
### Dataset and Prediction Task

This project analyzes a dataset containing information about university students and their academic and personal characteristics in order to predict whether a student is likely to drop out of their program. Each row in the dataset represents an individual student, while the columns represent features that describe aspects of the student’s background, behavior, and academic performance. These features include demographic attributes such as gender and age, academic information such as semester year and grades, and engagement indicators such as attendance, study hours, and participation in academic activities.

The target variable in this analysis is **student dropout status**, which indicates whether a student continues their education or leaves the program before completing their degree. By examining patterns within the data, the model aims to classify students into two groups: those who are likely to remain enrolled and those who may be at risk of dropping out. The goal is to use the available features to identify factors that are associated with higher dropout risk and generate predictions that can help institutions better understand student retention.

---

### Why This Problem Matters

Student dropout is an important issue for educational institutions because it affects both students and universities. When students leave their programs before graduating, they may experience financial loss, delayed career opportunities, and reduced long-term earning potential. For universities, high dropout rates can negatively impact graduation statistics, institutional reputation, and funding that may be tied to student success metrics.

Predictive analytics provides a way for institutions to address this challenge proactively. Instead of waiting until students have already disengaged or failed courses, universities can use data to identify early warning signs. By detecting students who may be at risk, institutions can provide targeted support such as academic advising, tutoring, mentoring, or financial guidance. This type of data-driven intervention can help improve retention rates and support students in completing their education successfully.

---

### Approach and Key Findings

To address the prediction task, the dataset was first explored and cleaned to ensure that all features were suitable for modeling. Categorical variables were encoded into numerical form so they could be used by the machine learning algorithm. Exploratory visualizations were created to better understand the relationships between features such as study habits, academic performance, and semester progression with student dropout outcomes.

A **logistic regression classification model** was then trained to predict whether a student is likely to drop out. Logistic regression was chosen because it is interpretable and allows us to understand how different features influence the probability of dropout. The model produces predictions that estimate the likelihood of a student leaving their program based on the characteristics available in the dataset.

To make the model more transparent and interpretable, **SHAP (SHapley Additive exPlanations)** was used to analyze which features contribute most strongly to the predictions. This helps translate the model’s results into insights that stakeholders can understand. Overall, the analysis demonstrates how student data can be used to identify patterns associated with dropout risk and highlights how predictive modeling can support earlier and more informed interventions aimed at improving student success and retention.
""")
    st.write("Dataset shape:", df.shape)


# -----------------------
# TAB 2
# -----------------------

with tab2:

    st.header("Descriptive Analytics")

    st.subheader("Dropout Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Student Dropout Outcomes")
    ax.set_xlabel("Dropout Status")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)

    st.write("""
    This bar chart shows the number of students who dropped out compared to those who remained enrolled. 
    Understanding this distribution helps determine whether the dataset is balanced and whether the model may need to account for class imbalance when making predictions.
    """)


    st.subheader("Age Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)

    st.pyplot(fig)

    st.write("""
    This histogram shows the distribution of student ages in the dataset. 
    It helps identify whether certain age groups are more common and whether age may be a relevant factor associated with dropout patterns.
    """)


    st.subheader("Gender Distribution")

    fig, ax = plt.subplots()
    df["Gender"].value_counts().plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.write("""
    This chart shows the number of students in each gender category represented in the dataset. 
    Examining demographic distributions helps determine whether the dataset represents students evenly and whether gender may influence dropout outcomes.
    """)


    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.write("""
    The correlation heatmap shows relationships between numerical variables in the dataset. 
    Strong positive or negative correlations highlight features that may have meaningful relationships with student dropout or other academic characteristics.
    """)

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