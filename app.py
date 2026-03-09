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

    st.write("This section explores patterns in the dataset to better understand factors associated with student dropout.")

    # Dropout distribution
    st.subheader("Dropout Distribution")

    fig, ax = plt.subplots()
    df["Dropout"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Dropout Status")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)

    st.write("""
    This chart shows the number of students who dropped out compared to those who remained enrolled. 
    Understanding this distribution helps determine whether the dataset is balanced and provides context for interpreting model performance.
    """)


    # Age distribution
    st.subheader("Age Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)

    st.pyplot(fig)

    st.write("""
    This histogram shows how student ages are distributed in the dataset. 
    Examining the age distribution helps determine whether certain age groups may be more associated with dropout risk.
    """)


    # Gender distribution
    st.subheader("Gender Distribution")

    fig, ax = plt.subplots()
    df["Gender"].value_counts().plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.write("""
    This plot shows the number of students in each gender category. 
    Demographic distributions provide insight into whether the dataset is representative and whether gender differences may relate to dropout patterns.
    """)


    # Study hours distribution
    if "StudyHours" in df.columns:

        st.subheader("Study Hours Distribution")

        fig, ax = plt.subplots()
        sns.histplot(df["StudyHours"], bins=20, kde=True, ax=ax)

        st.pyplot(fig)

        st.write("""
        This histogram shows how many hours students typically spend studying each week. 
        Study habits can influence academic success and may help explain patterns in student retention or dropout.
        """)


    # GPA distribution
    if "GPA" in df.columns:

        st.subheader("GPA Distribution")

        fig, ax = plt.subplots()
        sns.histplot(df["GPA"], bins=20, kde=True, ax=ax)

        st.pyplot(fig)

        st.write("""
        This plot shows the distribution of student grade point averages in the dataset. 
        Academic performance is often a strong indicator of whether students remain enrolled or are at risk of dropping out.
        """)


    # Dropout by gender
    if "Gender" in df.columns:

        st.subheader("Dropout Rate by Gender")

        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Gender", hue="Dropout", ax=ax)

        st.pyplot(fig)

        st.write("""
        This chart compares dropout outcomes across different gender groups. 
        By examining these differences, we can identify whether certain demographic groups experience higher dropout rates.
        """)


    # Dropout by semester
    if "Semester_Year" in df.columns:

        st.subheader("Dropout by Semester Year")

        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Semester_Year", hue="Dropout", ax=ax)

        st.pyplot(fig)

        st.write("""
        This visualization shows how dropout outcomes vary across different stages of the academic program. 
        It can reveal whether students are more likely to leave earlier or later in their academic journey.
        """)


    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64","float64"])

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.write("""
    The correlation heatmap illustrates relationships between numerical features in the dataset. 
    Strong positive or negative correlations may indicate variables that are important for predicting student dropout.
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

    st.header("Explainability & Interactive Prediction")

st.markdown("""
### Understanding Model Predictions

Machine learning models can make accurate predictions, but it is equally important to understand **why** the model makes those predictions. In this section, explainability techniques are used to show how different student characteristics influence the model's decision when predicting dropout risk. This helps make the model more transparent and allows stakeholders to see which factors have the strongest impact on predictions.

To provide this transparency, the app uses **SHAP (SHapley Additive exPlanations)**. SHAP is a widely used method for interpreting machine learning models because it assigns a contribution value to each feature in the dataset. These values show whether a feature increases or decreases the probability that a student will drop out. By visualizing these contributions, we can better understand how factors such as academic performance, attendance, or study habits affect the model’s predictions.

### Interactive Prediction Tool

This tab also includes an interactive prediction tool that allows users to experiment with different student profiles. By adjusting the input values using sliders or dropdown menus, users can simulate how changes in student characteristics affect the predicted dropout risk. For example, increasing study hours or improving attendance may reduce the predicted probability of dropping out.

Once the inputs are selected, the model generates a real-time prediction showing whether the student is likely to remain enrolled or be at risk of dropping out. The app also displays the predicted probability associated with the outcome.

### Explaining Individual Predictions

After generating a prediction, a SHAP waterfall plot explains how each feature contributed to the model’s decision for that specific student profile. Features that push the prediction toward a higher dropout risk appear in red, while features that reduce the risk appear in blue. This visualization helps users understand which factors played the largest role in the prediction and provides insight into how the model evaluates student characteristics.
""")

    # -----------------------------
    # Prepare data for SHAP
    # -----------------------------
    df_encoded = pd.get_dummies(df)

    feature_names = model.feature_names_in_

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_names].astype(float)

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    # -----------------------------
    # SHAP Summary Plot
    # -----------------------------
    st.subheader("SHAP Summary Plot")

    fig = plt.figure()
    shap.summary_plot(shap_values.values, X, show=False)

    st.pyplot(fig)

    st.write("""
    The SHAP summary plot shows which features most influence the model’s predictions 
    across the entire dataset. Features at the top of the plot have the largest impact 
    on whether the model predicts that a student will drop out.
    """)

    # -----------------------------
    # SHAP Bar Importance Plot
    # -----------------------------
    st.subheader("SHAP Feature Importance")

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)

    st.pyplot(fig)

    st.write("""
    This bar chart ranks the features based on their overall importance to the model. 
    The higher the bar, the greater the influence that feature has on predicting student dropout risk.
    """)

    # -----------------------------
    # Interactive Prediction
    # -----------------------------
    st.subheader("Interactive Prediction")

    st.write("Adjust the student characteristics below to see how the model prediction changes.")

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression"]
    )

    age = st.slider(
        "Age",
        int(df["Age"].min()),
        int(df["Age"].max()),
        int(df["Age"].mean())
    )

    study_hours = st.slider(
        "Study Hours per Week",
        0,
        40,
        10
    )

    attendance = st.slider(
        "Attendance Rate",
        0,
        100,
        int(df["Attendance_Rate"].mean())
    )

    # -----------------------------
    # Create input row
    # -----------------------------
    input_df = X.mean().to_frame().T

    input_df["Age"] = age
    input_df["StudyHours"] = study_hours
    input_df["Attendance_Rate"] = attendance

    input_df = input_df[feature_names]

    # -----------------------------
    # Make prediction
    # -----------------------------
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Prediction: Student likely to drop out (Probability: {probability:.2f})")
    else:
        st.success(f"Prediction: Student likely to remain enrolled (Probability: {probability:.2f})")

    # -----------------------------
    # SHAP explanation for user input
    # -----------------------------
    st.subheader("SHAP Explanation for This Prediction")

    user_shap = explainer(input_df)

    fig = plt.figure()

    shap.plots.waterfall(user_shap[0], show=False)

    st.pyplot(fig)

    st.write("""
    The SHAP waterfall plot explains how each feature contributed to this specific prediction. 
    Features pushing the prediction toward dropout risk appear in red, while features that decrease the risk appear in blue.
    """)