import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

# Load dataset
data = load_breast_cancer()

feature_names = data.feature_names
target_names = data.target_names

# Load trained model
saved_data = joblib.load("breast_cancer_model.pkl")

# Support both dictionary and direct model formats
if isinstance(saved_data, dict):
    model = saved_data.get("model", saved_data.get("pipeline"))
    if model is None:
        st.error("Model not found in the saved file.")
        st.stop()
    if "feature_names" in saved_data:
        feature_names = saved_data["feature_names"]
else:
    model = saved_data

# Calculate default values for input fields
default_values = np.median(data.data, axis=0)

# Sidebar navigation
st.sidebar.title("🎗️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Prediction", "Model Performance", "About"]
)

# HOME PAGE
if page == "Home":

    st.title("🎗️ Breast Cancer Detection System")

    st.subheader("Machine Learning Based Prediction")

    st.write(
        "This application uses Logistic Regression to classify "
        "breast cancer tumours as Benign or Malignant."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy", "98.25%")
    col2.metric("Dataset Records", "569")
    col3.metric("Algorithm", "Logistic Regression")

    st.info(
        "Navigate to the Prediction page to enter tumour measurements "
        "and obtain a model prediction."
    )

    st.warning(
        "This application is for academic demonstration only. "
        "It is not a medical diagnostic tool."
    )


# PREDICTION PAGE
elif page == "Prediction":

    st.title("🔍 Breast Cancer Prediction")

    st.write(
        "Enter the 30 numerical tumour measurements below "
        "and click Predict Result."
    )

    st.info(
        "Default values are dataset medians for demonstration. "
        "They are not measurements from an individual patient."
    )

    feature_values = []

    with st.form("prediction_form"):

        st.subheader("1. Mean Values")

        mean_values = {}

        col1, col2 = st.columns(2)

        for i in range(10):
            with col1 if i < 5 else col2:
                value = st.number_input(
                    feature_names[i],
                    min_value=0.0,
                    value=float(default_values[i]),
                    format="%.5f",
                    key=f"feature_{i}"
                )
                mean_values[i] = value

        st.divider()

        st.subheader("2. Standard Error Values")

        col1, col2 = st.columns(2)

        for i in range(10, 20):
            with col1 if i < 15 else col2:
                value = st.number_input(
                    feature_names[i],
                    min_value=0.0,
                    value=float(default_values[i]),
                    format="%.5f",
                    key=f"feature_{i}"
                )
                mean_values[i] = value

        st.divider()

        st.subheader("3. Worst Values")

        col1, col2 = st.columns(2)

        for i in range(20, 30):
            with col1 if i < 25 else col2:
                value = st.number_input(
                    feature_names[i],
                    min_value=0.0,
                    value=float(default_values[i]),
                    format="%.5f",
                    key=f"feature_{i}"
                )
                mean_values[i] = value

        predict_button = st.form_submit_button(
            "🔍 Predict Result",
            use_container_width=True
        )

    if predict_button:

        input_data = np.array(
            [mean_values[i] for i in range(30)]
        ).reshape(1, -1)

        try:
            prediction = int(model.predict(input_data)[0])

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 0:
                st.error("⚠️ Prediction: MALIGNANT")
            else:
                st.success("✅ Prediction: BENIGN")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]

                st.write("Prediction probabilities:")

                for i, probability in enumerate(probabilities):
                    st.write(
                        f"{target_names[i].capitalize()}: "
                        f"{probability * 100:.2f}%"
                    )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# MODEL PERFORMANCE PAGE
elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.metric("Model Accuracy", "98.25%")

    st.subheader("Classification Report")

    report = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Precision": [0.98, 0.99],
        "Recall": [0.98, 0.99],
        "F1 Score": [0.98, 0.99]
    })

    st.dataframe(report, use_container_width=True)

    st.subheader("Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [[41, 1], [1, 71]],
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )

    st.dataframe(confusion_matrix, use_container_width=True)

    st.write("Training records: 455")
    st.write("Testing records: 114")


# ABOUT PAGE
elif page == "About":

    st.title("ℹ️ About the Project")

    st.write("""
    **Project Title:** Breast Cancer Detection Using Machine Learning

    **Algorithm:** Logistic Regression

    **Programming Language:** Python

    **Framework:** Streamlit

    **Dataset:** Wisconsin Diagnostic Breast Cancer Dataset

    **Total Records:** 569

    **Total Features:** 30

    **Output Classes:**
    - Benign
    - Malignant
    """)

    st.info(
        "This project demonstrates how machine learning "
        "can classify breast cancer data."
    )