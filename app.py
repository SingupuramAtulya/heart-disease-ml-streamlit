import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title(" Heart Disease Classification – ML Model Comparison")

st.write(
    """
Upload a **CSV test dataset**, select a trained model, and view
performance metrics and confusion matrix.

Models supported:
- Logistic Regression  
- Decision Tree  
- KNN  
- Naive Bayes  
- Random Forest  
- XGBoost  
"""
)

scaler = joblib.load("model/scaler.pkl")
trained_columns = joblib.load("model/training_columns.pkl")

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}


uploaded_file = st.file_uploader("Upload CSV test dataset", type=["csv"])
model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader(" Uploaded Data Preview")
    st.dataframe(df.head())

    TARGET_COL = "num"

    if TARGET_COL not in df.columns:
        st.error("Target column 'num' not found in uploaded file.")
        st.stop()

    # Convert target to binary
    df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if x > 0 else 0)

    # Drop ID if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]

    # One-hot encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Align columns with training schema
    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[trained_columns]


    # Fill NaNs with median
    X = X.fillna(X.median())

    X_array = X.values.astype(float)
    X_scaled = scaler.transform(X_array)

    # Load model
    model = joblib.load(MODEL_PATHS[model_name])

    st.subheader(f"📊 Results for {model_name}")

    # Predict
    if model_name in ["Logistic Regression", "KNN"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X_array)
        y_prob = model.predict_proba(X_array)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", round(acc, 3))
        st.metric("AUC", round(auc, 3))
        st.metric("Precision", round(prec, 3))

    with col2:
        st.metric("Recall", round(rec, 3))
        st.metric("F1 Score", round(f1, 3))


    st.subheader(" Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

else:
    st.info(" Upload a CSV file to begin.")
