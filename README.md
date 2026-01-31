# Heart Disease Classification using Machine Learning and Streamlit

This project implements multiple machine learning classification models to predict the presence of heart disease using clinical patient data. A Streamlit web application is developed to allow interactive testing, model comparison, and visualization of evaluation metrics.

The app is deployed using Streamlit Community Cloud.

---

## Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide.  
The objective of this project is to build and compare multiple machine learning classification models that predict whether a patient has heart disease based on clinical attributes such as age, cholesterol level, chest pain type, resting blood pressure, and exercise-induced angina.

---

## Dataset Description

The dataset used is the UCI Heart Disease dataset obtained via Kaggle.

- Source: Kaggle (UCI Heart Disease Dataset)
- Total Instances: 920
- Total Features: 16 (after preprocessing and encoding)
- Target Column: `num`
  - 0 → No heart disease
  - 1 → Presence of heart disease

Preprocessing steps:
- Dropped non-informative `id` column
- Converted multiclass target to binary
- One-hot encoded categorical variables
- Imputed missing values using median strategy
- Feature scaling using StandardScaler

---

## Models Used and Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 |
|------|---------|-----|----------|--------|------|
| Logistic Regression | 0.821 | 0.922 | 0.811 | 0.882 | 0.845 |
| Decision Tree | 0.766 | 0.757 | 0.761 | 0.843 | 0.800 |
| KNN | 0.848 | 0.901 | 0.830 | 0.912 | 0.869 |
| Naive Bayes | 0.853 | 0.910 | 0.871 | 0.863 | 0.867 |
| Random Forest | 0.842 | 0.923 | 0.835 | 0.892 | 0.863 |
| XGBoost | 0.842 | 0.895 | 0.848 | 0.873 | 0.860 |

---

## Observations

| Model | Observation |
|------|-----------|
| Logistic Regression | Strong baseline with high AUC and recall, indicating good generalization ability. |
| Decision Tree | Lower performance compared to other models, possibly due to overfitting. |
| KNN | Performed very well after feature scaling, achieving high recall. |
| Naive Bayes | Simple probabilistic model with competitive accuracy. |
| Random Forest | Achieved the highest AUC among ensemble models, showing robustness. |
| XGBoost | Strong ensemble performance but slightly lower AUC compared to Random Forest. |


## Project Structure

