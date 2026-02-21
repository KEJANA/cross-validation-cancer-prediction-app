import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

st.title("Cancer Prediction with Cross Validation")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# Cross validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
st.write("Mean Cross Validation Accuracy:", round(cv_scores.mean(), 2))

st.subheader("Enter Patient Details")

radius = st.number_input("Mean Radius", float(X.mean()[0]))
texture = st.number_input("Mean Texture", float(X.mean()[1]))
perimeter = st.number_input("Mean Perimeter", float(X.mean()[2]))
area = st.number_input("Mean Area", float(X.mean()[3]))

# Prepare input
input_data = [radius, texture, perimeter, area] + list(X.mean()[4:])
input_data = scaler.transform([input_data])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    if pred == 0:
        st.error("Malignant (Cancerous)")
    else:
        st.success("Benign (Non-Cancerous)")