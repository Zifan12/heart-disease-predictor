"""
Heart Disease Risk Predictor Streamlit App
------------------------------------------
This streamlit application uses a trained XGBoost pipeline to predict the probablity of heart diseas
based on user inputs of patient data. Uses SHAP values to provide model interpretability.
"""


import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/best_xgb_pipeline.pkl")

# Add title/little description
st.title("Heart Disease Risk Predictor")
st.markdown("Predict the likelihood of heart disease based on patient input.")

# Input Form Header
st.sidebar.header("Patient Information")

def get_user_input():
    """
    Collects user input from Streamlit sidebar for heartdisease prediction.
    
    Returns:
        pd.DataFrame: A DataFrame that contains the patient's input features, where
        it matches the model's expected input structure.
    """
    fs_choice = st.sidebar.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dL", "> 120 mg/dL"])

    data = {
        'Age': st.sidebar.slider('Age', 20, 100, 50),
        'Sex': st.sidebar.selectbox('Sex', ['M', 'F']),
        'ChestPainType': st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA']),
        'RestingBP': st.sidebar.slider('Resting Blood Pressure', 80, 200, 120),
        'Cholesterol': st.sidebar.slider('Cholesterol', 100, 600,200),
        'FastingBS': 1 if fs_choice == "> 120 mg/dL" else 0,
        'RestingECG': st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH']),
        'MaxHR': st.sidebar.slider("Maximum Heart Rate", 60, 202, 150),
        'ExerciseAngina': st.sidebar.selectbox('Exercise-induced Angina', ['Y', 'N ']),
        'Oldpeak': st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0),
        'ST_Slope': st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down']),
    
    }
    return pd.DataFrame([data])

# Get user input
df_input = get_user_input()

# Prediction
st.subheader("Prediction")

# Predict Probability
prob = model.predict_proba(df_input)[0][1]
pred = model.predict(df_input)[0]

st.write(f"**Probability of Heart Disease:** `{prob:.2f}`")
st.write(f"**Prediction:**", "  **🟥 Heart Disease**" if pred == 1 else "**🟩 No Heart Disease**")

# Get components from pipeline
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# Get the transformed input
X_transformed = preprocessor.transform(df_input)

# Get Feature names
feature_names = preprocessor.get_feature_names_out()

# Clean unnecessary prefixes for feature_names
clean_feature_names = []
for name in feature_names:
    clean_name = name.split('__')[-1]
    clean_feature_names.append(clean_name)


# SHAP 
st.subheader("Model Explanation (SHAP)")

explainer = shap.Explainer(classifier)
shap_values = explainer(X_transformed)

shap_values.feature_names = clean_feature_names

st.markdown("""
**SHAP Explanation**: The plot below shows how each feature influenced the model's prediction.
            
- 🟥 **Positive values (red)** push the model toward predicting **heart disease**.
- 🟦 **Negative values (blue)** push the model toward predicting **no heart disease**.
            
Each bar represents the impact of a feature on the prediction for this specific patient.
            """)
# Waterfall plot
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(plt.gcf())