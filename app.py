import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor 💼", page_icon="💡", layout="centered")

# -----------------------------
# Load model and preprocessors
# -----------------------------
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }
        .main {
            background-color: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .stSlider label, .stNumberInput label, .stSelectbox label {
            font-weight: bold;
            color: #E0E0E0;
        }
        h1 {
            text-align: center;
            color: #F5B041;
        }
        .prediction-card {
            background-color: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: #FFF;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# App Header
# -----------------------------
st.markdown("<h1>💼 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether a customer will **leave your company** based on their details. Adjust inputs below 👇")

# -----------------------------
# Input Section (Two Columns)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👫 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=650)
    balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0)

with col2:
    estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0)
    tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 2)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox('✅ Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Churn", use_container_width=True):
    with st.spinner('Analyzing customer data...'):
        time.sleep(1.5)
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
    
    st.subheader("📊 Prediction Result")

    # Animated progress bar
    st.progress(int(prediction_proba * 100))
    time.sleep(0.5)
    
    # Display prediction card
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction_proba > 0.5:
        st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color:#E74C3C;'>⚠️ High Risk of Churn</h2>
            <p>Probability: <b>{prediction_proba:.2f}</b></p>
            <p>This customer is <b>likely to leave</b>. Consider loyalty programs or personalized offers.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color:#2ECC71;'>✅ Low Risk of Churn</h2>
            <p>Probability: <b>{prediction_proba:.2f}</b></p>
            <p>This customer is <b>likely to stay</b>. Keep maintaining engagement and satisfaction!</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
👨‍💻 Developed by **CyberConqueror**  
💬 Powered by TensorFlow + Streamlit
""")
