import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# --- UI Configuration ---
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

# --- Custom CSS for Theming ---
st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #f9fbf9;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #4a7c59 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 6px rgba(74, 124, 89, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #3b6347 !important;
        box-shadow: 0 6px 12px rgba(74, 124, 89, 0.3) !important;
        transform: translateY(-2px) !important;
        color: white !important;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #4a7c59 0%, #2c4a35 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(44, 74, 53, 0.3);
        margin-top: 1rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-card h2 {
        color: #ffffff !important;
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Typography */
    h1 {
        color: #2c4a35 !important;
        font-weight: 800 !important;
        padding-bottom: 1rem;
    }
    h3 {
        color: #4a7c59 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading & Training ---
@st.cache_resource
def load_and_train_model():
    # 1. Load dataset
    df = pd.read_csv("enhanced_crop_yield_dataset.csv")
    
    # 2. Separate features and target
    y = df["Crop Yield (kg per hectare)"]
    X_raw = df.drop("Crop Yield (kg per hectare)", axis=1)
    
    # 3. One-hot encode categorical columns
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    training_cols = X_encoded.columns 
    
    # 4. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # 5. Dimensionality Reduction (PCA)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # 6. Clustering-based feature engineering (DBSCAN)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_pca)
    
    # Add cluster labels as feature
    X_final = np.c_[X_pca, clusters]
    
    # 7. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    # 8. Train the Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    return lr, scaler, pca, training_cols, X_pca, clusters, df

# Load the pipeline components
lr, scaler, pca, training_cols, X_pca_train, clusters_train, df_raw = load_and_train_model()

# --- Header ---
st.title("🌾 Crop Yield Forecaster")
st.markdown("<p style='font-size: 1.2rem; color: #556b58; margin-bottom: 2rem;'>Enter the agricultural and environmental parameters below to generate an AI-driven crop yield prediction.</p>", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🌦️ Environmental Factors")
        rainfall = st.number_input("Rainfall (mm)", value=float(df_raw['rainfall'].mean()), format="%.2f")
        temperature = st.number_input("Temperature (°C)", value=float(df_raw['temperature'].mean()), format="%.2f")
        humidity = st.number_input("Humidity (%)", value=float(df_raw['humidity'].mean()), format="%.2f")
        
    with st.container(border=True):
        st.subheader("🧪 Soil Nutrients")
        n_val = st.number_input("Nitrogen (N)", value=float(df_raw['N'].mean()), format="%.2f")
        p_val = st.number_input("Phosphorus (P)", value=float(df_raw['P'].mean()), format="%.2f")
        k_val = st.number_input("Potassium (K)", value=float(df_raw['K'].mean()), format="%.2f")

with col2:
    with st.container(border=True):
        st.subheader("🚜 Agricultural Details")
        ph = st.number_input("Soil pH", value=float(df_raw['pH'].mean()), format="%.2f")
        
        soil_types = sorted(df_raw['Soil_Type'].dropna().unique().tolist())
        soil_type = st.selectbox("Soil Type", soil_types)
        
        crops = sorted(df_raw['Crop'].dropna().unique().tolist())
        crop = st.selectbox("Crop", crops)
        
        irrigation_methods = sorted(df_raw['Irrigation_Method'].dropna().unique().tolist())
        irrigation_method = st.selectbox("Irrigation Method", irrigation_methods)

st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Logic ---
if st.button("Generate Yield Prediction", use_container_width=True):
    
    # 1. Initialize user input with default values
    user_input = {}
    for col in df_raw.drop("Crop Yield (kg per hectare)", axis=1).columns:
        if df_raw[col].dtype == 'object':
            user_input[col] = df_raw[col].mode()[0]
        else:
            user_input[col] = df_raw[col].mean()
            
    # 2. Update dictionary
    user_input['rainfall'] = rainfall
    user_input['temperature'] = temperature
    user_input['humidity'] = humidity
    user_input['pH'] = ph
    user_input['N'] = n_val
    user_input['P'] = p_val
    user_input['K'] = k_val
    user_input['Soil_Type'] = soil_type
    user_input['Crop'] = crop
    user_input['Irrigation_Method'] = irrigation_method
    
    # Transform
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=training_cols, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    input_pca = pca.transform(input_scaled)
    
    # DBSCAN Mapping
    distances = pairwise_distances(input_pca, X_pca_train)
    nearest_neighbor_idx = np.argmin(distances)
    assigned_cluster = clusters_train[nearest_neighbor_idx]
    
    input_final = np.c_[input_pca, [assigned_cluster]]
    
    # Generate Prediction
    prediction = lr.predict(input_final)
    
    # Render Result Card
    st.markdown(f"""
    <div class="result-card">
        <p>Estimated Crop Yield</p>
        <h2>{prediction[0]:,.2f} <span style='font-size: 1.5rem; font-weight: 500; opacity: 0.9;'>kg/ha</span></h2>
    </div>
    """, unsafe_allow_html=True)
