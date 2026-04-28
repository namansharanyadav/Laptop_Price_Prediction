import streamlit as st
import pickle
import numpy as np
import pandas as pd
import locale

# Page config
st.set_page_config(page_title="Laptop Predictor", page_icon="💻", layout="centered")

# Custom CSS for light theme & styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #1b5e20;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Title
st.markdown("<h1>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 🔍 Enter Laptop Specifications")

# Inputs
company = st.selectbox('🏢 Brand', df['Company'].unique())
type = st.selectbox('💻 Type', df['TypeName'].unique())
ram = st.selectbox('⚡ RAM (GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('⚖️ Weight (kg)')
touchscreen = st.selectbox('📱 Touchscreen', ['No','Yes'])
ips = st.selectbox('🖥️ IPS Display', ['No','Yes'])
screen_size = st.slider('📏 Screen Size (inches)', 10.0,18.0, 13.0)

resolution = st.selectbox('🧾 Resolution',
['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('🧠 CPU', df['Cpu brand'].unique())
hdd = st.selectbox('💾 HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('🚀 SSD (GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())
os = st.selectbox('🪟 OS', df['os'].unique())

# Prediction
if st.button('💰 Predict Price'):
    
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'IPS': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    locale.setlocale(locale.LC_ALL, 'en_IN')
    price = int(np.exp(pipe.predict(query)[0]))

    formatted_price = locale.format_string("%d", price, grouping=True)

    # Attractive Result Box
    st.markdown(f"""
        <div class="result-box">
            💸 Estimated Price: ₹ {formatted_price}
        </div>
    """, unsafe_allow_html=True)