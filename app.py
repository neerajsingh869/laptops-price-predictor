import streamlit as st
import pickle
import numpy as np

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
Company = st.selectbox('Brand', df['Company'].value_counts().reset_index()['index'])

# type of laptop
TypeName = st.selectbox('Type', df['TypeName'].value_counts().reset_index()['index'])

# ram
ram_values = [2, 4, 6, 8, 12, 16, 24, 32, 64]
ram_default_ix = ram_values.index(8)
Ram = st.selectbox('RAM (in GB)', ram_values, index=ram_default_ix)

# weight
Weight = st.number_input('Weight of the Laptop (in Kg)', min_value=1.00, max_value=5.00)

# touchscreen or not
Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# ips
Ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, step=0.1, format="%.1f")

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
Cpu_brand = st.selectbox('CPU', df['Cpu_brand'].value_counts().reset_index()['index'])

hdd_values = [0,128,256,512,1024,2048]
hdd_default_ix = hdd_values.index(512)
HDD = st.selectbox('HDD (in GB)', hdd_values, index=hdd_default_ix)

ssd_values = [0,8,128,256,512,1024]
ssd_default_ix = ssd_values.index(512)
SSD = st.selectbox('SSD (in GB)', ssd_values, index=ssd_default_ix)

Gpu_brand = st.selectbox('GPU', df['Gpu_brand'].value_counts().reset_index()['index'])

# Only Apple laptops have Mac OS
if Company == 'Apple':
    os_values = ['Mac']
else:
    os_values = ['Windows', 'Others/No OS/Linux']
os = st.selectbox('OS', os_values)

if st.button('Predict Price'):
    # touchscreen
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    # ips
    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0
    # ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = np.round((((X_res**2) + (Y_res**2))**0.5/screen_size), 2)
    # query
    query = np.array([Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu_brand, HDD, SSD, Gpu_brand, os])
    # reshaping query
    query = query.reshape(1, 12)
    st.header("The predicted price for the given configuration is " + "₹" + str(np.round((np.exp(pipe.predict(query))[0]), 2)))
