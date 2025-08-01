import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =============================
# 🧠 LOAD MODEL
# =============================
reg_model = joblib.load("model_rainhour_best_rf.pkl")  # Ganti sesuai model kamu
clf_model = joblib.load("model_prob_rainhour_best_rf.pkl")

# =============================
# 📂 LOAD DATA PROYEKSI
# =============================
@st.cache_data
def load_data():
    data = {
        "SSP126": pd.read_excel("data/timeseries_pr_corrected_ssp126.xlsx"),
        "SSP245": pd.read_excel("data/timeseries_pr_corrected_ssp245.xlsx"),
        "SSP585": pd.read_excel("data/timeseries_pr_corrected_ssp585.xlsx")
    }
    for k in data:
        df = data[k]
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df['dayofyear'] = df['Tanggal'].dt.dayofyear
        df['month'] = df['Tanggal'].dt.month
        data[k] = df
    return data

data_all = load_data()

# =============================
# 🌐 UI: STREAMLIT
# =============================
st.title("Prediksi Rainhour & Probabilitas Jam Hujan")
st.markdown("Model ML prediktif berdasarkan skenario perubahan iklim (SSP).")

# Input
col1, col2 = st.columns(2)
with col1:
    tanggal = st.date_input("Pilih Tanggal", datetime(2030, 1, 1))
with col2:
    skenario = st.selectbox("Pilih Skenario Iklim", ["SSP126", "SSP245", "SSP585"])

# =============================
# 🔍 FILTER DATA
# =============================
df_selected = data_all[skenario]
df_today = df_selected[df_selected['Tanggal'] == pd.to_datetime(tanggal)]

if df_today.empty:
    st.warning("Data untuk tanggal ini tidak tersedia.")
else:
    input_features = df_today[['Rainfall', 'dayofyear', 'month']]
    
    # Placeholder untuk EWH & Indeks-Nino jika tidak ada → pakai rata-rata
    if 'EWH' not in df_today.columns:
        input_features['EWH'] = 0.5  # Default / bisa disesuaikan
    else:
        input_features['EWH'] = df_today['EWH']
    
    if 'Indeks-Nino' not in df_today.columns:
        input_features['Indeks-Nino'] = 0.5
    else:
        input_features['Indeks-Nino'] = df_today['Indeks-Nino']

    # =============================
    # 🔮 PREDIKSI
    # =============================
    rainhour_pred = reg_model.predict(input_features)[0]
    prob_pred = clf_model.predict_proba(input_features)

    # Bentuk prediksi probabilitas menjadi array 24 jam
    prob_vector = np.array([p[:, 1] for p in prob_pred]).flatten()

    # =============================
    # 📊 OUTPUT
    # =============================
    st.subheader("📈 Prediksi Rainhour")
    st.metric("Durasi Hujan", f"{rainhour_pred:.2f} jam")

    st.subheader("🕐 Probabilitas Jam Hujan (0–23)")
    st.bar_chart(pd.DataFrame({
        'Probabilitas': prob_vector
    }, index=[f"{i}:00" for i in range(24)]))
