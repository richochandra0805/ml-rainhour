# Rainhour Predictor App 🌧️

Aplikasi prediksi jam hujan harian berdasarkan proyeksi iklim (SSP126/245/585) menggunakan machine learning.

## Fitur
- Prediksi eksak Rainhour (jam hujan)
- Probabilitas hujan per jam (0–23)
- Menggunakan model ML hasil auto-selection (regresi + multioutput klasifikasi)

## Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
