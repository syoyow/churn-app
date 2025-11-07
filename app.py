import streamlit as st
import joblib
import numpy as np
import pandas as pd

# === LOAD MODEL ===
model = joblib.load('model.pkl')

st.set_page_config(page_title="Prediksi Churn")
st.title(" Prediksi Pelanggan Churn")
st.markdown("Masukkan data pelanggan untuk mengetahui apakah pelanggan akan **churn** atau tidak.")

# === INPUT DATA ===
# usia = st.number_input(" Usia Pelanggan", min_value=18, max_value=100, step=1)
# lama_langganan = st.number_input(" Lama Langganan (bulan)", min_value=1, max_value=120, step=1)
# jumlah_pengaduan = st.number_input("Jumlah Pengaduan", min_value=0, max_value=50, step=1)
input_data = pd.DataFrame([{
    'usia': usia,
    'lama_langganan_bulan': lama_langganan_bulan,
    'jumlah_pengaduan': jumlah_pengaduan,
    'paket_internet_Premium': paket_internet_Premium,
    'paket_internet_Standard': paket_internet_Standard,
    'metode_pembayaran_Transfer': metode_pembayaran_Transfer,
    'metode_pembayaran_e-Wallet': metode_pembayaran_eWallet
}])


# === PREDIKSI ===
if st.button("Prediksi"):
    # Buat DataFrame dengan nama kolom yang sama seperti saat training
    input_data = pd.DataFrame({
        'usia': [usia],
        'lama_langganan': [lama_langganan],
        'jumlah_pengaduan': [jumlah_pengaduan]
    })
    
    prediksi = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Hasil Prediksi:")
    if prediksi == 1:
        st.error("Pelanggan kemungkinan akan **CHURN**.")
    else:
        st.success("✅ Pelanggan kemungkinan **TIDAK CHURN**.")
    
    if probas is not None:
        st.write(f"Probabilitas Churn: **{probas:.2%}**")


