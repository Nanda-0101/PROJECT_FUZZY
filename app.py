"""Entry point: konfigurasi aplikasi dan navigasi 6 halaman.
Jalankan dengan:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Segmentasi Siswa FCM + PCA", page_icon="🎓", layout="wide")

pages = [
    st.Page("pages/1_Beranda.py", title="Beranda", icon="🏠", default=True),
    st.Page("pages/2_Input_Data.py", title="Input Data", icon="📥"),
    st.Page("pages/3_Eksplorasi_Data.py", title="Eksplorasi Data", icon="📈"),
    st.Page("pages/4_Proses_Pengolahan.py", title="Proses Pengolahan", icon="⚙️"),
    st.Page("pages/5_Hasil_Pengolahan.py", title="Hasil Pengolahan", icon="📊"),
    st.Page("pages/6_Prediksi_Siswa_Baru.py", title="Prediksi Siswa Baru", icon="🔮"),
    st.Page("pages/7_Evaluasi.py", title="Evaluasi", icon="✅"),
    st.Page("pages/8_Informasi_Sistem.py", title="Informasi Sistem", icon="ℹ️"),
]
st.navigation(pages).run()
