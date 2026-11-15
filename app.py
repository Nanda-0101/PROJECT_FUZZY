import streamlit as st

st.set_page_config(
    page_title="Segmentasi Siswa FCM + PCA",
    page_icon=None,
    layout="wide",
)

pages = [
    st.Page("pages/Beranda.py", title="Beranda", default=True),
    st.Page("pages/Input_Data.py", title="Input Data"),
    st.Page("pages/Eksplorasi_Data.py", title="Eksplorasi Data"),
    st.Page("pages/Proses_Pengolahan.py", title="Proses Pengolahan"),
    st.Page("pages/Hasil_Pengolahan.py", title="Hasil Pengolahan"),
    st.Page("pages/Prediksi_Siswa_Baru.py", title="Prediksi Siswa Baru"),
    st.Page("pages/Evaluasi.py", title="Evaluasi"),
    st.Page("pages/Informasi_Sistem.py", title="Informasi Sistem"),
]

st.navigation(pages).run()