"""Entry point: konfigurasi aplikasi dan navigasi 6 halaman.
Jalankan dengan:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Segmentasi Siswa FCM + PCA", page_icon="🎓", layout="wide")

pages = [
 
]
st.navigation(pages).run()
