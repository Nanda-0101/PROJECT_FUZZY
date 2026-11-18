import streamlit as st

from core import FEATURES
from ui import *


setup("Beranda")

st.header("Beranda")
st.caption(
    "Sistem segmentasi siswa menggunakan Principal Component Analysis (PCA) "
    "dan Fuzzy C-Means (FCM) untuk membantu memahami karakteristik serta "
    "kecenderungan kelompok siswa."
)

df = st.session_state.get("df")
if df is not None:
    stat_cards(
        [
            ("Jumlah siswa", len(df), "primary"),
            ("Fitur analisis", len(FEATURES), "success"),
            ("Rata-rata nilai akhir", f"{df['G3'].mean():.2f}", "warning"),
            ("Status model", "Siap" if "res" in st.session_state else "Belum dijalankan",
             "success" if "res" in st.session_state else "secondary"),
        ],
        120,
    )
    st.caption(f"Dataset aktif: {st.session_state.get('data_source', 'Data tersedia')}")
else:
    alert(
        "warning",
        "Belum ada dataset. Mulai dari menu <b>Input Data</b> untuk menggunakan "
        "dataset bawaan atau mengunggah data sendiri.",
    )

st.subheader("Alur penggunaan")
steps = [
    (
        "1",
        "Input & eksplorasi data",
        "Gunakan dataset bawaan atau unggah CSV, lalu tinjau distribusi, korelasi, "
        "outlier, dan kelayakan data untuk PCA.",
    ),
    (
        "2",
        "Proses PCA + FCM",
        "Data numerik distandardisasi, diringkas dengan PCA, lalu dikelompokkan "
        "menggunakan Fuzzy C-Means.",
    ),
    (
        "3",
        "Tinjau & gunakan hasil",
        "Pelajari profil setiap cluster, evaluasi kualitas pengelompokan, dan "
        "prediksi cluster untuk siswa baru.",
    ),
]
bs(
    '<div class="row g-2">'
    + "".join(
        f"""<div class="col-md-4"><div class="card h-100 shadow-sm bg-white text-dark">
<div class="card-body"><span class="step-num bg-primary text-white me-2">{number}</span>
<b>{title}</b><p class="small mt-2 mb-0">{description}</p></div></div></div>"""
        for number, title, description in steps
    )
    + "</div>",
    185,
)

st.subheader("Mulai dari sini")
col_data, col_explore, col_process = st.columns(3)
if col_data.button("Input Data", type="primary", use_container_width=True):
    st.switch_page("pages/Input_Data.py")
if col_explore.button("Eksplorasi Data", use_container_width=True):
    st.switch_page("pages/Eksplorasi_Data.py")
if col_process.button("Proses Pengolahan", use_container_width=True):
    st.switch_page("pages/Proses_Pengolahan.py")

if "res" in st.session_state:
    st.subheader("Lihat hasil")
    col_results, col_prediction = st.columns(2)
    if col_results.button("Buka Hasil Pengolahan", use_container_width=True):
        st.switch_page("pages/Hasil_Pengolahan.py")
    if col_prediction.button("Prediksi Siswa Baru", use_container_width=True):
        st.switch_page("pages/Prediksi_Siswa_Baru.py")
else:
    alert(
        "info",
        "Setelah menjalankan algoritma di halaman <b>Proses Pengolahan</b>, "
        "hasil segmentasi dan prediksi siswa baru dapat dibuka.",
        75,
    )