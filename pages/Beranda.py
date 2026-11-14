import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("🏠 Beranda")

bs("""<div class="hero p-4 p-md-5 mb-4 shadow">
<h1 class="display-6 fw-bold"><i class="bi bi-diagram-3-fill me-2"></i>Sistem Segmentasi Kinerja Siswa</h1>
<p class="lead mb-1">Pengelompokan siswa dengan <b>Fuzzy C-Means (FCM)</b> yang dikombinasikan dengan
<b>Principal Component Analysis (PCA)</b> pada dataset <i>Student Performance</i> (mata pelajaran Matematika).</p>
<p class="mb-0 small opacity-75">Setiap cluster diberi nama yang bermakna (mis. “Berprestasi Tinggi”, “Berprestasi Rendah (Berisiko)”), bukan sekadar Cluster 0 / Cluster 1.</p></div>
<div class="row">""" +
   card("Tujuan Sistem", "Mengelompokkan siswa berdasarkan latar belakang keluarga, kebiasaan, kehadiran, dan nilai agar sekolah dapat mengenali kelompok yang perlu pendampingan khusus.", "primary", "bullseye") +
   card("Mengapa FCM?", "Siswa tidak dipaksa masuk satu kelompok. Setiap siswa punya <b>derajat keanggotaan</b> pada semua cluster sehingga siswa “perbatasan” dapat terdeteksi.", "success", "intersect") +
   card("Mengapa PCA?", "PCA mereduksi 16 fitur menjadi beberapa komponen utama yang tidak berkorelasi, menghilangkan redundansi, mempercepat dan menstabilkan FCM, serta memudahkan visualisasi 2D.", "warning", "arrows-collapse") +
   """</div><h5 class="mt-2">Alur Penggunaan</h5><div class="d-flex flex-wrap gap-2"><span class="badge text-bg-primary p-2">1 Input Data</span><span class="align-self-center">→</span><span class="badge text-bg-primary p-2">2 Eksplorasi Data</span><span class="align-self-center">→</span><span class="badge text-bg-primary p-2">3 Proses Pengolahan</span><span class="align-self-center">→</span><span class="badge text-bg-primary p-2">4 Hasil Pengolahan</span><span class="align-self-center">→</span><span class="badge text-bg-primary p-2">5 Prediksi Siswa Baru</span><span class="align-self-center">→</span><span class="badge text-bg-primary p-2">6 Evaluasi</span><span class="align-self-center">→</span><span class="badge text-bg-secondary p-2">7 Informasi Sistem</span></div>""", 520)
st.markdown("#### Navigasi Cepat")
cols = st.columns(4) + st.columns(4)   # 2 baris, 7 tombol dipakai
for col, (label, path) in zip(cols, PAGE_FILES):
    if col.button(label, use_container_width=True, key=f"nav_{label}"):
        st.switch_page(path)
if st.session_state.get("df") is not None:
    d = st.session_state.df
    stat_cards([("Jumlah siswa", len(d), "primary"), ("Jumlah fitur", len(FEATURES), "success"),
                ("Rata-rata G3", f"{d['G3'].mean():.2f}", "warning"),
                ("Status model", "Siap" if "res" in st.session_state else "Belum dijalankan", "secondary")], 130)
