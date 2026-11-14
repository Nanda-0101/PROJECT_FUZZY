# Segmentasi Siswa – Fuzzy C-Means + PCA (Streamlit + Bootstrap 5)

## Menjalankan
    pip install -r requirements.txt
    streamlit run app.py

## Struktur folder
    fcm_pca_app/
    ├── app.py                          # Router navigasi (st.navigation)
    ├── pages/
    │   ├── 1_Beranda.py                # Halaman Utama
    │   ├── 2_Input_Data.py             # Input Data
    │   ├── 3_Eksplorasi_Data.py        # Eksplorasi Data (baru)
    │   ├── 4_Proses_Pengolahan.py      # Proses Pengolahan
    │   ├── 5_Hasil_Pengolahan.py       # Hasil Pengolahan
    │   ├── 6_Prediksi_Siswa_Baru.py    # Prediksi Siswa Baru (baru)
    │   ├── 7_Evaluasi.py               # Evaluasi
    │   └── 8_Informasi_Sistem.py       # Informasi Sistem
    ├── ui.py  core.py  student-mat.csv  requirements.txt

Catatan: Bootstrap 5 dimuat dari CDN (butuh internet). Data & hasil disimpan di session_state
sehingga terbawa antarhalaman; jalankan halaman Proses Pengolahan sebelum membuka Hasil/Evaluasi.
