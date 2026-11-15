# Segmentasi Siswa – Fuzzy C-Means + PCA

Aplikasi Streamlit untuk mensegmentasi siswa berdasarkan 16 fitur numerik (nilai, riwayat kegagalan,
kebiasaan belajar, konsumsi alkohol, dll.) menggunakan **Principal Component Analysis (PCA)** untuk
reduksi dimensi dan **Fuzzy C-Means (FCM)** untuk klasterisasi fuzzy.

Antarmuka dibangun dengan **Streamlit** dan **Bootstrap 5** (dimuat via CDN, sehingga membutuhkan
koneksi internet).

---

## Menjalankan

### 1. Clone & masuk folder
```bash
git clone <repo-url>
cd <nama-folder>
```

### 2. Buat virtual environment
**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`.

---