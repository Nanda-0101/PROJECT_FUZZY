import math

import pandas as pd
import streamlit as st

from core import FEATURES, FEATURE_INFO
from ui import bs, setup, stat_cards


# Konstanta tampilan
FEATURE_GROUPS = {
    "Akademik": {"G1", "G2", "G3"},
    "Profil siswa dan keluarga": {"age", "Medu", "Fedu", "famrel"},
    "Belajar dan kehadiran": {"traveltime", "studytime", "failures", "absences"},
    "Sosial dan kesehatan": {"freetime", "goout", "Dalc", "Walc", "health"},
}

K_MODE_LABEL = {"auto": "Otomatis (Silhouette)", "manual": "Manual"}


def _feature_group(feature):
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    return "Lainnya"


def _info_card(label, value, icon="gear"):
    """Kartu kecil untuk satu parameter."""
    return f"""<div class="col-6 col-lg-3 mb-3"><div class="card h-100 shadow-sm bg-white text-dark border-0">
<div class="card-body"><div class="small text-muted"><i class="bi bi-{icon} me-1"></i>{label}</div>
<div class="fw-bold fs-5">{value}</div></div></div></div>"""


# Tampilan awal
setup("Informasi Sistem")

st.header("Informasi Sistem")
st.caption(
    "Ringkasan tujuan, alur komputasi, parameter, fitur, dan batas interpretasi "
    "aplikasi segmentasi siswa dengan FCM dan PCA."
)
bs(
    """<div class="hero p-4 p-md-5 mb-3 shadow-sm">
<div class="row align-items-center"><div class="col-lg-9">
<span class="badge text-bg-light text-primary mb-2">PANDUAN SISTEM</span>
<h2 class="fw-bold mb-2">Dari data siswa menjadi kelompok yang bermakna</h2>
<p class="mb-0 opacity-75">PCA merangkum fitur numerik menjadi komponen utama.
Fuzzy C-Means membentuk kelompok dan memberikan derajat keanggotaan untuk setiap siswa.</p>
</div><div class="col-lg-3 text-lg-end mt-3 mt-lg-0">
<i class="bi bi-cpu-fill" style="font-size:4rem"></i></div></div></div>""",
    185,
)

data = st.session_state.get("df")
result = st.session_state.get("res")
params = result["params"] if result is not None else st.session_state.params

# Ringkasan status
if data is None:
    stat_cards(
        [
            ("Metode", "PCA + FCM", "primary"),
            ("Fitur numerik", len(FEATURES), "success"),
            ("Status data", "Belum tersedia", "warning"),
            ("Status model", "Belum dijalankan", "secondary"),
        ],
        125,
    )
else:
    stat_cards(
        [
            ("Jumlah siswa", len(data), "primary"),
            ("Fitur numerik", len(FEATURES), "success"),
            ("Komponen PCA", f"{result['n_comp']} dari {len(FEATURES)}" if result else "Menunggu proses", "warning"),
            ("Jumlah cluster", result["K"] if result else "Menunggu proses", "secondary"),
        ],
        125,
    )
    st.markdown(
        f"**Sumber data aktif**  \n{st.session_state.get('data_source', 'Dataset aktif')}  \n"
        f"Data yang lolos validasi berisi **{len(data)} siswa** dan **{len(FEATURES)} fitur**."
    )

# Alur pengolahan
st.subheader("Alur pengolahan")
steps = [
    ("1", "Validasi data", "database-check", "primary",
     f"Memastikan seluruh {len(FEATURES)} fitur tersedia, dapat dibaca sebagai angka, dan berada dalam "
     "rentang yang ditentukan. Baris tidak valid dibuang dan dibutuhkan minimal 30 baris valid."),
    ("2", "Standardisasi", "sliders", "info",
     "Setiap fitur diubah ke skala z-score agar perbedaan satuan dan rentang nilai "
     "tidak mendominasi perhitungan jarak."),
    ("3", "Uji kelayakan PCA", "clipboard2-check", "warning",
     "KMO dan Bartlett dihitung dari korelasi fitur sebelum ekstraksi komponen. "
     "Keduanya memberi konteks kelayakan reduksi dimensi."),
    ("4", "Ekstraksi PCA", "pie-chart", "success",
     f"PCA memilih jumlah komponen paling sedikit yang mencapai ambang kumulatif {params['var_thr']:.0%}."),
    ("5", "Pemilihan jumlah cluster", "123", "primary",
     "FCM dijalankan untuk setiap k dari 2 sampai 8 pada ruang PCA. Mode otomatis memilih k "
     "dengan Silhouette tertinggi. Mode manual memakai k dari pengguna."),
    ("6", "Klasterisasi FCM", "diagram-3", "info",
     "FCM menghitung pusat cluster dan derajat keanggotaan secara berulang. Hasil terbaik dari "
     "beberapa inisialisasi dipilih berdasarkan fungsi objektif terendah."),
    ("7", "Penamaan cluster", "tags", "warning",
     "Cluster diurutkan dari rata-rata nilai G1 sampai G3 tertinggi. Nama dan saran tindak lanjut "
     "disusun dari profil fitur asli setiap cluster."),
    ("8", "Evaluasi dan interpretasi", "clipboard2-data", "danger",
     "Silhouette menilai pemisahan label, Fuzzy Partition Coefficient menilai ketegasan keanggotaan, "
     "dan profil fitur membantu menafsirkan kelompok."),
]
step_cards = "".join(
    f"""<div class="col-6 col-lg-3 mb-3"><div class="card h-100 border-{color} shadow-sm bg-white text-dark">
<div class="card-body"><div class="d-flex align-items-center mb-2">
<span class="step-num bg-{color} text-white me-2">{number}</span>
<i class="bi bi-{icon} text-{color} fs-5"></i></div>
<b>{title}</b><p class="small text-muted mt-2 mb-0">{description}</p></div></div></div>"""
    for number, title, icon, color, description in steps
)
bs(f'<div class="row">{step_cards}</div>', 560)

# Rumus
st.subheader("Cara kerja dan rumus utama")
st.markdown(
    "PCA dan FCM bekerja berurutan. Data distandardisasi, direduksi dengan PCA, "
    "lalu dikelompokkan pada ruang komponen yang terpilih."
)
tab_pca, tab_fcm, tab_eval = st.tabs(["Kelayakan PCA", "Fuzzy C-Means", "Ukuran evaluasi"])

with tab_pca:
    st.markdown(
        "**KMO keseluruhan** membandingkan korelasi sederhana dengan korelasi parsial. "
        "Nilai lebih tinggi menunjukkan pola korelasi yang lebih sesuai untuk diringkas."
    )
    st.latex(
        r"\mathrm{KMO} = \frac{\sum_{i \ne j} r_{ij}^{2}}"
        r"{\sum_{i \ne j} r_{ij}^{2} + \sum_{i \ne j} q_{ij}^{2}}"
    )
    st.markdown(
        "Bartlett menguji hipotesis bahwa matriks korelasi adalah matriks identitas. "
        "Implementasi memakai koreksi ukuran sampel dan derajat bebas p kali (p - 1) dibagi 2, "
        "dengan p sebagai jumlah fitur."
    )
    st.latex(r"\chi^2 = -\left(n-1-\frac{2p+5}{6}\right)\ln|R|")

with tab_fcm:
    st.markdown(
        "FCM memberi setiap siswa derajat keanggotaan pada seluruh cluster. "
        "Jumlah keanggotaan satu siswa bernilai satu. Parameter fuzzifier m "
        "mengatur seberapa lembut pembagian tersebut."
    )
    st.latex(r"J_m = \sum_{i=1}^{n}\sum_{j=1}^{k} u_{ij}^{m}\lVert x_i-v_j\rVert^2")
    st.latex(r"u_{ij} = \left[\sum_{l=1}^{k}\left(\frac{\lVert x_i-v_j\rVert}{\lVert x_i-v_l\rVert}\right)^{\frac{2}{m-1}}\right]^{-1}")
    st.markdown(
        f"Model aktif memakai m = {params['m']:.2f}, maksimal {params['max_iter']} iterasi, "
        f"toleransi {params['tol']:g}, dan {params['n_init']} inisialisasi acak."
    )

with tab_eval:
    st.markdown(
        "Silhouette memakai label terbesar dari keanggotaan fuzzy dan jarak siswa pada ruang PCA. "
        "Fuzzy Partition Coefficient mempertahankan informasi keanggotaan lunak."
    )
    st.latex(r"s_i = \frac{b_i-a_i}{\max(a_i,b_i)}")
    st.latex(r"\mathrm{FPC} = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k}u_{ij}^{2}")
    if result:
        st.markdown(f"Untuk k = {result['K']}, nilai FPC berada antara {1 / result['K']:.3f} dan 1.")
    else:
        st.markdown("Nilai FPC berada antara 1/k dan 1, dengan k sebagai jumlah cluster.")
    st.latex(r"\mathrm{ExplainedVarianceRatio}_j = \frac{\lambda_j}{\sum_{t=1}^{p}\lambda_t}")

# Fitur
st.subheader("Fitur yang digunakan")
feature_rows = []
for feature in FEATURES:
    label, minimum, maximum, _, description = FEATURE_INFO[feature]
    row = {
        "Kelompok": _feature_group(feature),
        "Kode": feature,
        "Nama fitur": label,
        "Rentang valid": f"{minimum} sampai {maximum}",
        "Keterangan": description,
    }
    if data is not None:
        row["Rata-rata data aktif"] = round(float(data[feature].mean()), 2)
    feature_rows.append(row)
st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
st.info(
    f"Fitur kategorikal seperti sekolah, jenis kelamin, dan pekerjaan orang tua tidak dipakai. "
    f"Perhitungan PCA dan FCM menggunakan {len(FEATURES)} fitur numerik di atas."
)

# Parameter
st.subheader("Parameter dan hasil sesi")
if result is None:
    st.info(
        "Model belum dijalankan. Nilai parameter di bawah adalah konfigurasi yang "
        "akan dipakai saat proses dijalankan."
    )
    k_text = "Dipilih otomatis" if params["k_mode"] == "auto" else f"k = {params.get('k_manual', '-')}"
else:
    k_text = f"k = {result['K']}"

parameter_values = [
    ("Fuzzifier m", f"{params['m']:.2f}", "sliders2"),
    ("Ambang varians PCA", f"{params['var_thr']:.0%}", "pie-chart"),
    ("Mode jumlah cluster", K_MODE_LABEL.get(params["k_mode"], params["k_mode"]), "toggles"),
    ("Jumlah cluster", k_text, "123"),
    ("Maksimal iterasi", str(params["max_iter"]), "arrow-repeat"),
    ("Toleransi konvergensi", f"{params['tol']:g}", "bullseye"),
    ("Inisialisasi FCM", str(params["n_init"]), "shuffle"),
    ("Random seed", str(params["seed"]), "dice-5"),
]
if result is not None:
    parameter_values += [
        ("Iterasi konvergen", str(result["n_iter"]), "check2-all"),
        ("Komponen PCA", f"{result['n_comp']} dari {len(FEATURES)}", "diagram-2"),
    ]
parameter_cards = "".join(_info_card(label, value, icon) for label, value, icon in parameter_values)
bs(f'<div class="row">{parameter_cards}</div>', 20 + 125 * math.ceil(len(parameter_values) / 4))

# Batasan
st.subheader("Batasan interpretasi")
bs(
    """<div class="row g-3">
<div class="col-md-4"><div class="card h-100 border-warning shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-warning"><i class="bi bi-exclamation-circle me-1"></i>Jarak dan skala</h6>
<p class="small mb-0">Model berbasis jarak Euclidean. Standardisasi mengatasi perbedaan skala, tetapi tidak menghilangkan pengaruh pencilan.</p></div></div></div>
<div class="col-md-4"><div class="card h-100 border-info shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-info"><i class="bi bi-diagram-2 me-1"></i>Label cluster</h6>
<p class="small mb-0">Silhouette memakai label tegas hasil keanggotaan terbesar, sedangkan FPC memakai seluruh derajat keanggotaan fuzzy.</p></div></div></div>
<div class="col-md-4"><div class="card h-100 border-danger shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-danger"><i class="bi bi-shield-exclamation me-1"></i>Penggunaan hasil</h6>
<p class="small mb-0">Cluster menggambarkan kemiripan pada data yang tersedia. Hasil bukan bukti sebab-akibat dan perlu ditafsirkan bersama konteks pendidikan.</p></div></div></div>
</div>""",
    195,
)

# Navigasi lanjutan
st.subheader("Langkah berikutnya")
nav_a, nav_b = st.columns(2)
if nav_a.button("Buka Proses Pengolahan", use_container_width=True):
    st.switch_page("pages/Proses_Pengolahan.py")
if nav_b.button("Buka Evaluasi", use_container_width=True, disabled=result is None):
    st.switch_page("pages/Evaluasi.py")