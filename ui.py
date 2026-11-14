import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "student-mat.csv")
CMAP = plt.get_cmap("tab10")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# (judul halaman, path file) - dipakai navigasi cepat
PAGE_FILES = [
    ("Input Data", "pages/2_Input_Data.py"),
    ("Eksplorasi Data", "pages/3_Eksplorasi_Data.py"),
    ("Proses Pengolahan", "pages/4_Proses_Pengolahan.py"),
    ("Hasil Pengolahan", "pages/5_Hasil_Pengolahan.py"),
    ("Prediksi Siswa Baru", "pages/6_Prediksi_Siswa_Baru.py"),
    ("Evaluasi", "pages/7_Evaluasi.py"),
    ("Informasi Sistem", "pages/8_Informasi_Sistem.py"),
]

# Bootstrap helpers
BOOT_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
BOOT_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
BOOT_ICON = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"


def bs(body, height=200):
    """Render potongan HTML yang di-style Bootstrap 5 (CSS dimuat via CDN)."""
    components.html(
        f"""<!doctype html><html data-bs-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="{BOOT_CSS}" rel="stylesheet"><link href="{BOOT_ICON}" rel="stylesheet">
<style>body{{background:transparent;font-size:.95rem}}
.hero{{background:linear-gradient(135deg,#0d6efd,#6610f2);color:#fff;border-radius:1rem}}
.step-num{{width:2.2rem;height:2.2rem;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}</style>
</head><body><div class="container-fluid px-0">{body}</div>
<script src="{BOOT_JS}"></script></body></html>""",
        height=height, scrolling=True)


def alert(kind, text, height=70):
    icon = {"success": "check-circle", "warning": "exclamation-triangle",
            "danger": "x-octagon", "info": "info-circle", "error": "x-octagon"}.get(kind, "info-circle")
    kind = "danger" if kind == "error" else kind
    bs(f'<div class="alert alert-{kind} mb-0"><i class="bi bi-{icon} me-2"></i>{text}</div>', height)


def alerts(msgs):
    for k, t in msgs:
        alert(k, t, 62 if len(t) < 110 else 90)


def bs_table(df, height=300, float_fmt="{:.4f}", index=True):
    html = df.to_html(classes="table table-sm table-striped table-hover align-middle mb-0",
                      border=0, index=index, float_format=lambda v: float_fmt.format(v), escape=False)
    bs(f'<div class="card bg-white text-dark shadow-sm"><div class="table-responsive">{html}</div></div>', height)


def navbar(active):
    bs(f"""<nav class="navbar navbar-dark bg-dark rounded-3 px-3">
<span class="navbar-brand mb-0 h1"><i class="bi bi-mortarboard-fill me-2"></i>Segmentasi Siswa · FCM + PCA</span>
<span class="badge text-bg-primary fs-6">{active}</span></nav>""", 64)


def card(title, body, color="primary", icon="info-circle", col="col-md-4"):
    return f"""<div class="{col} mb-3"><div class="card h-100 border-{color} shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="card-title text-{color}"><i class="bi bi-{icon} me-1"></i>{title}</h6>
<p class="card-text small mb-0">{body}</p></div></div></div>"""


def stat_cards(items, height=120):
    cols = "".join(f"""<div class="col-6 col-md-3 mb-2"><div class="card text-center shadow-sm bg-white text-dark h-100">
<div class="card-body py-3"><div class="text-muted small">{lbl}</div><div class="fs-4 fw-bold text-{clr}">{val}</div></div></div></div>"""
                   for lbl, val, clr in items)
    bs(f'<div class="row">{cols}</div>', height)



# State
def set_data(df, source):
    st.session_state.df = df
    st.session_state.data_source = source
    st.session_state.pop("res", None)       # hasil lama tidak berlaku lagi


def load_default():
    if os.path.exists(DEFAULT_CSV):
        clean, msgs, info = core.validate_dataset(core.read_dataset(DEFAULT_CSV))
        if clean is not None:
            set_data(clean, "student-mat.csv (bawaan)")


@st.cache_data(show_spinner=False)
def cached_pipeline(df, p):
    return core.run_pipeline(df, m=p["m"], max_iter=p["max_iter"], tol=p["tol"], n_init=p["n_init"],
                             var_thr=p["var_thr"], k_max=p["k_max"], k_mode=p["k_mode"],
                             k_manual=p["k_manual"], seed=p["seed"])


def need_data():
    if st.session_state.get("df") is None:
        alert("warning", "Belum ada dataset. Silakan buka menu <b>Input Data</b> terlebih dahulu.")
        return True
    return False


def need_result():
    if need_data():
        return True
    if "res" not in st.session_state:
        alert("warning", "Algoritma belum dijalankan. Buka menu <b>Proses Pengolahan</b> lalu tekan tombol <b>Jalankan Algoritma</b>.")
        return True
    return False



def init_state():
    if "df" not in st.session_state:
        load_default()
    if "params" not in st.session_state:
        st.session_state.params = dict(m=1.5, max_iter=500, tol=1e-5, n_init=5, var_thr=0.80,
                                       k_mode="auto", k_manual=3, k_max=8, seed=42)


def setup(title):
    """Panggil di awal setiap halaman: state, status sidebar, navbar."""
    init_state()
    with st.sidebar:
        st.divider()
        if st.session_state.get("df") is not None:
            st.success(f"Data: {st.session_state.data_source}\n\n{len(st.session_state.df)} siswa")
        else:
            st.warning("Belum ada data")
        if "res" in st.session_state:
            st.success("Model siap ✔")
        else:
            st.info("Model belum dijalankan")
    navbar(title)


# Prediksi siswa baru
def render_prediction(res, values):
    u, zv = core.predict_one(res, values)
    j = int(u.argmax())
    info = res["cluster_info"][j]
    rows = "".join(f"""<div class="mb-2"><div class="d-flex justify-content-between small">
<span>{res['names'][i]}</span><b>{u[i]:.1%}</b></div><div class="progress" style="height:14px">
<div class="progress-bar {'bg-success' if i == j else 'bg-secondary'}" style="width:{u[i]*100:.1f}%"></div></div></div>"""
                   for i in range(len(u)))
    bs(f"""<div class="card shadow-sm border-success bg-white text-dark"><div class="card-body">
<h5 class="text-success"><i class="bi bi-person-check me-1"></i>Siswa ini masuk: <b>{info['nama']}</b></h5>
<p class="small">{info['deskripsi']}</p>
<p class="small"><b>Saran:</b> {info['saran']}</p>
<hr><div class="small text-muted mb-2">Derajat keanggotaan fuzzy (total = 100%)</div>{rows}
{'<div class="alert alert-warning py-1 small mt-2 mb-0">Keanggotaan hampir seimbang: siswa berada di perbatasan antar kelompok.</div>' if u.max() < 0.6 else ''}
</div></div>""", 240 + 45 * len(u))
    return u, zv


