import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Input Data")

st.header("Input Data")
t1, t2, t3 = st.tabs(["Dataset (Unggah / Bawaan)", "Input Manual Siswa", "Penjelasan Variabel"])

with t1:
    st.write("Gunakan dataset bawaan atau unggah file CSV sendiri (separator `;` atau `,` terdeteksi otomatis). "
             "Kolom wajib: " + ", ".join(f"`{c}`" for c in FEATURES) + ".")
    c1, c2 = st.columns([1, 2])
    with c1:
        if os.path.exists(DEFAULT_CSV) and st.button("Gunakan dataset bawaan (student-mat.csv)", use_container_width=True):
            load_default(); st.rerun()
    with c2:
        up = st.file_uploader("Unggah dataset (.csv)", type=["csv"])
    if up is not None:
        try:
            raw = core.read_dataset(up)
            clean, msgs, info = core.validate_dataset(raw)
            alerts(msgs)
            if clean is not None:
                st.caption(f"{info['baris_awal']} baris terbaca, {info['baris_valid']} baris valid.")
                if st.button("Pakai dataset yang diunggah", type="primary"):
                    set_data(clean, f"{up.name} (unggahan)"); st.rerun()
        except Exception as e:
            alert("danger", f"File tidak dapat dibaca: {e}")
    if st.session_state.get("df") is not None:
        d = st.session_state.df
        st.subheader("Data aktif")
        stat_cards([("Sumber", st.session_state.data_source, "primary"), ("Baris", len(d), "success"),
                    ("Fitur dipakai", len(FEATURES), "warning"), ("Missing", int(d[FEATURES].isna().sum().sum()), "secondary")], 120)
        st.dataframe(d[FEATURES].head(50), use_container_width=True)
        st.markdown("**Statistik deskriptif**")
        st.dataframe(d[FEATURES].describe().T.round(3), use_container_width=True)

with t2:
    st.write("Isi data satu siswa. Sistem memvalidasi rentang nilai, lalu (jika model sudah dijalankan) "
             "menentukan cluster dan derajat keanggotaannya.")
    with st.form("manual"):
        vals = {}
        cols = st.columns(4)
        for i, c in enumerate(FEATURES):
            lbl, lo, hi, de, desc = FEATURE_INFO[c]
            with cols[i % 4]:
                vals[c] = st.number_input(f"{lbl} ({c})", min_value=float(lo - 100), max_value=float(hi + 1000),
                                          value=float(de), step=1.0, help=f"{desc} Rentang valid {lo}–{hi}.")
        ok = st.form_submit_button("Validasi & Tentukan Cluster", type="primary")
    if ok:
        errs = core.validate_single(vals)
        if errs:
            for e in errs:
                alert("danger", e, 58)
        else:
            st.session_state.manual = vals
            alert("success", "Data masukan valid.", 58)
    if "manual" in st.session_state:
        if "res" in st.session_state:
            render_prediction(st.session_state.res, st.session_state.manual)
        else:
            alert("info", "Data tersimpan. Jalankan algoritma di menu <b>Proses Pengolahan</b> untuk melihat cluster siswa ini.")

with t3:
    rows = [{"Variabel": c, "Nama": v[0], "Rentang": f"{v[1]}–{v[2]}", "Penjelasan": v[4]}
            for c, v in FEATURE_INFO.items()]
    bs_table(pd.DataFrame(rows), 640, index=False)
    alert("info", "Atribut kategorikal (school, sex, Mjob, dll.) tidak dipakai karena FCM dan PCA berbasis jarak Euclidean pada data numerik.", 75)