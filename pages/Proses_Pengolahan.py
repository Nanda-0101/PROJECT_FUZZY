import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Proses Pengolahan")

st.header("Proses Pengolahan")
if not need_data():
    bs("""<div class="row g-2">""" + "".join(
        f"""<div class="col-md-4"><div class="card h-100 shadow-sm bg-white text-dark"><div class="card-body">
<span class="step-num bg-primary text-white me-2">{n}</span><b>{t}</b><p class="small mt-2 mb-0">{d}</p></div></div></div>"""
        for n, t, d in [
            (1, "Seleksi & Standardisasi", "16 fitur numerik dipilih lalu di-standardisasi (z-score) agar skala yang berbeda (mis. absences 0–93 vs famrel 1–5) tidak mendominasi jarak."),
            (2, "PCA", "Mencari komponen utama; jumlah komponen = minimum yang mencapai ambang kumulatif explained variance."),
            (3, "Pemilihan jumlah cluster (k)", "FCM dijalankan untuk tiap k pada ruang PCA; k terbaik = Silhouette tertinggi (atau ditentukan manual)."),
            (4, "FCM", "Iterasi update pusat cluster & derajat keanggotaan hingga perubahan maksimum < toleransi."),
            (5, "Penamaan cluster", "Cluster diurutkan berdasarkan rata-rata nilai, lalu dinamai dari profil fitur aslinya."),
            (6, "Evaluasi", "KMO dan Bartlett (sebelum ekstraksi), explained variance (sesudah ekstraksi), Silhouette score, dan Fuzzy Partition Coefficient.")]) +
       "</div>", 330)

    st.subheader("Parameter")
    p = st.session_state.params
    c1, c2, c3, c4 = st.columns(4)
    m = c1.number_input("Fuzzifier (m)", 1.05, 3.0, float(p["m"]), 0.05,
                        help="m>1. Nilai 2.0 pada data ini cenderung menghasilkan solusi degenerat (semua keanggotaan mendekati 1/c).")
    max_iter = c2.number_input("Maks. iterasi", 50, 2000, int(p["max_iter"]), 50)
    tol = c3.selectbox("Toleransi konvergensi", [1e-3, 1e-4, 1e-5, 1e-6], index=[1e-3, 1e-4, 1e-5, 1e-6].index(p["tol"]) if p["tol"] in [1e-3, 1e-4, 1e-5, 1e-6] else 2, format_func=lambda v: f"{v:g}")
    n_init = c4.number_input("Jumlah inisialisasi acak", 1, 20, int(p["n_init"]))
    c1, c2, c3, c4 = st.columns(4)
    var_thr = c1.slider("Ambang explained variance PCA", 0.5, 0.99, float(p["var_thr"]), 0.01)
    k_mode = c2.radio("Penentuan jumlah cluster", ["auto", "manual"], index=0 if p["k_mode"] == "auto" else 1,
                      format_func=lambda v: "Otomatis (Silhouette)" if v == "auto" else "Manual", horizontal=True)
    k_manual = c3.number_input("k manual", 2, 8, int(p["k_manual"]), disabled=(k_mode == "auto"))
    seed = c4.number_input("Random seed", 0, 99999, int(p["seed"]))
    newp = dict(m=float(m), max_iter=int(max_iter), tol=float(tol), n_init=int(n_init), var_thr=float(var_thr),
                k_mode=k_mode, k_manual=int(k_manual), k_max=8, seed=int(seed))
    st.session_state.params = newp

    if st.button("Jalankan Algoritma", type="primary"):
        with st.spinner("Menjalankan PCA dan Fuzzy C-Means ..."):
            st.session_state.res = cached_pipeline(st.session_state.df, newp)
        st.success("Selesai. Lihat hasil antara di bawah dan menu Hasil Pengolahan.")

    if "res" in st.session_state:
        r = st.session_state.res
        if r["degenerate"]:
            alert("danger", "Solusi degenerat terdeteksi (PC = 1/c). Turunkan nilai fuzzifier m (mis. 1.5).", 70)
        stat_cards([("Komponen PCA", f"{r['n_comp']} / {len(FEATURES)}", "primary"),
                    ("Varians dipertahankan", f"{r['cum'][r['n_comp']-1]:.1%}", "success"),
                    ("k dipakai", r["K"], "warning"), ("Iterasi konvergen", r["n_iter"], "secondary")], 120)
        st.subheader("Hasil antara")
        with st.expander("Tahap 1 - Standardisasi", expanded=False):
            a = st.session_state.df[FEATURES].head(8)
            b = pd.DataFrame(r["X"][:8], columns=FEATURES)
            st.write("Data asli (8 baris pertama)"); st.dataframe(a, use_container_width=True)
            st.write("Setelah z-score (rata-rata 0, simpangan baku 1)"); st.dataframe(b.round(3), use_container_width=True)
        with st.expander("Tahap 2 - PCA: explained variance", expanded=True):
            evr_t = pd.DataFrame({"Komponen": [f"PC{i+1}" for i in range(len(r["evr"]))],
                                  "Explained variance": r["evr"], "Kumulatif": r["cum"]}).set_index("Komponen")
            c1, c2 = st.columns([1, 1.4])
            c1.dataframe(evr_t.round(4), use_container_width=True, height=400)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            x = np.arange(1, len(r["evr"]) + 1)
            ax[0].bar(x, r["evr"], color="#4C72B0"); ax[0].set(title="Scree plot", xlabel="Komponen", ylabel="Rasio varians")
            ax[1].plot(x, r["cum"], "o-", color="#DD8452")
            ax[1].axhline(newp["var_thr"], ls="--", color="gray"); ax[1].axvline(r["n_comp"], ls="--", color="red")
            ax[1].set(title="Kumulatif", xlabel="Jumlah komponen")
            plt.tight_layout(); c2.pyplot(fig); plt.close(fig)
            st.caption(f"Dipilih {r['n_comp']} komponen (kumulatif >= {newp['var_thr']:.0%}).")
        with st.expander("Tahap 3 - Pemilihan jumlah cluster (k)", expanded=True):
            kt = r["k_table"].drop(columns="Degenerat")
            st.dataframe(kt.round(4), use_container_width=True)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
            ax[0].plot(kt.index, kt["Silhouette"], "o-"); ax[0].axvline(r["K"], color="red", ls="--")
            ax[0].set(title="Silhouette vs k (naik lebih baik)", xlabel="k")
            ax[1].plot(kt.index, kt["Partition Coefficient"], "o-", label="PC")
            ax[1].plot(kt.index, 1 / kt.index.values, ":", color="gray", label="batas 1/k")
            ax[1].set(title="Partition Coefficient vs k", xlabel="k"); ax[1].legend()
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption(f"k otomatis (Silhouette tertinggi) = {r['k_auto']}; k yang dipakai = {r['K']}.")
        with st.expander("Tahap 4 - Iterasi FCM & pusat cluster", expanded=False):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(range(1, len(r["hist"]) + 1), r["hist"]); ax.set(title="Konvergensi fungsi objektif J", xlabel="Iterasi", ylabel="J")
            st.pyplot(fig); plt.close(fig)
            cen = r["scaler"].inverse_transform(r["pca"].inverse_transform(r["centers"]))
            cdf = pd.DataFrame(cen, columns=FEATURES, index=r["names"]).T
            st.write("Pusat cluster (dikembalikan ke skala fitur asli)")
            st.dataframe(cdf.round(2), use_container_width=True)
        if st.button("Lihat Hasil Pengolahan"):
            st.switch_page("pages/Hasil_Pengolahan.py")