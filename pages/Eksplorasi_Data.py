import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Eksplorasi Data")

st.header("Eksplorasi Data")
st.caption("Memahami karakteristik data sebelum diproses: sebaran, korelasi, outlier, dan kelayakan data untuk PCA.")

if not need_data():
    d = st.session_state.df
    X = d[FEATURES].astype(float)
    stat_cards([("Jumlah siswa", len(d), "primary"), ("Jumlah fitur", len(FEATURES), "success"),
                ("Rata-rata G3", f"{d['G3'].mean():.2f}", "warning"),
                ("Siswa G3 < 10", f"{(d['G3'] < 10).mean():.1%}", "danger")], 120)

    t1, t2, t3, t4 = st.tabs(["Distribusi", "Korelasi", "Outlier", "Kelayakan PCA"])

    with t1:
        fig, axs = plt.subplots(4, 4, figsize=(13, 9))
        for ax, c in zip(axs.ravel(), FEATURES):
            ax.hist(X[c], bins=min(20, max(5, X[c].nunique())), color="#4C72B0", edgecolor="white")
            ax.set_title(f"{c}", fontsize=9); ax.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        sk = pd.DataFrame({"Nama": [FEATURE_INFO[c][0] for c in FEATURES], "Rata-rata": X.mean(), "Std": X.std(),
                           "Min": X.min(), "Maks": X.max(), "Skewness": X.skew()})
        st.dataframe(sk.round(3), use_container_width=True)
        st.caption("Skewness |nilai| > 1 menandakan sebaran sangat miring (mis. absences dan failures), sehingga standardisasi diperlukan agar fitur sebanding.")

    with t2:
        C = X.corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
        ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
        plt.colorbar(im, label="Korelasi Pearson"); ax.set_title("Matriks korelasi fitur")
        st.pyplot(fig); plt.close(fig)
        pairs = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().reset_index()
        pairs.columns = ["Fitur A", "Fitur B", "Korelasi"]
        pairs["|r|"] = pairs["Korelasi"].abs()
        st.write("Pasangan fitur dengan korelasi terkuat")
        st.dataframe(pairs.sort_values("|r|", ascending=False).head(10).drop(columns="|r|").round(3),
                     use_container_width=True, hide_index=True)

    with t3:
        q1, q3 = X.quantile(.25), X.quantile(.75)
        iqr = q3 - q1
        out = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr))
        ot = pd.DataFrame({"Jumlah outlier": out.sum(), "Persentase": out.mean() * 100}).sort_values("Jumlah outlier", ascending=False)
        c1, c2 = st.columns([1, 1.4])
        c1.dataframe(ot.round(2), use_container_width=True, height=470)
        Z = (X - X.mean()) / X.std()
        fig, ax = plt.subplots(figsize=(7, 5.4))
        ax.boxplot([Z[c] for c in FEATURES], vert=False, flierprops=dict(markersize=3))
        ax.set_yticks(range(1, len(FEATURES) + 1)); ax.set_yticklabels(FEATURES)
        ax.set_xlabel("z-score"); ax.set_title("Boxplot fitur terstandardisasi")
        c2.pyplot(fig); plt.close(fig)
        alert("info", "Outlier <b>tidak dibuang</b>: nilai ekstrem seperti absensi tinggi atau nilai 0 adalah data siswa yang sah dan justru informatif untuk mendeteksi siswa berisiko.", 75)

    with t4:
        s = core.pca_suitability(d)
        kmo_cat = ("sangat baik" if s["kmo"] >= .8 else "baik" if s["kmo"] >= .7 else "cukup" if s["kmo"] >= .6
                   else "kurang" if s["kmo"] >= .5 else "buruk")
        stat_cards([("KMO", f"{s['kmo']:.3f}", "primary"), ("Kategori KMO", kmo_cat, "success"),
                    ("Bartlett chi2", f"{s['chi2']:.1f}", "warning"),
                    ("p-value Bartlett", "< 0.001" if s["p"] < .001 else f"{s['p']:.4f}", "secondary")], 120)
        mean_abs = np.abs(X.corr().values[np.triu_indices(len(FEATURES), 1)]).mean()
        lines = [f"Uji Bartlett {'signifikan' if s['p'] < .05 else 'tidak signifikan'} (p {'< 0,05' if s['p'] < .05 else '>= 0,05'}): "
                 f"{'ada korelasi antar fitur sehingga reduksi dimensi bermakna' if s['p'] < .05 else 'korelasi antar fitur lemah, PCA kurang bermanfaat'}.",
                 f"KMO = {s['kmo']:.2f} ({kmo_cat}). Nilai >= 0,6 umumnya dianggap memadai untuk PCA.",
                 f"Rata-rata korelasi absolut antar fitur = {mean_abs:.2f}; korelasi terkuat umumnya antar nilai G1, G2, G3 sehingga informasi nilai redundan dan layak diringkas PCA."]
        bs('<div class="card bg-white text-dark shadow-sm"><div class="card-body"><ul class="mb-0">' +
           "".join(f"<li class='mb-1'>{l}</li>" for l in lines) + "</ul></div></div>", 190)
        st.write("KMO per fitur")
        st.dataframe(s["kmo_per"].rename("KMO").round(3).to_frame().T, use_container_width=True)

    if st.button("Lanjut ke Proses Pengolahan"):
        st.switch_page("pages/4_Proses_Pengolahan.py")