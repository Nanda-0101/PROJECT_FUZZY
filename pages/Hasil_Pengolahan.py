import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Hasil Pengolahan")

st.header("Hasil Pengolahan")
if not need_result():
    r = st.session_state.res
    K, names, info = r["K"], r["names"], r["cluster_info"]
    rd = r["result_df"]
    stat_cards([("Jumlah siswa", len(rd), "primary"), ("Jumlah cluster", K, "success"),
                ("Silhouette", f"{r['metrics']['Silhouette (ruang PCA)']:.3f}", "warning"),
                ("Partition Coefficient", f"{r['metrics']['Partition Coefficient']:.3f}", "secondary")], 120)

    st.subheader("Arti setiap cluster")
    clr = ["primary", "success", "warning", "danger", "info", "secondary", "dark", "primary"]
    cards = "".join(f"""<div class="col-md-{12 // min(K, 3) if K <= 3 else 6} mb-3"><div class="card h-100 border-{clr[i % 8]} shadow-sm bg-white text-dark">
<div class="card-header bg-{clr[i % 8]} text-white"><b>Cluster {i}</b> · {c['jumlah']} siswa</div>
<div class="card-body"><h6 class="text-{clr[i % 8] if clr[i % 8] != 'warning' else 'dark'}">{c['nama']}</h6>
<p class="small">{c['deskripsi']}</p><p class="small mb-0"><i class="bi bi-lightbulb text-warning me-1"></i><b>Tindak lanjut:</b> {c['saran']}</p></div></div></div>"""
                    for i, c in enumerate(info))
    bs(f'<div class="row">{cards}</div>', 330 if K <= 3 else 330 * ((K + 1) // 2))

    st.subheader("Visualisasi")
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(6.2, 5))
        ev = r["pca2"].explained_variance_ratio_
        for j in range(K):
            pts = r["X_vis"][r["labels"] == j]
            ax.scatter(pts[:, 0], pts[:, 1], s=22, alpha=.7, color=CMAP(j), label=names[j])
        cv = r["pca2"].transform(r["pca"].inverse_transform(r["centers"]))
        ax.scatter(cv[:, 0], cv[:, 1], marker="X", s=200, c="black", edgecolor="white", label="Pusat cluster")
        ax.set(title="Sebaran siswa pada PC1-PC2", xlabel=f"PC1 ({ev[0]:.1%})", ylabel=f"PC2 ({ev[1]:.1%})")
        ax.legend(fontsize=7, loc="best"); st.pyplot(fig); plt.close(fig)
    with v2:
        fig, ax = plt.subplots(figsize=(6.2, 5))
        sc = ax.scatter(r["X_vis"][:, 0], r["X_vis"][:, 1], c=r["U"].max(1), cmap="viridis", s=24)
        plt.colorbar(sc, label="Derajat keanggotaan tertinggi")
        ax.set(title="Ketegasan keanggotaan fuzzy", xlabel="PC1", ylabel="PC2"); st.pyplot(fig); plt.close(fig)
    v3, v4 = st.columns(2)
    with v3:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        cnt = [c["jumlah"] for c in info]
        ax.barh([f"C{i}: {n}" for i, n in enumerate(names)][::-1], cnt[::-1], color=[CMAP(i) for i in range(K)][::-1])
        for i, v in enumerate(cnt[::-1]):
            ax.text(v, i, f" {v}", va="center")
        ax.set(title="Jumlah siswa per cluster"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with v4:
        mu, sd = st.session_state.df[FEATURES].mean(), st.session_state.df[FEATURES].std()
        z = ((r["profile"] - mu) / sd).T
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"C{i}" for i in range(K)])
        ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8); ax.grid(False)
        for a in range(z.shape[0]):
            for b in range(z.shape[1]):
                ax.text(b, a, f"{z.values[a, b]:.1f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, label="z-score vs rata-rata"); ax.set(title="Profil cluster (z-score)")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Profil rata-rata fitur per cluster")
    prof = r["profile"].T.copy(); prof.columns = [f"C{i}: {n}" for i, n in enumerate(names)]
    st.dataframe(prof.round(2), use_container_width=True)

    st.subheader("Tabel hasil per siswa")
    pick = st.multiselect("Filter cluster", names, default=names)
    view = rd[rd["Nama Cluster"].isin(pick)].copy()
    view = view.rename(columns={f"U{j}": f"Keanggotaan C{j}" for j in range(K)})
    st.dataframe(view.style.format({c: "{:.3f}" for c in view.columns if c.startswith("Keanggotaan")}),
                 use_container_width=True, height=360)
    st.download_button("Unduh hasil (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "hasil_cluster_siswa.csv", "text/csv")

    st.subheader("Interpretasi")
    maxu = r["U"].max(1)
    amb = float((maxu < 0.6).mean())
    biggest = max(info, key=lambda c: c["jumlah"])
    lines = [f"Siswa terbagi menjadi <b>{K} kelompok</b>: " + "; ".join(f"<b>{c['nama']}</b> ({c['jumlah']} siswa)" for c in info) + ".",
             f"Kelompok terbesar adalah <b>{biggest['nama']}</b> ({biggest['jumlah'] / len(rd):.1%}).",
             f"Rata-rata derajat keanggotaan tertinggi {maxu.mean():.2f}; <b>{amb:.1%}</b> siswa memiliki keanggotaan tertinggi &lt; 0,6 sehingga berada di wilayah perbatasan antar kelompok (kelebihan FCM dibanding klasterisasi tegas).",
             f"Variabel yang paling membedakan kelompok adalah nilai (G1-G3) dan riwayat kegagalan; PC1 didominasi nilai (loading G2={r['loadings'].loc['G2','PC1']:.2f}), PC2 oleh konsumsi alkohol dan kebiasaan bergaul."]
    if r["metrics"]["Silhouette (ruang PCA)"] < 0.25:
        lines.append("Silhouette yang rendah menandakan kelompok saling tumpang tindih; gunakan hasil sebagai <b>gambaran kecenderungan</b>, bukan batas tegas.")
    bs('<div class="card bg-white text-dark shadow-sm"><div class="card-body"><ul class="mb-0">' +
       "".join(f"<li class='mb-1'>{l}</li>" for l in lines) + "</ul></div></div>", 120 + 52 * len(lines))

    st.subheader("Prediksi siswa baru")
    alert("info", "Untuk menentukan cluster siswa baru (satu siswa atau banyak siswa via CSV), buka halaman <b>Prediksi Siswa Baru</b>.", 62)
    if st.button("Buka Prediksi Siswa Baru"):
        st.switch_page("pages/6_Prediksi_Siswa_Baru.py")