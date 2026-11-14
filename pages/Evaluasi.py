import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("✅ Evaluasi")

st.header("✅ Evaluasi")
if not need_result():
    r = st.session_state.res
    M = r["metrics"]; K = r["K"]

    def sil_cat(s):
        return ("Kuat" if s > .7 else "Cukup baik" if s > .5 else "Lemah" if s > .25 else "Sangat lemah / tumpang tindih")

    pcn = (M["Partition Coefficient"] - 1 / K) / (1 - 1 / K)
    stab = r["stab"].mean()
    stat_cards([("Silhouette (PCA)", f"{M['Silhouette (ruang PCA)']:.3f}", "primary"),
                ("Partition Coefficient", f"{M['Partition Coefficient']:.3f}", "success"),
                ("Explained variance", f"{M['Explained variance (komponen terpilih)']:.1%}", "warning"),
                ("Stabilitas (ARI)", f"{stab:.3f}", "secondary")], 120)
    rows = [
        ("Silhouette (ruang PCA)", M["Silhouette (ruang PCA)"], "−1…1, ↑", f"Struktur cluster: {sil_cat(M['Silhouette (ruang PCA)'])}"),
        ("Silhouette (ruang fitur asli)", M["Silhouette (ruang fitur asli)"], "−1…1, ↑", f"{sil_cat(M['Silhouette (ruang fitur asli)'])} (ruang 16 fitur terstandardisasi)"),
        ("Partition Coefficient", M["Partition Coefficient"], f"{1/K:.2f}…1, ↑", f"Ketegasan partisi ternormalisasi {pcn:.0%} (0% = sepenuhnya kabur, 100% = tegas)"),
        ("Partition Entropy", M["Partition Entropy"], f"0…{np.log(K):.2f}, ↓", "Semakin kecil semakin tegas"),
        ("Davies-Bouldin", M["Davies-Bouldin"], "≥0, ↓", "Semakin kecil cluster semakin terpisah"),
        ("Calinski-Harabasz", M["Calinski-Harabasz"], "↑", "Rasio dispersi antar/dalam cluster"),
        ("Explained variance (komponen terpilih)", M["Explained variance (komponen terpilih)"], "0…1, ↑", f"Informasi yang dipertahankan dari {len(FEATURES)} fitur oleh {r['n_comp']} komponen"),
        ("Explained variance (PC1+PC2)", M["Explained variance (PC1+PC2)"], "0…1, ↑", "Informasi pada plot 2D (visualisasi hanya proyeksi)"),
        ("Fungsi objektif J", M["Fungsi objektif J"], "↓", "Nilai akhir kriteria FCM"),
        ("Stabilitas ARI antar-seed (rata-rata)", stab, "−1…1, ↑", "Konsistensi hasil pada inisialisasi acak berbeda"),
    ]
    ev = pd.DataFrame(rows, columns=["Metrik", "Nilai", "Rentang / arah", "Interpretasi"]).set_index("Metrik")
    st.subheader("Metrik evaluasi")
    bs_table(ev, 120 + 44 * len(ev), "{:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        y0 = 0
        for j in range(K):
            s = np.sort(r["sil_samples"][r["labels"] == j])
            ax.fill_betweenx(np.arange(y0, y0 + len(s)), 0, s, color=CMAP(j), alpha=.8, label=f"C{j}")
            y0 += len(s) + 8
        ax.axvline(M["Silhouette (ruang PCA)"], color="red", ls="--")
        ax.set(title="Silhouette per siswa", xlabel="Koefisien silhouette"); ax.set_yticks([]); ax.legend(fontsize=8)
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        kt = r["k_table"]
        ax.plot(kt.index, kt["Silhouette"], "o-", label="Silhouette")
        ax.plot(kt.index, kt["Partition Coefficient"], "s--", label="PC")
        ax.axvline(K, color="red", ls=":"); ax.set(title="Metrik vs jumlah cluster", xlabel="k"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    sp = pd.DataFrame({"Cluster": [f"C{j}: {n}" for j, n in enumerate(r["names"])],
                       "Rata-rata silhouette": [r["sil_samples"][r["labels"] == j].mean() for j in range(K)]}).set_index("Cluster")
    st.write("Silhouette rata-rata per cluster"); st.dataframe(sp.round(4), use_container_width=True)

    st.subheader("Kualitas & keterbatasan")
    good = [f"PCA mempertahankan {M['Explained variance (komponen terpilih)']:.1%} varians dengan {r['n_comp']} dari {len(FEATURES)} fitur.",
            f"Hasil {'stabil' if stab > .9 else 'cukup stabil' if stab > .7 else 'kurang stabil'} terhadap inisialisasi acak (ARI rata-rata {stab:.2f}).",
            f"Partisi fuzzy {'tegas' if pcn > .6 else 'cukup tegas' if pcn > .3 else 'kabur'} (PC ternormalisasi {pcn:.0%})."]
    lim = [f"Silhouette {M['Silhouette (ruang PCA)']:.2f} tergolong <b>{sil_cat(M['Silhouette (ruang PCA)']).lower()}</b>: data survei siswa tidak memiliki kelompok yang terpisah jelas.",
           "Nilai G1–G3 sangat dominan pada PC1, sehingga pengelompokan sebagian besar mencerminkan tingkat prestasi.",
           "Hanya 16 fitur numerik yang dipakai; atribut kategorikal tidak ikut dipertimbangkan.",
           "Hasil bergantung pada fuzzifier m, jumlah komponen, dan seed; m = 2 dapat menghasilkan solusi degenerat pada data ini.",
           "Tidak ada label kebenaran (ground truth); evaluasi bersifat internal dan nama cluster adalah interpretasi dari profil rata-rata.",
           "Dataset hanya mata pelajaran Matematika dari dua sekolah di Portugal (±395 siswa) sehingga generalisasi terbatas.",
           "Visualisasi 2D hanya memuat sebagian varians, sehingga cluster dapat tampak saling menumpuk."]
    bs('<div class="row"><div class="col-md-5 mb-3"><div class="card h-100 border-success bg-white text-dark"><div class="card-header bg-success text-white">Kelebihan</div><ul class="list-group list-group-flush">' +
       "".join(f'<li class="list-group-item small">{g}</li>' for g in good) +
       '</ul></div></div><div class="col-md-7 mb-3"><div class="card h-100 border-danger bg-white text-dark"><div class="card-header bg-danger text-white">Keterbatasan</div><ul class="list-group list-group-flush">' +
       "".join(f'<li class="list-group-item small">{g}</li>' for g in lim) + "</ul></div></div></div>", 480)
