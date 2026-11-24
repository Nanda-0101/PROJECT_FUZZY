import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Eksplorasi Data")

st.header("Eksplorasi Data")
st.caption("Memahami karakteristik data sebelum diproses: sebaran, korelasi, outlier, "
           "dan kelayakan data untuk PCA. Parameter PCA mengikuti setelan di halaman Proses Pengolahan.")

# STATUS KELENGKAPAN DATA (banner di bagian atas)
if not need_data():
    d = st.session_state.df
    source = st.session_state.get("data_source", "data aktif")

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(
                f"**Data siap dianalisis** - sumber: `{source}` | "
                f"{len(d)} siswa x {len(FEATURES)} fitur numerik"
            )
            st.caption(
                "Seluruh output eksplorasi di bawah ini (distribusi, korelasi, "
                "outlier, dan kelayakan PCA) dihitung otomatis dari data yang telah "
                "Anda input. Ubah data di halaman Input Data, maka halaman ini akan "
                "memuat ulang hasil eksplorasi."
            )
        with c2:
            if st.button("Ganti data", use_container_width=True):
                st.switch_page("pages/Input_Data.py")
        with c3:
            if st.button("Lanjut ke proses pengolahan", type="primary", use_container_width=True):
                st.switch_page("pages/Proses_Pengolahan.py")

    X = d[FEATURES].astype(float)

    # Ambil parameter aktif dari session_state (disinkronkan dengan Proses Pengolahan)
    params = st.session_state.get("params", {}) or {}
    var_thr = float(params.get("var_thr", 0.80))
    pca_seed = int(params.get("seed", 42))

    @st.cache_data(show_spinner=False)
    def _pca_preview(df_in, var_thr, seed):
        Z = StandardScaler().fit_transform(df_in[FEATURES].astype(float).values)
        pca_full = PCA(random_state=seed).fit(Z)
        evr = pca_full.explained_variance_ratio_
        cum = np.cumsum(evr)
        n_comp = int(np.argmax(cum >= var_thr) + 1)
        return evr, cum, n_comp

    evr_prev, cum_prev, n_comp_prev = _pca_preview(d, var_thr, pca_seed)

    stat_cards([("Jumlah siswa", len(d), "primary"),
                ("Jumlah fitur", len(FEATURES), "success"),
                ("Rata-rata G3", f"{d['G3'].mean():.2f}", "warning"),
                ("Siswa G3 < 10", f"{(d['G3'] < 10).mean():.1%}", "danger")], 120)

    t1, t2, t3, t4 = st.tabs(["Distribusi", "Korelasi", "Outlier", "Kelayakan PCA"])

    # TAB 1 - DISTRIBUSI
    with t1:
        fig, axs = plt.subplots(4, 4, figsize=(13, 9))
        for ax, c in zip(axs.ravel(), FEATURES):
            ax.hist(X[c], bins=min(20, max(5, X[c].nunique())),
                    color="#4C72B0", edgecolor="white")
            ax.set_title(f"{c}", fontsize=9)
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        sk = pd.DataFrame({
            "Nama": [FEATURE_INFO[c][0] for c in FEATURES],
            "Rata-rata": X.mean(),
            "Std": X.std(),
            "Min": X.min(),
            "Maks": X.max(),
            "Skewness": X.skew(),
        })
        st.dataframe(sk.round(3), use_container_width=True)
        st.caption("Skewness dengan |nilai| lebih besar dari 1 menandakan sebaran sangat miring "
                   "(mis. absences dan failures), sehingga standardisasi diperlukan agar fitur sebanding.")

    # TAB 2 - KORELASI
    with t2:
        C = X.corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(FEATURES, fontsize=8)
        plt.colorbar(im, label="Korelasi Pearson")
        ax.set_title("Matriks korelasi fitur")
        st.pyplot(fig)
        plt.close(fig)
        pairs = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().reset_index()
        pairs.columns = ["Fitur A", "Fitur B", "Korelasi"]
        pairs["|r|"] = pairs["Korelasi"].abs()
        st.write("Pasangan fitur dengan korelasi terkuat")
        st.dataframe(
            pairs.sort_values("|r|", ascending=False).head(10).drop(columns="|r|").round(3),
            use_container_width=True, hide_index=True,
        )

    # TAB 3 - OUTLIER
    with t3:
        q1, q3 = X.quantile(.25), X.quantile(.75)
        iqr = q3 - q1
        out = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
        ot = pd.DataFrame({
            "Jumlah outlier": out.sum(),
            "Persentase": out.mean() * 100,
        }).sort_values("Jumlah outlier", ascending=False)
        c1, c2 = st.columns([1, 1.4])
        c1.dataframe(ot.round(2), use_container_width=True, height=470)
        Z = (X - X.mean()) / X.std()
        fig, ax = plt.subplots(figsize=(7, 5.4))
        ax.boxplot([Z[c] for c in FEATURES], vert=False, flierprops=dict(markersize=3))
        ax.set_yticks(range(1, len(FEATURES) + 1))
        ax.set_yticklabels(FEATURES)
        ax.set_xlabel("z-score")
        ax.set_title("Boxplot fitur terstandardisasi")
        c2.pyplot(fig)
        plt.close(fig)
        alert("info",
              "Outlier tidak dibuang: nilai ekstrem seperti absensi tinggi atau nilai 0 "
              "adalah data siswa yang sah dan justru informatif untuk mendeteksi siswa berisiko.",
              75)

    # TAB 4 - KELAYAKAN PCA
    with t4:
        s = core.pca_suitability(d)

        # Panel ubah parameter PCA
        with st.expander("Ubah Parameter PCA", expanded=False):
            st.caption("Perubahan di sini akan diteruskan ke halaman Proses Pengolahan.")
            new_thr = st.slider("Ambang explained variance PCA",
                                0.50, 0.99, var_thr, 0.01, key="eksp_var_thr")
            new_seed = st.number_input("Random seed",
                                       0, 99999, pca_seed, key="eksp_seed")
            if st.button("Sinkronkan ke Proses Pengolahan", key="sync_pca_par"):
                st.session_state.params = {
                    **st.session_state.params,
                    "var_thr": float(new_thr),
                    "seed": int(new_seed),
                }
                st.success(
                    f"Parameter diperbarui: ambang = {new_thr:.0%}, "
                    f"seed = {new_seed}. Halaman Proses Pengolahan akan memakai "
                    f"nilai ini saat algoritma dijalankan."
                )
                st.rerun()

        # Kartu ringkasan PCA
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jumlah komponen PCA terpilih",
                      f"{n_comp_prev} dari {len(FEATURES)}")
            c2.metric("Total varians dipertahankan",
                      f"{cum_prev[n_comp_prev - 1]:.2%}")
            c3.metric("Varians PC1 + PC2", f"{cum_prev[1]:.2%}")
            c4.metric("Reduksi dimensi",
                      f"{(1 - n_comp_prev / len(FEATURES)) * 100:.1f}%")

        st.divider()

        # BARTLETT
        st.markdown("### Uji Bartlett's Test of Sphericity")
        st.markdown(
            "**H0:** matriks korelasi fitur = matriks identitas "
            "(tidak ada korelasi antar fitur).  \n"
            "**H1:** minimal ada satu pasangan fitur yang berkorelasi."
        )
        st.latex(r"\chi^2 = -\left(n-1-\frac{2p+5}{6}\right)\cdot \ln\det(R)")
        st.latex(r"df = \frac{p(p-1)}{2}")

        with st.expander("Langkah perhitungan Bartlett", expanded=True):
            st.markdown(f"""
            1. **n** = {s['n']}, **p** = {s['p']}
            2. **ln det(R)** = {s['bartlett']['logdet']:.6f}
            3. **Faktor koreksi** = n - 1 - (2p+5)/6
               = {s['n']} - 1 - (2.{s['p']}+5)/6
               = {s['n'] - 1 - (2*s['p'] + 5)/6:.4f}
            4. **chi2** = - ({s['n']-1 - (2*s['p']+5)/6:.4f}) x ({s['bartlett']['logdet']:.6f})
               = **{s['bartlett']['chi2']:.4f}**
            5. **df** = {s['p']} x ({s['p']}-1)/2 = **{s['bartlett']['dof']}**
            6. **p-value** = P(chi2 dengan df {s['bartlett']['dof']} > {s['bartlett']['chi2']:.2f})
               = **{s['bartlett']['p']:.6g}**
            """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Bartlett chi2", f"{s['bartlett']['chi2']:.2f}")
        c2.metric("df", s["bartlett"]["dof"])
        c3.metric("p-value",
                  "< 0.001" if s["bartlett"]["p"] < .001 else f"{s['bartlett']['p']:.4f}")

        if s["bartlett_sig"]:
            alert("success",
                  "Bartlett signifikan (p < 0,05). Matriks korelasi berbeda dari matriks identitas "
                  "sehingga ada korelasi antar fitur dan PCA bermakna.", 75)
        else:
            alert("warning",
                  "Bartlett tidak signifikan (p >= 0,05). Korelasi antar fitur lemah "
                  "sehingga PCA kurang bermanfaat pada data ini.", 75)

        st.divider()

        # KMO
        st.markdown("### Uji Kaiser-Meyer-Olkin (KMO)")
        st.markdown(
            "KMO mengukur proporsi varians yang mungkin disebabkan faktor bersama. "
            "Nilai rendah berarti korelasi parsial besar sehingga data tidak cocok untuk PCA."
        )
        st.latex(
            r"KMO = \frac{\sum_{j\neq k} r_{jk}^2}"
            r"{\sum_{j\neq k} r_{jk}^2 + \sum_{j\neq k} a_{jk}^2}"
        )
        st.markdown(
            r"dengan $r_{jk}$ = korelasi sederhana (matriks $R$), "
            r"$a_{jk}$ = korelasi parsial atau anti-image "
            r"($A = -D^{-1/2} R^{-1} D^{-1/2}$, $D = \mathrm{diag}(R^{-1})$)."
        )

        with st.expander("Langkah perhitungan KMO", expanded=True):
            st.markdown(f"""
            1. Hitung invers matriks korelasi R-inv (ukuran {s['p']} x {s['p']}).
            2. Hitung matriks anti-image:
               A = -R-inv dibagi sqrt(diag(R-inv) x diag(R-inv) transpos).
            3. Nolkan diagonal R dan A (hanya off-diagonal yang dihitung).
            4. Jumlah kuadrat korelasi sederhana off-diagonal
               sum r_jk^2 = **{s['kmo_num']:.4f}**
            5. Jumlah kuadrat korelasi parsial off-diagonal
               sum a_jk^2 = **{s['kmo_den'] - s['kmo_num']:.4f}**
            6. **KMO** = {s['kmo_num']:.4f} / ({s['kmo_num']:.4f} + {s['kmo_den']-s['kmo_num']:.4f})
               = **{s['kmo']:.4f}** sehingga kategori **{s['kmo_cat']}**
            """)

        st.markdown("**Skala interpretasi KMO (Kaiser, 1974):**")
        skala = pd.DataFrame([
            ("0.90 - 1.00", "Marvelous", "sangat baik"),
            ("0.80 - 0.89", "Meritorious", "berjasa"),
            ("0.70 - 0.79", "Middling", "cukup baik"),
            ("0.60 - 0.69", "Mediocre", "sedang"),
            ("0.50 - 0.59", "Miserable", "buruk"),
            ("< 0.50", "Unacceptable", "tidak dapat diterima"),
        ], columns=["Rentang", "Kategori", "Arti"])
        st.dataframe(skala, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("KMO Total", f"{s['kmo']:.3f}")
        c2.metric("Kategori KMO", s["kmo_cat"])

        if s["kmo"] >= 0.60:
            alert("success",
                  f"KMO = {s['kmo']:.3f} ({s['kmo_cat']}). Nilai ini memadai "
                  f"untuk PCA (ambang umum: 0,60; baik: 0,70; sangat baik: 0,80).", 75)
        else:
            alert("warning",
                  f"KMO = {s['kmo']:.3f} ({s['kmo_cat']}). Nilai ini di bawah 0,60, "
                  f"sehingga PCA perlu dipertimbangkan ulang.", 75)

        st.divider()

        # KESIMPULAN + NAVIGASI
        st.markdown("### Kesimpulan Kelayakan PCA")
        layak = s["bartlett_sig"] and s["kmo"] >= 0.60
        if layak:
            alert(
                "success",
                f"Data LAYAK untuk PCA. Bartlett signifikan "
                f"(p = {s['bartlett']['p']:.2g}) dan KMO = {s['kmo']:.3f} "
                f"({s['kmo_cat']}). Dengan ambang {var_thr:.0%}, PCA akan memilih "
                f"{n_comp_prev} komponen yang mempertahankan "
                f"{cum_prev[n_comp_prev - 1]:.2%} varians.",
                110,
            )
        else:
            alasan = []
            if not s["bartlett_sig"]:
                alasan.append("Bartlett tidak signifikan")
            if s["kmo"] < 0.60:
                alasan.append(f"KMO = {s['kmo']:.3f} di bawah 0,60")
            alert("warning",
                  "PCA perlu dipertimbangkan ulang. Alasan: " + "; ".join(alasan) + ".",
                  110)

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("Kembali ke Input Data", use_container_width=True):
            st.switch_page("pages/Input_Data.py")
        if c2.button("Lanjut ke Proses Pengolahan", type="primary",
                     use_container_width=True):
            st.switch_page("pages/Proses_Pengolahan.py")