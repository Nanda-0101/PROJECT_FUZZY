import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Prediksi Siswa Baru")

st.header("Prediksi Siswa Baru")
st.caption("Menentukan cluster siswa yang tidak ada di dataset latih: data distandarisasi dengan scaler yang sama, "
           "diproyeksikan ke PCA, lalu dihitung derajat keanggotaan fuzzy terhadap pusat cluster.")

if not need_result():
    r = st.session_state.res
    t1, t2 = st.tabs(["Satu siswa", "Banyak siswa (CSV)"])

    with t1:
        default = st.session_state.get("manual", {c: FEATURE_INFO[c][3] for c in FEATURES})
        with st.form("pred_form"):
            vals = {}
            cols = st.columns(4)
            for i, c in enumerate(FEATURES):
                lbl, lo, hi, de, desc = FEATURE_INFO[c]
                with cols[i % 4]:
                    vals[c] = st.number_input(f"{lbl} ({c})", min_value=float(lo - 100), max_value=float(hi + 1000),
                                              value=float(default.get(c, de)), step=1.0, key=f"pred_{c}",
                                              help=f"{desc} Rentang valid {lo}-{hi}.")
            ok = st.form_submit_button("Prediksi Cluster", type="primary")
        if ok:
            errs = core.validate_single(vals)
            if errs:
                for e in errs:
                    alert("danger", e, 58)
            else:
                st.session_state.manual = vals

        if "manual" in st.session_state and not (ok and core.validate_single(vals)):
            u, zv = render_prediction(r, st.session_state.manual)

            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            for j in range(r["K"]):
                pts = r["X_vis"][r["labels"] == j]
                ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=.45, color=CMAP(j), label=r["names"][j])
            ax.scatter(zv[0], zv[1], marker="*", s=420, c="gold", edgecolor="black", label="Siswa baru", zorder=5)
            ax.set(title="Posisi siswa baru pada PC1-PC2", xlabel="PC1", ylabel="PC2")
            ax.legend(fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

            # =================================================================
            # RUMUS & PROSES INFERENCE PREDIKSI CLUSTER
            # =================================================================
            st.markdown("---")
            st.subheader("Rumus & Proses Inference Prediksi Cluster")
            st.markdown("""
**Alur inference** untuk siswa baru (data mentah -> cluster):

1. **Standardisasi** dengan scaler yang sudah di-fit pada data latih.
2. **Proyeksi PCA** dengan komponen yang sudah di-fit pada data latih.
3. **Hitung jarak** ke setiap pusat cluster di ruang PCA.
4. **Hitung derajat keanggotaan** fuzzy.
5. **Pilih cluster** dengan derajat keanggotaan tertinggi.
            """)

            # Ambil data latih & parameter dari hasil training
            scaler = r["scaler"]
            pca = r["pca"]
            pca2 = r["pca2"]
            centers = r["centers"]          # (K, n_comp) di ruang PCA
            m_val = r["params"]["m"]
            K = r["K"]
            n_comp = r["n_comp"]

            values = st.session_state.manual

            # LANGKAH 1: Standardisasi
            with st.expander("Langkah 1 - Standardisasi (z-score) dengan scaler data latih", expanded=True):
                st.markdown("""
**Rumus:**

$$z_{j} = \\dfrac{x_{j} - \\bar{x}_j}{s_j}$$

dengan $\\bar{x}_j$ dan $s_j$ diambil dari **data latih** (mean dan std
yang tersimpan dalam objek `StandardScaler`), bukan dari data siswa baru.
                """)

                x_raw = np.array([[float(values[c]) for c in FEATURES]])
                z_new = scaler.transform(x_raw)[0]

                df_l1 = pd.DataFrame({
                    "Fitur": FEATURES,
                    "x (input)": x_raw[0],
                    "mean latih": scaler.mean_,
                    "std latih": scaler.scale_,
                    "z = (x - mean)/std": z_new,
                })
                st.dataframe(df_l1.round(4), use_container_width=True, hide_index=True)
                st.caption(
                    "Contoh perhitungan fitur pertama: "
                    f"z = ({x_raw[0, 0]:.4f} - {scaler.mean_[0]:.4f}) / {scaler.scale_[0]:.4f} "
                    f"= {z_new[0]:.4f}"
                )

            # LANGKAH 2: Proyeksi PCA
            with st.expander("Langkah 2 - Proyeksi ke ruang PCA", expanded=True):
                st.markdown(f"""
**Rumus:**

$$z^{{\\text{{PCA}}}} = W^T (z - \\mu_{{\\text{{PCA}}}})$$

Pada dasarnya: vektor $z$ (16 dimensi) dikalikan dengan matriks komponen
utama $W$ berukuran (16 x {n_comp}) yang diperoleh dari data latih, kemudian
dikurangi rata-rata PCA jika ada. Hasilnya vektor berdimensi {n_comp}.
                """)

                zp_new = pca.transform(z_new.reshape(1, -1))[0]

                df_l2 = pd.DataFrame({
                    "Komponen": [f"PC{i+1}" for i in range(n_comp)],
                    "Nilai siswa baru": zp_new,
                })
                st.dataframe(df_l2.round(4), use_container_width=True, hide_index=True)
                st.caption(
                    f"Vektor 16 fitur diringkas menjadi {n_comp} komponen utama. "
                    f"Informasi ini yang akan dipakai untuk menghitung jarak ke pusat cluster."
                )

            # LANGKAH 3: Jarak ke setiap pusat cluster
            with st.expander("Langkah 3 - Jarak Euclidean ke setiap pusat cluster", expanded=True):
                st.markdown(f"""
**Rumus:**

$$d_k = \\sqrt{{\\sum_{{j=1}}^{{{n_comp}}} (z^{{\\text{{PCA}}}}_j - v_{{kj}})^2}}$$

dengan $v_{{kj}}$ = komponen ke-$j$ dari pusat cluster ke-$k$ (dalam ruang PCA).
                """)

                diff = zp_new[None, :] - centers              # (K, n_comp)
                d_new = np.sqrt((diff ** 2).sum(axis=1))      # (K,)
                d_safe = np.maximum(d_new, 1e-12)

                df_l3 = pd.DataFrame({
                    "Cluster": [f"C{i+1}: {r['names'][i]}" for i in range(K)],
                    "Jarak d_k": d_new,
                })
                st.dataframe(df_l3.round(4), use_container_width=True, hide_index=True)
                st.caption(
                    "Jarak Euclidean diukur pada ruang PCA (bukan ruang fitur asli). "
                    "Semakin kecil nilai d_k, semakin dekat siswa ke pusat cluster k."
                )

            # LANGKAH 4: Derajat keanggotaan fuzzy
            with st.expander("Langkah 4 - Derajat keanggotaan fuzzy (FCM)", expanded=True):
                st.markdown(f"""
**Rumus FCM untuk keanggotaan siswa baru:**

$$u_k = \\dfrac{{d_k^{{-2/(m-1)}}}}{{\\sum_{{l=1}}^{{c}} d_l^{{-2/(m-1)}}}}$$

dengan:
- $d_k$ = jarak siswa ke pusat cluster $k$,
- $m$ = fuzzifier (parameter model saat ini: **{m_val}**),
- jumlah $u_k$ untuk semua $k$ = **1**.
                """)

                exp_val = -2.0 / (m_val - 1.0)
                P = d_safe ** exp_val
                u_new = P / P.sum()

                df_l4 = pd.DataFrame({
                    "Cluster": [f"C{i+1}: {r['names'][i]}" for i in range(K)],
                    "d_k": d_new,
                    f"d_k^{exp_val:.4f}": P,
                })
                df_l4.loc["TOTAL"] = ["-", np.nan, P.sum()]
                st.dataframe(df_l4.round(6), use_container_width=True, hide_index=True)

                df_l5 = pd.DataFrame({
                    "Cluster": [f"C{i+1}: {r['names'][i]}" for i in range(K)],
                    "u_k (derajat keanggotaan)": u_new,
                })
                st.dataframe(df_l5.round(4), use_container_width=True, hide_index=True)
                st.caption(f"Cek jumlah u_k = **{u_new.sum():.6f}** (harus sama dengan 1).")

            # LANGKAH 5: Cluster dengan keanggotaan tertinggi
            with st.expander("Langkah 5 - Tentukan cluster akhir (argmax)", expanded=True):
                st.markdown("""
**Aturan keputusan:**

$$k^{*} = \\arg\\max_k \\, u_k$$

Cluster akhir = cluster dengan derajat keanggotaan fuzzy terbesar.
                """)

                k_best = int(np.argmax(u_new))
                df_l6 = pd.DataFrame({
                    "Cluster": [f"C{i+1}" for i in range(K)],
                    "u_k": u_new,
                    "Terpilih": ["<- ya" if i == k_best else "" for i in range(K)],
                })
                st.dataframe(df_l6.round(4), use_container_width=True, hide_index=True)

                st.success(
                    f"Siswa baru masuk ke **Cluster {k_best + 1}: {r['names'][k_best]}** "
                    f"dengan derajat keanggotaan **{u_new[k_best]:.4f}** "
                    f"({u_new[k_best]*100:.2f}% keyakinan fuzzy)."
                )

                if u_new.max() < 0.6:
                    st.warning(
                        "Derajat keanggotaan tertinggi < 0,60: siswa berada di wilayah perbatasan "
                        "antar cluster. Pertimbangkan untuk memeriksa cluster kedua atau "
                        "meminta data tambahan untuk mempertegas posisi."
                    )

            # VISUALISASI POSISI RELATIF
            st.markdown("#### Posisi siswa baru relatif terhadap pusat cluster (ruang PC1-PC2)")
            st.caption(
                "Grafik ini memproyeksikan siswa baru dan pusat cluster ke 2 komponen utama pertama "
                "untuk visualisasi. Perhitungan jarak dan keanggotaan tetap memakai semua komponen "
                "PCA terpilih, bukan hanya PC1-PC2."
            )
            zv_new = pca2.transform(z_new.reshape(1, -1))[0]
            zv_centers = pca2.transform(pca.inverse_transform(centers))

            fig, ax = plt.subplots(figsize=(7, 5))
            for j in range(K):
                pts = r["X_vis"][r["labels"] == j]
                ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=.35, color=CMAP(j), label=r["names"][j])
            ax.scatter(zv_centers[:, 0], zv_centers[:, 1], marker="X", s=200,
                       c="black", edgecolor="white", label="Pusat cluster", zorder=4)
            ax.scatter(zv_new[0], zv_new[1], marker="*", s=420, c="gold",
                       edgecolor="black", label="Siswa baru", zorder=5)
            ax.set(title="Posisi relatif siswa baru", xlabel="PC1", ylabel="PC2")
            ax.legend(fontsize=7, loc="best")
            st.pyplot(fig)
            plt.close(fig)

    # TAB 2 - BANYAK SISWA (CSV)
    with t2:
        st.write("Unggah CSV siswa baru dengan kolom: " + ", ".join(f"`{c}`" for c in FEATURES) + ".")
        tpl = pd.DataFrame([{c: FEATURE_INFO[c][3] for c in FEATURES}])
        st.download_button("Unduh template CSV", tpl.to_csv(index=False, sep=";").encode("utf-8"),
                           "template_siswa_baru.csv", "text/csv")
        up = st.file_uploader("Unggah CSV siswa baru", type=["csv"], key="pred_upload")
        if up is not None:
            try:
                clean, msgs, info = core.validate_dataset(core.read_dataset(up))
                alerts(msgs)
                if clean is not None:
                    out = core.predict_batch(r, clean)
                    cnt = out["Nama Cluster"].value_counts().rename("Jumlah siswa").to_frame()
                    cnt["Persentase"] = (cnt["Jumlah siswa"] / len(out) * 100).round(1)
                    c1, c2 = st.columns([1, 1.6])
                    c1.dataframe(cnt, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(6, 3.6))
                    zv = r["pca2"].transform(r["scaler"].transform(out[FEATURES].values))
                    ax.scatter(r["X_vis"][:, 0], r["X_vis"][:, 1], s=14, c="lightgray", label="Data latih")
                    for j in range(r["K"]):
                        m_ = out["Cluster"].values == j
                        ax.scatter(zv[m_, 0], zv[m_, 1], s=40, color=CMAP(j),
                                   edgecolor="black", label=r["names"][j])
                    ax.set(title="Siswa baru pada PC1-PC2", xlabel="PC1", ylabel="PC2")
                    ax.legend(fontsize=7)
                    c2.pyplot(fig)
                    plt.close(fig)
                    view = out.rename(columns={f"U{j}": f"Keanggotaan C{j}" for j in range(r["K"])})
                    st.dataframe(view.style.format({c: "{:.3f}" for c in view.columns if c.startswith("Keanggotaan")}),
                                 use_container_width=True, height=320)
                    st.download_button("Unduh hasil prediksi (CSV)", view.to_csv(index=False).encode("utf-8"),
                                       "prediksi_siswa_baru.csv", "text/csv")
            except Exception as e:
                alert("danger", f"File tidak dapat diproses: {e}")