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
            (1, "Seleksi & Standardisasi", "16 fitur numerik dipilih lalu di-standardisasi (z-score) agar skala yang berbeda (mis. absences 0-93 vs famrel 1-5) tidak mendominasi jarak."),
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

        # TAHAP 1 - STANDARDISASI
        with st.expander("Tahap 1 - Standardisasi", expanded=False):
            st.markdown("""
**Tujuan:** menyetarakan skala setiap fitur agar fitur dengan rentang besar
(mis. `absences` 0-93) tidak mendominasi perhitungan jarak Euclidean pada FCM.

**Rumus z-score:**

$$z_{ij} = \\dfrac{x_{ij} - \\bar{x}_j}{s_j}$$

dengan $x_{ij}$ = nilai asli, $\\bar{x}_j$ = rata-rata kolom ke-$j$, dan
$s_j$ = simpangan baku kolom ke-$j$.

**Hasil yang diharapkan:** setiap kolom memiliki rata-rata 0 dan simpangan
baku 1, sehingga seluruh fitur berada pada skala yang sebanding.
            """)
            a = st.session_state.df[FEATURES].head(8)
            b = pd.DataFrame(r["X"][:8], columns=FEATURES)
            st.write("Data asli (8 baris pertama)")
            st.dataframe(a, use_container_width=True)
            st.write("Setelah z-score (rata-rata 0, simpangan baku 1)")
            st.dataframe(b.round(3), use_container_width=True)

        # TAHAP 2 - PCA
        with st.expander("Tahap 2 - PCA: explained variance", expanded=True):
            st.markdown("""
**Tujuan:** mereduksi 16 fitur menjadi beberapa komponen utama yang saling
ortogonal dan menangkap sebagian besar variansi data.

**Rumus utama PCA:**

1. Matriks kovarians dari data terstandardisasi:

$$C = \\dfrac{1}{n-1} X^T X$$

2. Eigen-decomposition:

$$C = V \\Lambda V^{-1}$$

dengan $\\Lambda$ = matriks diagonal nilai eigen ($\\lambda_1 \\geq \\lambda_2 \\geq \\dots$) dan $V$ = matriks eigenvector.

3. Proyeksi data ke $k$ komponen teratas:

$$Z = X \\cdot V_k$$

4. Explained variance ratio per komponen:

$$\\text{EVR}_i = \\dfrac{\\lambda_i}{\\sum_{j=1}^{p} \\lambda_j}$$

5. Kumulatif:

$$\\text{EVR}_{\\text{kum}}(k) = \\sum_{i=1}^{k} \\text{EVR}_i$$

**Aturan pemilihan $k$:** jumlah komponen minimum yang memenuhi
$\\text{EVR}_{\\text{kum}}(k) \\geq$ ambang (parameter di halaman ini).
            """)
            evr_t = pd.DataFrame({"Komponen": [f"PC{i+1}" for i in range(len(r["evr"]))],
                                  "Explained variance": r["evr"], "Kumulatif": r["cum"]}).set_index("Komponen")
            c1, c2 = st.columns([1, 1.4])
            c1.dataframe(evr_t.round(4), use_container_width=True, height=400)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            x = np.arange(1, len(r["evr"]) + 1)
            ax[0].bar(x, r["evr"], color="#4C72B0")
            ax[0].set(title="Scree plot", xlabel="Komponen", ylabel="Rasio varians")
            ax[1].plot(x, r["cum"], "o-", color="#DD8452")
            ax[1].axhline(newp["var_thr"], ls="--", color="gray")
            ax[1].axvline(r["n_comp"], ls="--", color="red")
            ax[1].set(title="Kumulatif", xlabel="Jumlah komponen")
            plt.tight_layout()
            c2.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"Dipilih {r['n_comp']} komponen (kumulatif >= {newp['var_thr']:.0%}). "
                f"Total varians yang dipertahankan: {r['cum'][r['n_comp']-1]:.2%}."
            )

        # TAHAP 3 - PEMILIHAN K
        with st.expander("Tahap 3 - Pemilihan jumlah cluster (k)", expanded=True):
            st.markdown("""
**Tujuan:** memilih jumlah cluster $k$ terbaik dengan membandingkan kualitas
partisi dari beberapa kandidat $k$ pada ruang PCA.

**Prosedur:**
1. Untuk tiap $k \\in \\{2, 3, \\dots, k_{\\max}\\}$, jalankan FCM pada $X_{\\text{PCA}}$.
2. Hitung **Silhouette Score** dan **Partition Coefficient** dari hasil tiap $k$.
3. Pilih $k$ dengan Silhouette tertinggi (mode otomatis) atau gunakan $k$ manual.

**Rumus Silhouette Score (rata-rata):**

$$s(i) = \\dfrac{b(i) - a(i)}{\\max\\{a(i), b(i)\\}}$$

$$S = \\dfrac{1}{n} \\sum_{i=1}^{n} s(i)$$

dengan:
- $a(i)$ = jarak rata-rata data $i$ ke anggota lain **dalam** cluster yang sama
  (kohesi),
- $b(i)$ = jarak rata-rata data $i$ ke anggota cluster **terdekat lainnya**
  (separasi),
- rentang $S \\in [-1, 1]$; **semakin tinggi semakin baik**.

**Rumus Partition Coefficient (FPC):**

$$\\text{FPC} = \\dfrac{1}{n} \\sum_{i=1}^{n} \\sum_{k=1}^{c} u_{ik}^2$$

- rentang $\\text{FPC} \\in [1/c, 1]$,
- mendekati 1 berarti partisi tegas (crisp), mendekati $1/c$ berarti kabur.

**Catatan:** Silhouette tinggi menunjukkan cluster padat dan terpisah;
FPC tinggi menunjukkan keanggotaan fuzzy yang tegas. Keduanya dipakai bersama
agar tidak memilih $k$ hanya dari satu sudut pandang.
            """)
            kt = r["k_table"].drop(columns="Degenerat")
            st.dataframe(kt.round(4), use_container_width=True)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
            ax[0].plot(kt.index, kt["Silhouette"], "o-")
            ax[0].axvline(r["K"], color="red", ls="--")
            ax[0].set(title="Silhouette vs k (naik lebih baik)", xlabel="k")
            ax[1].plot(kt.index, kt["Partition Coefficient"], "o-", label="PC")
            ax[1].plot(kt.index, 1 / kt.index.values, ":", color="gray", label="batas 1/k")
            ax[1].set(title="Partition Coefficient vs k", xlabel="k")
            ax[1].legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"k otomatis (Silhouette tertinggi) = {r['k_auto']}; k yang dipakai = {r['K']}.")

        # TAHAP 4 - ITERASI FCM
        with st.expander("Tahap 4 - Iterasi FCM & pusat cluster", expanded=False):
            st.markdown("""
**Tujuan:** menjalankan FCM final dengan $k$ terpilih untuk memperoleh pusat
cluster dan derajat keanggotaan akhir setiap siswa.

**Rumus FCM:**

- Pusat cluster: $v_{kj} = \\dfrac{\\sum_i u_{ik}^m \\, x_{ij}}{\\sum_i u_{ik}^m}$
- Jarak: $d_{ik} = \\sqrt{\\sum_j (x_{ij} - v_{kj})^2}$
- Keanggotaan: $u_{ik} = \\dfrac{d_{ik}^{-2/(m-1)}}{\\sum_l d_{il}^{-2/(m-1)}}$
- Fungsi objektif: $J = \\sum_i \\sum_k u_{ik}^m \\, d_{ik}^2$

Proses berulang hingga perubahan maksimum keanggotaan
$\\Delta = \\max_{i,k} |u_{ik}^{(t)} - u_{ik}^{(t-1)}| < \\epsilon$.
            """)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(range(1, len(r["hist"]) + 1), r["hist"])
            ax.set(title="Konvergensi fungsi objektif J", xlabel="Iterasi", ylabel="J")
            st.pyplot(fig)
            plt.close(fig)
            cen = r["scaler"].inverse_transform(r["pca"].inverse_transform(r["centers"]))
            cdf = pd.DataFrame(cen, columns=FEATURES, index=r["names"]).T
            st.write("Pusat cluster (dikembalikan ke skala fitur asli)")
            st.dataframe(cdf.round(2), use_container_width=True)

        # TAHAP 5 - PERHITUNGAN DETAIL FCM
        with st.expander("Tahap 5 - Perhitungan Detail FCM (angka per iterasi)", expanded=False):
            st.markdown("""
**Rumus yang dipakai:**

- Pusat cluster: $v_{kj} = \\dfrac{\\sum_i (u_{ik})^m \\, x_{ij}}{\\sum_i (u_{ik})^m}$
- Jarak Euclidean: $d_{ik} = \\sqrt{\\sum_j (x_{ij} - v_{kj})^2}$
- Pembaruan keanggotaan: $u_{ik} = \\dfrac{d_{ik}^{-2/(m-1)}}{\\sum_l d_{il}^{-2/(m-1)}}$
- Fungsi objektif: $J = \\sum_i \\sum_k (u_{ik})^m d_{ik}^2$
- Berhenti jika $\\Delta = \\max_{i,k} |u_{ik}^{(t)} - u_{ik}^{(t-1)}| < \\epsilon$
            """)

            X_pca = r["X_pca"]
            U_final = r["U"]
            V_final = r["centers"]
            K = r["K"]
            m_val = r["params"]["m"]
            eps_val = r["params"]["tol"]
            n_data, n_comp = X_pca.shape

            st.markdown("#### Matriks data di ruang PCA (n x n_comp)")
            st.caption(f"Data direduksi menjadi {n_comp} komponen utama. Berikut {min(10, n_data)} baris pertama.")
            df_Xpca = pd.DataFrame(X_pca[:min(10, n_data)], columns=[f"PC{i+1}" for i in range(n_comp)])
            st.dataframe(df_Xpca.round(4), use_container_width=True)

            st.markdown("#### Matriks keanggotaan akhir U dan pusat cluster V")
            st.caption(f"U berukuran {n_data} x {K} (dari hasil iterasi FCM). Hanya {min(10, n_data)} baris pertama ditampilkan.")
            df_U = pd.DataFrame(U_final[:min(10, n_data)],
                                columns=[f"u(C{i+1})" for i in range(K)])
            df_U.insert(0, "Data", [f"D{i+1}" for i in range(min(10, n_data))])
            st.dataframe(df_U.round(4), use_container_width=True)

            df_V = pd.DataFrame(V_final, columns=[f"PC{i+1}" for i in range(n_comp)],
                                index=[f"C{i+1}" for i in range(K)])
            st.write("Pusat cluster di ruang PCA:")
            st.dataframe(df_V.round(4), use_container_width=True)

            # RINCIAN LANGKAH FCM
            with st.expander("Simulasi 1 langkah iterasi FCM (dari U_final)", expanded=False):
                st.caption(
                    "Langkah demonstrasi proses hitung: mengambil U_final lalu melakukan satu iterasi "
                    "update pusat -> jarak -> update U -> hitung J. Angka yang ditampilkan adalah hasil "
                    "satu langkah iterasi, sehingga diperoleh gambaran besaran perhitungan per iterasi."
                )

                W = U_final ** m_val
                denom = W.sum(axis=0)
                V_calc = (W.T @ X_pca) / denom[:, None]

                st.markdown("**Langkah 1: hitung pusat cluster V dari U**")
                st.latex(r"v_{kj} = \frac{\sum_i (u_{ik})^m \, x_{ij}}{\sum_i (u_{ik})^m}")
                show_n = min(8, n_data)
                for k in range(K):
                    tab = pd.DataFrame({
                        f"u_i{k+1}": U_final[:show_n, k],
                        f"(u_i{k+1})^m": W[:show_n, k],
                        f"(u^m)*PC1": W[:show_n, k] * X_pca[:show_n, 0],
                    }, index=[f"D{i+1}" for i in range(show_n)])
                    tab.loc["JUMLAH"] = [np.nan, W[:, k].sum(), (W[:, k] * X_pca[:, 0]).sum()]
                    st.write(f"Cluster C{k+1}")
                    st.dataframe(tab.round(4), use_container_width=True)
                    st.caption(
                        f"v(C{k+1}, PC1) = {(W[:, k] * X_pca[:, 0]).sum():.4f} / "
                        f"{W[:, k].sum():.4f} = {V_calc[k, 0]:.4f}"
                    )

                st.markdown("**Langkah 2: hitung jarak Euclidean d_ik**")
                st.latex(r"d_{ik} = \sqrt{\sum_j (x_{ij} - v_{kj})^2}")
                diff = X_pca[:, None, :] - V_calc[None, :, :]
                D = np.sqrt((diff ** 2).sum(axis=2))
                D = np.maximum(D, 1e-12)
                df_D = pd.DataFrame(D[:show_n], columns=[f"d_i(C{i+1})" for i in range(K)],
                                    index=[f"D{i+1}" for i in range(show_n)])
                st.dataframe(df_D.round(4), use_container_width=True)

                st.markdown("**Langkah 3: hitung keanggotaan baru U**")
                st.latex(r"u_{ik} = \frac{d_{ik}^{-2/(m-1)}}{\sum_l d_{il}^{-2/(m-1)}}")
                exp_val = -2.0 / (m_val - 1.0)
                P = D ** exp_val
                U_new = P / P.sum(axis=1, keepdims=True)
                df_Unew = pd.DataFrame(index=[f"D{i+1}" for i in range(show_n)])
                for k in range(K):
                    df_Unew[f"d_i(C{k+1})"] = D[:show_n, k]
                    df_Unew[f"d^{exp_val:.2f}"] = P[:show_n, k]
                df_Unew["jumlah"] = P[:show_n].sum(axis=1)
                for k in range(K):
                    df_Unew[f"u_i(C{k+1}) baru"] = U_new[:show_n, k]
                df_Unew["cek Σu"] = U_new[:show_n].sum(axis=1)
                st.dataframe(df_Unew.round(4), use_container_width=True)

                st.markdown("**Langkah 4: hitung fungsi objektif J**")
                st.latex(r"J = \sum_i \sum_k (u_{ik})^m \, d_{ik}^2")
                kontribusi = (U_new ** m_val) * D ** 2
                df_J = pd.DataFrame(kontribusi[:show_n],
                                    columns=[f"(u_i(C{k+1}))^m x d^2" for k in range(K)],
                                    index=[f"D{i+1}" for i in range(show_n)])
                df_J["jumlah"] = kontribusi[:show_n].sum(axis=1)
                df_J.loc["TOTAL"] = list(kontribusi.sum(axis=0)) + [kontribusi.sum()]
                st.dataframe(df_J.round(4), use_container_width=True)
                st.caption(f"J hasil 1 langkah = **{kontribusi.sum():.4f}**  "
                           f"(J akhir algoritma = {r['J']:.4f})")

                st.markdown("**Langkah 5: delta dan uji konvergensi**")
                delta = float(np.max(np.abs(U_new - U_final)))
                st.write(f"Delta = max|U_baru - U_lama| = **{delta:.6f}**  "
                         f"(toleransi epsilon = {eps_val}).")
                if delta < eps_val:
                    alert("success", "Delta < epsilon -> konvergen, iterasi akan berhenti.", 60)
                else:
                    alert("info", "Delta >= epsilon -> lanjut iterasi berikutnya.", 60)

            # RIWAYAT KONVERGENSI
            st.markdown("#### Riwayat iterasi keseluruhan (dari fungsi FCM)")
            hist = r["hist"]
            it_range = list(range(1, len(hist) + 1))
            df_hist = pd.DataFrame({
                "Iterasi": it_range,
                "J": hist,
                "Delta": [hist[i-1] - hist[i] if i < len(hist) else hist[i-1] - hist[-1] for i in range(len(hist))],
            }).set_index("Iterasi")
            st.dataframe(df_hist.round(6), use_container_width=True)
            st.caption(
                f"Algoritma konvergen setelah **{r['n_iter']} iterasi** dengan "
                f"J akhir = **{r['J']:.6f}** dan toleransi = {eps_val}."
            )

            # INDEKS VALIDITAS (HANYA J & FPC)
            st.markdown("#### Indeks validitas fuzzy (setelah konvergen)")

            def hitung_indeks(U, V, m_):
                n_, c_ = U.shape
                fpc = float((U ** 2).sum() / n_)
                J_ = float(((U ** m_) * (np.linalg.norm(
                    X_pca[:, None, :] - V[None, :, :], axis=2) ** 2)).sum())
                return dict(J=J_, FPC=fpc)

            idx = hitung_indeks(U_final, V_final, m_val)
            df_idx = pd.DataFrame({
                "Nilai": [idx["J"], idx["FPC"]],
                "Rentang / Arah baik": [
                    "turun = baik",
                    f"1/c = {1/K:.4f} sampai 1; naik = baik",
                ],
            }, index=["J (fungsi objektif)", "FPC (Fuzzy Partition Coefficient)"])
            st.dataframe(df_idx.round(4), use_container_width=True)

            st.caption(
                "J mengukur total jarak berbobot keanggotaan ke pusat cluster; nilainya turun "
                "seiring iterasi dan tidak dipakai lintas c atau m. FPC mengukur ketegasan partisi: "
                "mendekati 1 berarti tiap data jelas milik satu cluster, mendekati 1/c berarti semua "
                "cluster menerima keanggotaan hampir sama (kabur)."
            )

        if st.button("Lihat Hasil Pengolahan"):
            st.switch_page("pages/Hasil_Pengolahan.py")