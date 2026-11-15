import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import core
from core import FEATURES, FEATURE_INFO
from ui import *

setup("Prediksi Siswa Baru")

st.header("Prediksi Siswa Baru")
st.caption("Menentukan cluster siswa yang tidak ada di dataset latih: data distandardisasi dengan scaler yang sama, "
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
                                              help=f"{desc} Rentang valid {lo}–{hi}.")
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
            ax.set(title="Posisi siswa baru pada PC1-PC2", xlabel="PC1", ylabel="PC2"); ax.legend(fontsize=7)
            st.pyplot(fig); plt.close(fig)

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
                        ax.scatter(zv[m_, 0], zv[m_, 1], s=40, color=CMAP(j), edgecolor="black", label=r["names"][j])
                    ax.set(title="Siswa baru pada PC1-PC2", xlabel="PC1", ylabel="PC2"); ax.legend(fontsize=7)
                    c2.pyplot(fig); plt.close(fig)
                    view = out.rename(columns={f"U{j}": f"Keanggotaan C{j}" for j in range(r["K"])})
                    st.dataframe(view.style.format({c: "{:.3f}" for c in view.columns if c.startswith("Keanggotaan")}),
                                 use_container_width=True, height=320)
                    st.download_button("Unduh hasil prediksi (CSV)", view.to_csv(index=False).encode("utf-8"),
                                       "prediksi_siswa_baru.csv", "text/csv")
            except Exception as e:
                alert("danger", f"File tidak dapat diproses: {e}")