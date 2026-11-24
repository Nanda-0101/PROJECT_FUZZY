import os
import time

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


# =====================================================================
# HELPER
# =====================================================================
def _nav_to_eksplorasi(key: str):
    """Tombol navigasi ke halaman Eksplorasi Data."""
    st.markdown("---")
    if st.button("Lanjut ke Eksplorasi Data", type="primary",
                 use_container_width=True, key=key):
        st.switch_page("pages/Eksplorasi_Data.py")


def _load_default_with_feedback():
    """Muat dataset bawaan + tampilkan pesan interaktif bertahap."""
    with st.spinner("Memuat dataset bawaan student-mat.csv ..."):
        time.sleep(0.4)
        try:
            load_default()
            df = st.session_state.get("df")
        except Exception as e:
            st.error(f"Gagal memuat dataset: {e}")
            return

    st.toast("Dataset bawaan berhasil dimuat", icon="✅")

    if df is not None:
        st.success(
            f"Dataset bawaan **student-mat.csv** berhasil dimuat. "
            f"Total **{len(df)} siswa** × **{len(FEATURES)} fitur** siap diproses."
        )

    st.balloons()
    st.info(
        "Langkah berikutnya: periksa kualitas data di halaman **Eksplorasi Data**, "
        "atau langsung ke **Proses Pengolahan** untuk menjalankan FCM + PCA."
    )


def _try_use_manual(df_manual: pd.DataFrame):
    """Validasi akhir dengan core.validate_dataset, lalu set sebagai data latih."""
    clean, msgs, info = core.validate_dataset(df_manual)
    alerts(msgs)
    if clean is None:
        alert("danger", "Dataset manual gagal divalidasi oleh core. Periksa pesan di atas.", 70)
        return

    # Validasi tambahan: setelah cleaning, pastikan masih ada variasi
    if clean[FEATURES].std(ddof=0).eq(0).all():
        alert(
            "danger",
            "❌ Setelah pembersihan, seluruh fitur menjadi konstan. "
            "Dataset tidak dapat dipakai untuk FCM + PCA.",
            70,
        )
        return

    # Validasi tambahan: minimal 30 baris unik
    if len(clean.drop_duplicates()) < 30:
        alert(
            "danger",
            f"❌ Setelah pembersihan, hanya **{len(clean.drop_duplicates())} baris unik** "
            f"dari {len(clean)} baris. Minimal **30 baris unik** agar FCM stabil.",
            70,
        )
        return

    set_data(clean, f"Input manual ({len(clean)} baris)")
    st.toast("Dataset manual aktif sebagai data latih.", icon="📥")
    st.rerun()


# =====================================================================
# TAB 1 — DATASET (UNGGAH / BAWAAN)
# =====================================================================
with t1:
    st.write(
        "Gunakan dataset bawaan atau unggah file CSV sendiri "
        "(separator `;` atau `,` terdeteksi otomatis). "
        "Kolom wajib: " + ", ".join(f"`{c}`" for c in FEATURES) + "."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if os.path.exists(DEFAULT_CSV) and st.button(
            "Gunakan dataset bawaan (student-mat.csv)",
            use_container_width=True,
        ):
            _load_default_with_feedback()
            st.rerun()

        if not os.path.exists(DEFAULT_CSV):
            st.warning("File `student-mat.csv` tidak ditemukan di folder aplikasi.")

    with c2:
        up = st.file_uploader("Unggah dataset (.csv)", type=["csv"])

    if up is not None:
        try:
            raw = core.read_dataset(up)
            clean, msgs, info = core.validate_dataset(raw)
            alerts(msgs)
            if clean is not None:
                st.caption(
                    f"{info['baris_awal']} baris terbaca, "
                    f"{info['baris_valid']} baris valid."
                )
                if st.button("Pakai dataset yang diunggah", type="primary"):
                    with st.spinner("Menyiapkan dataset unggahan ..."):
                        time.sleep(0.3)
                        set_data(clean, f"{up.name} (unggahan)")
                    st.toast("Dataset unggahan aktif", icon="📥")
                    st.rerun()
        except Exception as e:
            alert("danger", f"File tidak dapat dibaca: {e}")

    if st.session_state.get("df") is not None:
        d = st.session_state.df
        st.subheader("Data aktif")
        stat_cards(
            [
                ("Sumber", st.session_state.data_source, "primary"),
                ("Baris", len(d), "success"),
                ("Fitur dipakai", len(FEATURES), "warning"),
                ("Missing", int(d[FEATURES].isna().sum().sum()), "secondary"),
            ],
            120,
        )
        st.dataframe(d[FEATURES].head(50), use_container_width=True)
        st.markdown("**Statistik deskriptif**")
        st.dataframe(d[FEATURES].describe().T.round(3), use_container_width=True)

    _nav_to_eksplorasi("nav_eksp_from_dataset")


# =====================================================================
# TAB 2 — INPUT MANUAL SISWA (EDITOR DATASET LATIH)
# =====================================================================
with t2:
    st.write(
        "Bangun **dataset latih** secara manual dengan menambahkan satu atau beberapa siswa. "
        "Dataset manual ini akan dipakai oleh **FCM + PCA** untuk membentuk cluster. "
        "Minimal **30 baris unik** agar klasterisasi valid. "
        "Untuk prediksi siswa **baru** (inference), gunakan menu **Prediksi Siswa Baru**."
    )

    # -----------------------------------------------------------------
    # Inisialisasi dataset manual di session_state
    # -----------------------------------------------------------------
    if "manual_df" not in st.session_state:
        st.session_state["manual_df"] = pd.DataFrame(columns=FEATURES)

    manual_df = st.session_state["manual_df"]

    # -----------------------------------------------------------------
    # Input massal via CSV (template + upload)
    # -----------------------------------------------------------------
    st.markdown("### 📥 Input Massal via CSV")
    tpl = pd.DataFrame([{c: FEATURE_INFO[c][3] for c in FEATURES}])
    st.download_button(
        "⬇️ Unduh Template CSV",
        data=tpl.to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name="template_dataset_manual.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_tpl_manual",
    )
    up_manual = st.file_uploader(
        "Unggah CSV untuk ditambahkan ke dataset manual",
        type=["csv"],
        key="up_manual",
    )
    if up_manual is not None:
        try:
            raw = core.read_dataset(up_manual)
            clean, msgs, info = core.validate_dataset(raw)
            alerts(msgs)
            if clean is not None:
                if st.button("Tambahkan ke Dataset Manual", type="primary", key="btn_merge_manual"):
                    st.session_state["manual_df"] = pd.concat(
                        [st.session_state["manual_df"], clean[FEATURES]],
                        ignore_index=True,
                    )
                    st.toast(f"{len(clean)} baris ditambahkan.", icon="📥")
                    st.rerun()
        except Exception as e:
            alert("danger", f"File tidak dapat dibaca: {e}")

    # -----------------------------------------------------------------
    # Form tambah satu baris
    # -----------------------------------------------------------------
    st.markdown("### ➕ Tambah Satu Siswa")
    with st.form("manual_add", clear_on_submit=True):
        vals = {}
        cols = st.columns(4)
        for i, c in enumerate(FEATURES):
            lbl, lo, hi, de, desc = FEATURE_INFO[c]
            with cols[i % 4]:
                vals[c] = st.number_input(
                    f"{lbl} ({c})",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(de),
                    step=1.0,
                    help=f"{desc} Rentang valid {lo}–{hi}.",
                    key=f"inp_{c}",
                )
        add = st.form_submit_button("Tambah ke Dataset Manual", type="primary")

    if add:
        errs = core.validate_single(vals)
        if errs:
            for e in errs:
                alert("danger", e, 58)
        else:
            new_row = pd.DataFrame([vals])
            st.session_state["manual_df"] = pd.concat(
                [st.session_state["manual_df"], new_row], ignore_index=True
            )
            st.toast(f"Baris ditambahkan (total {len(st.session_state['manual_df'])}).", icon="✅")
            st.rerun()

    # -----------------------------------------------------------------
    # Tabel dataset manual + hapus baris
    # -----------------------------------------------------------------
    manual_df = st.session_state["manual_df"]
    st.markdown(f"### 📋 Dataset Manual ({len(manual_df)} baris)")

    if len(manual_df) == 0:
        alert("info", "Dataset manual masih kosong. Tambahkan minimal 30 baris unik.", 58)
    else:
        show = manual_df.copy()
        show.insert(0, "No", range(1, len(show) + 1))
        st.dataframe(show, use_container_width=True, height=300)

        # Peringatan dini: semua baris identik / fitur konstan
        n_unique_early = len(manual_df.drop_duplicates())
        zero_var_early = [c for c in FEATURES if manual_df[c].std(ddof=0) == 0]
        if n_unique_early == 1:
            alert(
                "danger",
                f"⚠️ Semua {len(manual_df)} baris identik. Dataset ini **tidak akan bisa** "
                f"dipakai untuk FCM + PCA. Ubah nilai beberapa baris.",
                70,
            )
        elif zero_var_early:
            st.caption(
                f"ℹ️ Fitur konstan saat ini: {', '.join(zero_var_early)}. "
                f"Boleh dilanjutkan, tetapi tidak berkontribusi pada PCA."
            )

        # Hapus baris tertentu
        c1, c2 = st.columns([2, 1])
        with c1:
            no_hapus = st.number_input(
                "Nomor baris yang ingin dihapus",
                min_value=1,
                max_value=len(manual_df),
                value=1,
                step=1,
                key="del_no",
            )
        with c2:
            if st.button("🗑️ Hapus Baris", use_container_width=True):
                st.session_state["manual_df"] = manual_df.drop(index=no_hapus - 1).reset_index(drop=True)
                st.toast(f"Baris {no_hapus} dihapus.", icon="🗑️")
                st.rerun()

        # Hapus semua
        if st.button("🗑️ Hapus Semua Data Manual", use_container_width=True):
            st.session_state["manual_df"] = pd.DataFrame(columns=FEATURES)
            st.toast("Dataset manual dikosongkan.", icon="🗑️")
            st.rerun()

        # Unduh CSV dataset manual
        st.download_button(
            "⬇️ Unduh Dataset Manual (CSV)",
            data=manual_df.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="dataset_manual.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_manual",
        )

        # -------------------------------------------------------------
        # Pakai dataset manual sebagai dataset latih
        # -------------------------------------------------------------
        st.markdown("---")
        n_min = 30
        n_rows = len(manual_df)

        if n_rows < n_min:
            alert(
                "warning",
                f"Dataset manual baru **{n_rows} baris**. "
                f"Minimal **{n_min} baris unik** agar FCM valid. "
                f"Tambahkan {n_min - n_rows} baris lagi.",
                70,
            )
        else:
            if st.button(
                f"✅ Pakai Dataset Manual ({n_rows} baris) sebagai Dataset Latih",
                type="primary",
                use_container_width=True,
                key="use_manual",
            ):
                # Validasi 1: semua baris identik?
                dup_all = manual_df.duplicated().all()
                n_unique = len(manual_df.drop_duplicates())
                n_total = len(manual_df)

                # Validasi 2: fitur konstan
                zero_var_cols = [c for c in FEATURES if manual_df[c].std(ddof=0) == 0]

                # Validasi 3: semua fitur hanya 1 nilai unik
                unique_per_feature = {c: manual_df[c].nunique() for c in FEATURES}
                all_single = all(v == 1 for v in unique_per_feature.values())

                # CABANG A: semua baris identik
                if dup_all or (n_unique == 1 and n_total >= 1):
                    alert(
                        "danger",
                        f"❌ Dataset manual ditolak: **semua {n_total} baris identik**. "
                        f"FCM & PCA membutuhkan variasi antar data. "
                        f"Ubah nilai minimal satu fitur pada beberapa baris, "
                        f"atau tambahkan data siswa dengan profil berbeda.",
                        90,
                    )

                # CABANG B: semua fitur konstan
                elif all_single:
                    alert(
                        "danger",
                        "❌ Dataset manual ditolak: **tidak ada variasi sama sekali** "
                        "di seluruh fitur (semua kolom bernilai konstan). "
                        "Isi data dengan nilai yang berbeda antar siswa.",
                        90,
                    )

                # CABANG C: ada fitur konstan
                elif zero_var_cols:
                    if len(zero_var_cols) == len(FEATURES):
                        alert(
                            "danger",
                            "❌ Dataset manual ditolak: **seluruh fitur bernilai konstan**. "
                            "Tidak ada informasi yang bisa dipelajari FCM/PCA.",
                            90,
                        )
                    else:
                        st.warning(
                            f"⚠️ Fitur berikut bernilai konstan (std = 0) dan tidak "
                            f"berkontribusi pada PCA: **{', '.join(zero_var_cols)}**. "
                            f"Dataset tetap diproses, tetapi pertimbangkan memberi variasi."
                        )
                        _try_use_manual(manual_df)

                # CABANG D: baris unik < 30
                elif n_unique < n_min:
                    alert(
                        "danger",
                        f"❌ Dataset manual ditolak: hanya **{n_unique} baris unik** "
                        f"dari {n_total} baris. Minimal **{n_min} baris unik** "
                        f"agar FCM stabil. Hapus duplikat atau tambah data berbeda.",
                        90,
                    )

                # CABANG E: lolos semua validasi
                else:
                    _try_use_manual(manual_df)

    # -----------------------------------------------------------------
    # Info: dataset latih aktif saat ini
    # -----------------------------------------------------------------
    if st.session_state.get("df") is not None:
        st.markdown("---")
        st.caption(
            f"Dataset latih aktif: **{st.session_state.data_source}** "
            f"({len(st.session_state.df)} baris)."
        )

    _nav_to_eksplorasi("nav_eksp_from_manual")


# =====================================================================
# TAB 3 — PENJELASAN VARIABEL
# =====================================================================
with t3:
    rows = [
        {"Variabel": c, "Nama": v[0], "Rentang": f"{v[1]}–{v[2]}", "Penjelasan": v[4]}
        for c, v in FEATURE_INFO.items()
    ]
    bs_table(pd.DataFrame(rows), 640, index=False)
    alert(
        "info",
        "Atribut kategorikal (school, sex, Mjob, dll.) tidak dipakai karena "
        "FCM dan PCA berbasis jarak Euclidean pada data numerik.",
        75,
    )

    _nav_to_eksplorasi("nav_eksp_from_variabel")