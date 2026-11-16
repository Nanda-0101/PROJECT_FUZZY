"""Merangkum empat komponen evaluasi model.
1. KMO dan Bartlett sebelum ekstraksi PCA.
2. Explained variance setelah ekstraksi PCA.
3. Silhouette Score pada ruang PCA.
4. Fuzzy Partition Coefficient dari derajat keanggotaan FCM.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core import FEATURES, FEATURE_INFO
from ui import CMAP, alert, bs, need_result, setup


# Fungsi bantu kategori
def _kmo_category(value):
    """Kategori KMO menurut panduan umum Kaiser."""
    if value >= 0.8:
        return "Sangat baik"
    if value >= 0.7:
        return "Baik"
    if value >= 0.6:
        return "Cukup"
    if value >= 0.5:
        return "Kurang"
    return "Belum memadai"


def _kmo_color(value):
    if value >= 0.6:
        return "success"
    return "warning" if value >= 0.5 else "danger"


def _silhouette_category(value):
    if value < 0:
        return "Pemisahan lemah"
    if value < 0.25:
        return "Pemisahan rendah"
    if value < 0.5:
        return "Pemisahan sedang"
    if value < 0.7:
        return "Pemisahan baik"
    return "Pemisahan kuat"


def _silhouette_color(value):
    if np.isnan(value):
        return "secondary"
    if value < 0:
        return "danger"
    return "warning" if value < 0.25 else "success"


def _variance_color(value):
    if value >= 0.8:
        return "success"
    return "warning" if value >= 0.6 else "danger"


def _format_p_value(value):
    if value < 0.001:
        return "< 0,001"
    return f"{value:.4f}".replace(".", ",")


def _modified_pc(fpc, k):
    """Partition Coefficient ternormalisasi ke rentang 0 sampai 1."""
    return (k * fpc - 1) / (k - 1)


def _score_card(title, value, badge, color, note, icon):
    """Satu kartu ringkasan Bootstrap untuk bagian atas halaman."""
    return f"""<div class="col-6 col-lg mb-3"><div class="card h-100 shadow-sm bg-white text-dark border-0 border-start border-4 border-{color}">
<div class="card-body py-3"><div class="d-flex justify-content-between align-items-start">
<span class="text-muted small text-uppercase fw-semibold">{title}</span>
<i class="bi bi-{icon} text-{color}"></i></div>
<div class="fs-3 fw-bold">{value}</div>
<span class="badge text-bg-{color}">{badge}</span>
<div class="small text-muted mt-2">{note}</div></div></div></div>"""


# Tampilan awal
setup("Evaluasi")

st.header("Evaluasi Model")
st.caption(
    "Halaman ini merangkum kelayakan data untuk PCA, informasi varians setelah reduksi, "
    "dan kualitas hasil pengelompokan Fuzzy C-Means."
)
bs(
    """<div class="hero p-4 p-md-5 mb-3 shadow-sm">
<div class="row align-items-center"><div class="col-lg-9">
<span class="badge text-bg-light text-primary mb-2">DASHBOARD EVALUASI</span>
<h2 class="fw-bold mb-2">Baca kualitas model dari setiap tahap</h2>
<p class="mb-0 opacity-75">KMO dan Bartlett memeriksa kelayakan korelasi sebelum PCA.
Explained variance menilai informasi yang dipertahankan setelah PCA. Silhouette dan
Partition Coefficient mengevaluasi hasil FCM.</p></div>
<div class="col-lg-3 text-lg-end mt-3 mt-lg-0"><i class="bi bi-clipboard2-data-fill"
style="font-size:4rem"></i></div></div></div>""",
    205,
)

if not need_result():
    # Ambil hasil dari sesi
    result = st.session_state.res
    metrics = result["metrics"]
    suit = result["suit"]
    K = result["K"]
    names = result["names"]
    labels = result["labels"]
    evr, cum = result["evr"], result["cum"]
    n_comp = result["n_comp"]
    threshold = result["params"]["var_thr"]

    kmo = float(suit["kmo"])
    p_value = float(suit["p"])
    bartlett_ok = p_value < 0.05
    selected_variance = float(metrics["Explained variance (komponen terpilih)"])
    top2_variance = float(metrics["Explained variance (PC1+PC2)"])
    silhouette = float(metrics["Silhouette (ruang PCA)"])
    has_silhouette = not np.isnan(silhouette)
    fpc = float(metrics["Partition Coefficient"])
    fpc_floor = 1 / K
    mpc = _modified_pc(fpc, K)
    sil_values = np.asarray(result["sil_samples"], dtype=float)

    # Kartu ringkasan
    cards = "".join(
        [
            _score_card("KMO", f"{kmo:.3f}", _kmo_category(kmo), _kmo_color(kmo),
                        "Sebelum ekstraksi PCA", "check2-circle"),
            _score_card("Bartlett", _format_p_value(p_value),
                        "Signifikan" if bartlett_ok else "Tidak signifikan",
                        "success" if bartlett_ok else "danger",
                        f"p-value dengan chi-square {suit['chi2']:.1f}", "diagram-3"),
            _score_card("Explained variance", f"{selected_variance:.1%}",
                        f"{n_comp} dari {len(evr)} komponen",
                        _variance_color(selected_variance),
                        "Setelah ekstraksi PCA", "pie-chart"),
            _score_card("Silhouette", f"{silhouette:.3f}" if has_silhouette else "N/A",
                        _silhouette_category(silhouette) if has_silhouette else "Tidak tersedia",
                        _silhouette_color(silhouette),
                        "Pada ruang PCA", "bullseye"),
            _score_card("Partition Coefficient", f"{fpc:.3f}",
                        "Mendekati 1/k" if result["degenerate"] else "Keanggotaan tegas",
                        "danger" if result["degenerate"] else "success",
                        f"Batas bawah 1/k = {fpc_floor:.3f}", "layers"),
        ]
    )
    bs(f'<div class="row">{cards}</div>', 215)

    if result["degenerate"]:
        alert(
            "warning",
            "Partition Coefficient mendekati batas 1/k. Keanggotaan fuzzy sangat merata, "
            "sehingga keanggotaan cluster perlu ditafsirkan dengan hati-hati. "
            "Coba turunkan nilai fuzzifier m pada halaman Proses Pengolahan.",
            90,
        )

    # 1. KMO dan Bartlett
    st.subheader("1. Kelayakan data sebelum ekstraksi PCA")
    st.markdown(
        "Uji ini dihitung dari korelasi antarfitur pada data numerik sebelum PCA. "
        "KMO menilai kecukupan pola korelasi. Bartlett menguji apakah matriks korelasi "
        "berbeda dari matriks identitas."
    )

    kmo_col, bartlett_col = st.columns(2)
    with kmo_col:
        st.markdown("#### Kaiser-Meyer-Olkin")
        st.metric("KMO keseluruhan", f"{kmo:.3f}", _kmo_category(kmo), delta_color="off")
        if kmo >= 0.6:
            st.success(
                "Nilai KMO mencapai 0,6. Pola korelasi dinilai cukup untuk mendukung "
                "reduksi dimensi pada data ini."
            )
        else:
            st.warning(
                "Nilai KMO di bawah 0,6. Tinjau kecukupan sampel dan hubungan antarfitur "
                "sebelum mengandalkan hasil PCA."
            )
    with bartlett_col:
        st.markdown("#### Uji sphericity Bartlett")
        b1, b2 = st.columns(2)
        b1.metric("Chi-square", f"{suit['chi2']:.2f}", f"{suit['dof']} derajat bebas", delta_color="off")
        b2.metric("p-value", _format_p_value(p_value))
        if bartlett_ok:
            st.success(
                "p-value di bawah 0,05. Hipotesis matriks identitas ditolak, "
                "sehingga ada korelasi yang mendukung penggunaan PCA."
            )
        else:
            st.warning(
                "p-value tidak di bawah 0,05. Bukti korelasi antarfitur belum cukup kuat "
                "untuk menolak hipotesis matriks identitas."
            )

    kmo_per = suit["kmo_per"].sort_values()
    chart_col, table_col = st.columns([1.2, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        bar_colors = ["#54A24B" if v >= 0.6 else "#F2B134" if v >= 0.5 else "#E45756" for v in kmo_per]
        ax.barh(kmo_per.index, kmo_per.values, color=bar_colors)
        ax.axvline(0.6, color="#444444", linestyle="--", linewidth=1, label="Batas 0,6")
        ax.axvline(0.5, color="#999999", linestyle=":", linewidth=1, label="Batas 0,5")
        ax.set(title="KMO per fitur", xlabel="Nilai KMO", xlim=(0, 1))
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with table_col:
        kmo_table = kmo_per.rename("KMO").to_frame()
        kmo_table.insert(0, "Nama fitur", [FEATURE_INFO[f][0] for f in kmo_table.index])
        kmo_table["Kategori"] = kmo_table["KMO"].map(_kmo_category)
        st.dataframe(
            kmo_table.style.format({"KMO": "{:.3f}"}),
            use_container_width=True,
            height=400,
        )
    low_features = kmo_per[kmo_per < 0.5].index.tolist()
    if low_features:
        st.caption(
            f"Fitur dengan KMO di bawah 0,5 adalah {', '.join(low_features)}. "
            "Fitur ini kurang berbagi varians dengan fitur lain dan kontribusinya pada PCA terbatas."
        )
    else:
        st.caption("Seluruh fitur memiliki KMO minimal 0,5 sehingga tidak ada fitur yang menonjol bermasalah.")
    st.caption(
        f"Pengujian memakai {len(st.session_state.df)} baris dan {len(FEATURES)} fitur. "
        "Korelasi dihitung dari data asli, hasilnya sama dengan korelasi data terstandardisasi."
    )

    # 2. Explained variance
    st.divider()
    st.subheader("2. Informasi yang dipertahankan oleh PCA")
    st.markdown(
        f"PCA memilih **{n_comp} dari {len(evr)} komponen** untuk mencapai ambang kumulatif "
        f"{threshold:.0%}. Varians yang dipertahankan sebesar **{selected_variance:.1%}**, "
        f"sehingga sekitar **{1 - selected_variance:.1%}** varians tidak masuk ke ruang yang dipakai FCM. "
        f"Dua komponen pertama menjelaskan {top2_variance:.1%} dan dipakai untuk visualisasi."
    )
    variance_col, variance_table_col = st.columns([1.5, 1])
    with variance_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        numbers = np.arange(1, len(evr) + 1)
        ax.bar(numbers, evr, color="#4C78A8", alpha=0.82, label="Varians per komponen")
        ax.plot(numbers, cum, color="#F58518", marker="o", linewidth=2, label="Kumulatif")
        ax.axhline(threshold, color="#E45756", linestyle="--", label=f"Ambang {threshold:.0%}")
        ax.axvline(n_comp, color="#54A24B", linestyle=":", label=f"{n_comp} komponen terpilih")
        ax.set(title="Varians yang dijelaskan oleh komponen PCA", xlabel="Komponen utama",
               ylabel="Proporsi varians", ylim=(0, 1.03))
        ax.set_xticks(numbers)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with variance_table_col:
        variance_table = pd.DataFrame(
            {
                "Komponen": [f"PC{i}" for i in numbers],
                "Varians": evr,
                "Kumulatif": cum,
                "Dipakai FCM": ["Ya" if i <= n_comp else "Tidak" for i in numbers],
            }
        )
        st.dataframe(
            variance_table.style.format({"Varians": "{:.2%}", "Kumulatif": "{:.2%}"}),
            use_container_width=True,
            height=440,
            hide_index=True,
        )
    st.caption(
        "FCM dijalankan hanya pada komponen terpilih. Kurva kumulatif menunjukkan "
        "berapa banyak variasi data yang dijelaskan hingga tiap komponen."
    )

    # 3. Kualitas FCM
    st.divider()
    st.subheader("3. Kualitas pengelompokan Fuzzy C-Means")
    score_col, fpc_col = st.columns(2)
    with score_col:
        st.markdown("#### Silhouette Score")
        st.metric(
            "Skor pada ruang PCA",
            f"{silhouette:.3f}" if has_silhouette else "Tidak tersedia",
            _silhouette_category(silhouette) if has_silhouette else None,
            delta_color="off",
        )
        if not has_silhouette:
            st.warning("Silhouette tidak dapat dihitung karena hasil hanya memiliki satu label.")
        elif silhouette < 0:
            st.warning("Skor negatif menunjukkan banyak siswa lebih dekat ke kelompok lain daripada kelompoknya sendiri.")
        elif silhouette < 0.25:
            st.info("Pemisahan cluster masih rendah dan kelompok kemungkinan saling tumpang tindih.")
        else:
            st.success("Skor positif menunjukkan struktur cluster terdeteksi. Nilai lebih tinggi berarti pemisahan lebih jelas.")
        st.caption(
            "Silhouette dihitung dari label berkeanggotaan terbesar pada ruang PCA. "
            "Rentang teoritisnya dari -1 sampai 1 dan batas kategori hanya panduan umum."
        )
    with fpc_col:
        st.markdown("#### Fuzzy Partition Coefficient")
        f1, f2 = st.columns(2)
        f1.metric("Partition Coefficient", f"{fpc:.4f}")
        f2.metric("Ternormalisasi", f"{mpc:.4f}", help="Dihitung sebagai (k x FPC - 1) dibagi (k - 1) agar berada pada rentang 0 sampai 1.")
        st.progress(
            min(1.0, max(0.0, fpc)),
            text=f"Batas bawah 1/k untuk k = {K} adalah {fpc_floor:.4f}",
        )
        if result["degenerate"]:
            st.warning("Nilai mendekati 1/k. Keanggotaan siswa tersebar relatif merata di antara cluster.")
        else:
            st.info("Nilai di atas 1/k menunjukkan keanggotaan lebih terkonsentrasi. Nilai mendekati 1 menandakan partisi lebih tegas.")
        st.caption(
            "FPC adalah rata-rata kuadrat derajat keanggotaan. Nilainya berada antara 1/k dan 1, "
            "sehingga perbandingan antar jumlah cluster sebaiknya memakai nilai ternormalisasi."
        )

    if has_silhouette:
        hist_col, cluster_col = st.columns(2)
        with hist_col:
            st.markdown("#### Sebaran silhouette siswa")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.hist(sil_values, bins=24, color="#72B7B2", edgecolor="white")
            ax.axvline(0, color="#E45756", linestyle="--", label="Batas 0")
            ax.set(title="Silhouette per siswa", xlabel="Silhouette", ylabel="Jumlah siswa")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                f"Sebanyak {(sil_values < 0).mean():.1%} siswa memiliki silhouette negatif. "
                "Proporsi ini membantu melihat tumpang tindih pada tingkat individu."
            )
        with cluster_col:
            st.markdown("#### Silhouette per cluster")
            sil_df = pd.DataFrame({"Cluster": labels, "Silhouette": sil_values})
            grouped = sil_df.groupby("Cluster")["Silhouette"]
            per_cluster = pd.DataFrame(
                {
                    "Cluster": [f"C{j}" for j in range(K)],
                    "Nama": names,
                    "Siswa": grouped.size().reindex(range(K), fill_value=0).values,
                    "Rata-rata": grouped.mean().reindex(range(K)).values,
                    "Negatif": grouped.apply(lambda s: (s < 0).mean()).reindex(range(K)).values,
                }
            )
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.bar(per_cluster["Cluster"], per_cluster["Rata-rata"].fillna(0),
                   color=[CMAP(j) for j in range(K)])
            ax.axhline(silhouette, color="#444444", linestyle="--", label=f"Rata-rata total {silhouette:.3f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title="Rata-rata silhouette per cluster", xlabel="Cluster", ylabel="Silhouette")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(
                per_cluster.style.format({"Rata-rata": "{:.3f}", "Negatif": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

    # Perbandingan jumlah cluster
    st.markdown("#### Perbandingan jumlah cluster")
    k_table = result["k_table"].copy()
    plot_table = k_table.dropna(subset=["Silhouette"])
    k_chart_col, k_table_col = st.columns([1.4, 1])
    with k_chart_col:
        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_table.empty:
            ax.text(0.5, 0.5, "Skor tidak tersedia", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.plot(plot_table.index, plot_table["Silhouette"], marker="o", color="#4C78A8", label="Silhouette")
            ax.set(title="Silhouette dan Partition Coefficient untuk setiap k", xlabel="Jumlah cluster",
                   ylabel="Silhouette")
            ax2 = ax.twinx()
            ax2.plot(k_table.index, k_table["Partition Coefficient"], marker="s", color="#F58518",
                     label="Partition Coefficient")
            ax2.set_ylabel("Partition Coefficient")
            ax2.grid(False)
            ax.axvline(K, linestyle="--", color="#E45756", label=f"k terpilih = {K}")
            handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
            texts = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
            ax.legend(handles, texts, fontsize=8, loc="best")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with k_table_col:
        k_view = k_table.copy()
        k_view["Degenerat"] = k_view["Degenerat"].map({True: "Ya", False: "Tidak"})
        st.dataframe(
            k_view.style.format({"Silhouette": "{:.4f}", "Partition Coefficient": "{:.4f}"}),
            use_container_width=True,
        )
    st.caption(
        f"Mode pemilihan k adalah {result['params']['k_mode']}. "
        f"Silhouette tertinggi ada pada k = {result['k_auto']} dan k yang dipakai adalah {K}."
    )

    # Kesimpulan otomatis
    st.divider()
    st.subheader("Kesimpulan evaluasi")
    summary = [
        f"KMO sebesar <b>{kmo:.3f}</b> berkategori <b>{_kmo_category(kmo).lower()}</b>. "
        f"Uji Bartlett <b>{'signifikan' if bartlett_ok else 'tidak signifikan'}</b> dengan p-value {_format_p_value(p_value)}. "
        + ("Data layak direduksi dengan PCA." if kmo >= 0.6 and bartlett_ok
           else "Kelayakan PCA perlu dipertimbangkan kembali."),
        f"PCA mempertahankan <b>{selected_variance:.1%}</b> varians dengan <b>{n_comp}</b> komponen. "
        f"Dimensi berkurang dari {len(FEATURES)} menjadi {n_comp}.",
        (f"Silhouette sebesar <b>{silhouette:.3f}</b> menunjukkan <b>{_silhouette_category(silhouette).lower()}</b>."
         if has_silhouette else "Silhouette tidak dapat dihitung karena hanya ada satu label."),
        f"Partition Coefficient sebesar <b>{fpc:.3f}</b> dengan nilai ternormalisasi <b>{mpc:.3f}</b>. "
        + ("Keanggotaan hampir merata sehingga hasil perlu ditafsirkan hati-hati."
           if result["degenerate"] else "Keanggotaan fuzzy cukup terkonsentrasi pada cluster utama."),
    ]
    items = "".join(f"<li class='mb-2'>{line}</li>" for line in summary)
    bs(
        f'<div class="card bg-white text-dark shadow-sm border-0"><div class="card-body">'
        f'<ul class="mb-0">{items}</ul></div></div>',
        120 + 40 * len(summary),
    )

    # Panduan membaca
    bs(
        """<div class="row g-3">
<div class="col-md-6 col-lg-3"><div class="card h-100 border-primary shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-primary"><i class="bi bi-check2-circle me-1"></i>KMO</h6>
<p class="small mb-0">Nilai minimal 0,6 sering dipakai sebagai panduan kecukupan pola korelasi.</p></div></div></div>
<div class="col-md-6 col-lg-3"><div class="card h-100 border-success shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-success"><i class="bi bi-diagram-3 me-1"></i>Bartlett</h6>
<p class="small mb-0">p-value di bawah 0,05 mendukung penolakan matriks korelasi identitas.</p></div></div></div>
<div class="col-md-6 col-lg-3"><div class="card h-100 border-warning shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-warning"><i class="bi bi-bullseye me-1"></i>Silhouette</h6>
<p class="small mb-0">Mengukur pemisahan label tegas. Nilai makin dekat 1 berarti kelompok makin terpisah.</p></div></div></div>
<div class="col-md-6 col-lg-3"><div class="card h-100 border-danger shadow-sm bg-white text-dark">
<div class="card-body"><h6 class="text-danger"><i class="bi bi-layers me-1"></i>Partition Coefficient</h6>
<p class="small mb-0">Mengukur ketegasan keanggotaan fuzzy. Nilai makin dekat 1 berarti partisi makin tegas.</p></div></div></div>
</div>""",
        190,
    )