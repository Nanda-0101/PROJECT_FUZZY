"""Logika komputasi Fuzzy C-Means + PCA (tanpa dependensi Streamlit)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score)

FEATURES = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health",
            "absences", "G1", "G2", "G3"]

# (label Indonesia, min, max, default, deskripsi)
FEATURE_INFO = {
    "age":        ("Usia", 15, 22, 17, "Usia siswa (tahun), rentang 15–22."),
    "Medu":       ("Pendidikan ibu", 0, 4, 2, "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi."),
    "Fedu":       ("Pendidikan ayah", 0, 4, 2, "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi."),
    "traveltime": ("Waktu tempuh ke sekolah", 1, 4, 1, "1 = <15 mnt, 2 = 15–30 mnt, 3 = 30–60 mnt, 4 = >60 mnt."),
    "studytime":  ("Waktu belajar mingguan", 1, 4, 2, "1 = <2 jam, 2 = 2–5 jam, 3 = 5–10 jam, 4 = >10 jam."),
    "failures":   ("Jumlah kegagalan kelas", 0, 4, 0, "Jumlah kegagalan kelas sebelumnya (n jika 1≤n<3, selain itu 3–4)."),
    "famrel":     ("Kualitas hubungan keluarga", 1, 5, 4, "1 = sangat buruk … 5 = sangat baik."),
    "freetime":   ("Waktu luang setelah sekolah", 1, 5, 3, "1 = sangat sedikit … 5 = sangat banyak."),
    "goout":      ("Frekuensi bergaul/keluar", 1, 5, 3, "1 = sangat jarang … 5 = sangat sering."),
    "Dalc":       ("Konsumsi alkohol hari kerja", 1, 5, 1, "1 = sangat rendah … 5 = sangat tinggi."),
    "Walc":       ("Konsumsi alkohol akhir pekan", 1, 5, 1, "1 = sangat rendah … 5 = sangat tinggi."),
    "health":     ("Status kesehatan", 1, 5, 3, "1 = sangat buruk … 5 = sangat baik."),
    "absences":   ("Jumlah absensi", 0, 93, 4, "Jumlah ketidakhadiran di sekolah (0–93)."),
    "G1":         ("Nilai periode 1", 0, 20, 10, "Nilai periode pertama (skala 0–20)."),
    "G2":         ("Nilai periode 2", 0, 20, 10, "Nilai periode kedua (skala 0–20)."),
    "G3":         ("Nilai akhir", 0, 20, 10, "Nilai akhir (skala 0–20)."),
}


# ----------------------------------------------------------------- data
def read_dataset(file_or_path):
    """Baca CSV; separator (; atau ,) dideteksi otomatis."""
    df = pd.read_csv(file_or_path, sep=None, engine="python")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def validate_dataset(df):
    """Kembalikan (df_bersih, daftar_pesan[(level,teks)], ringkasan_dict)."""
    msgs = []
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        msgs.append(("error", f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}."))
        return None, msgs, {}
    d = df.copy()
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    n0 = len(d)
    bad_na = d[FEATURES].isna().any(axis=1)
    bad_rng = pd.Series(False, index=d.index)
    for c in FEATURES:
        _, lo, hi, _, _ = FEATURE_INFO[c]
        bad_rng |= (d[c] < lo) | (d[c] > hi)
    bad = bad_na | bad_rng
    if bad_na.sum():
        msgs.append(("warning", f"{int(bad_na.sum())} baris memiliki nilai kosong/non-numerik pada fitur dan dibuang."))
    if (bad_rng & ~bad_na).sum():
        msgs.append(("warning", f"{int((bad_rng & ~bad_na).sum())} baris memiliki nilai di luar rentang valid dan dibuang."))
    clean = d.loc[~bad].reset_index(drop=True)
    if len(clean) < 30:
        msgs.append(("error", f"Data valid hanya {len(clean)} baris; minimal 30 baris dibutuhkan."))
        return None, msgs, {}
    zero_var = [c for c in FEATURES if clean[c].std() == 0]
    if zero_var:
        msgs.append(("warning", f"Fitur tanpa variasi (konstan): {', '.join(zero_var)}."))
    if not bad.any():
        msgs.append(("success", "Seluruh baris lolos validasi (kolom lengkap, numerik, dan dalam rentang)."))
    info = {"baris_awal": n0, "baris_valid": len(clean), "fitur": len(FEATURES)}
    return clean, msgs, info


def validate_single(values):
    errs = []
    for c, v in values.items():
        lbl, lo, hi, _, _ = FEATURE_INFO[c]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errs.append(f"{lbl} ({c}) kosong.")
        elif not (lo <= v <= hi):
            errs.append(f"{lbl} ({c}) harus di antara {lo} dan {hi}.")
    return errs


# ----------------------------------------------------------------- FCM
def _dist(X, V):
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ V.T + (V ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(d2, 1e-12))


def memberships(X, V, m):
    D = _dist(X, V)
    inv = D ** (-2.0 / (m - 1))
    return inv / inv.sum(1, keepdims=True)


def fcm_single(X, c, m, max_iter, tol, rng):
    N = X.shape[0]
    U = rng.random((N, c)); U /= U.sum(1, keepdims=True)
    hist = []
    for it in range(1, max_iter + 1):
        U_old = U
        Um = U ** m
        V = (Um.T @ X) / Um.sum(0)[:, None]
        D = _dist(X, V)
        inv = D ** (-2.0 / (m - 1))
        U = inv / inv.sum(1, keepdims=True)
        hist.append(float(((U ** m) * D ** 2).sum()))
        if np.abs(U - U_old).max() < tol:
            break
    J = float(((U ** m) * _dist(X, V) ** 2).sum())
    return U, V, J, it, hist


def fcm(X, c, m=1.5, max_iter=500, tol=1e-5, n_init=5, seed=42):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        r = fcm_single(X, c, m, max_iter, tol, rng)
        if best is None or r[2] < best[2]:
            best = r
    U, V, J, n_it, hist = best
    return {"U": U, "centers": V, "J": J, "n_iter": n_it, "hist": hist, "labels": U.argmax(1)}


def partition_coefficient(U):
    return float((U ** 2).sum() / U.shape[0])


def partition_entropy(U):
    return float(-(U * np.log(np.clip(U, 1e-12, 1))).sum() / U.shape[0])


def is_degenerate(U, tol=1e-3):
    return partition_coefficient(U) < 1.0 / U.shape[1] + tol


# ----------------------------------------------------------------- penamaan cluster
def _tier_names(k):
    if k == 2:
        return ["Berprestasi Tinggi", "Berprestasi Rendah (Berisiko)"]
    if k == 3:
        return ["Berprestasi Tinggi", "Berprestasi Menengah", "Berprestasi Rendah (Berisiko)"]
    if k == 4:
        return ["Berprestasi Tinggi", "Menengah Atas", "Menengah Bawah", "Berprestasi Rendah (Berisiko)"]
    return (["Berprestasi Tinggi", "Menengah Atas"] + ["Menengah"] * (k - 4)
            + ["Menengah Bawah", "Berprestasi Rendah (Berisiko)"])


TIER_ADVICE = {
    "Berprestasi Tinggi": "Pertahankan dengan program pengayaan/akselerasi dan peran tutor sebaya.",
    "Menengah Atas": "Potensial naik ke kelompok tertinggi; dorong konsistensi belajar dan target nilai lebih tinggi.",
    "Menengah": "Perlu pendampingan belajar rutin dan monitoring perkembangan nilai antarperiode.",
    "Menengah Bawah": "Perlu bimbingan belajar tambahan dan pemantauan kehadiran serta kebiasaan belajar.",
    "Berprestasi Rendah (Berisiko)": "Prioritas intervensi: konseling akademik, remedial, kolaborasi dengan orang tua, dan pemantauan absensi.",
    "Berprestasi Menengah": "Perlu pendampingan belajar rutin dan monitoring perkembangan nilai antarperiode.",
}


def _traits(z):
    """Skor sifat gabungan (z-score) -> list (|skor|, frasa)."""
    g = lambda cols: float(np.mean([z[c] for c in cols]))
    t = []
    s = g(["failures"]);               t.append((abs(s), "Riwayat Kegagalan Tinggi" if s > 0 else "Jarang Gagal Kelas", s))
    s = g(["absences"]);               t.append((abs(s), "Sering Absen" if s > 0 else "Kehadiran Baik", s))
    s = g(["Dalc", "Walc", "goout"]);  t.append((abs(s), "Pergaulan & Konsumsi Alkohol Tinggi" if s > 0 else "Gaya Hidup Tenang", s))
    s = g(["Medu", "Fedu"]);           t.append((abs(s), "Orang Tua Berpendidikan Tinggi" if s > 0 else "Orang Tua Berpendidikan Rendah", s))
    s = g(["studytime"]);              t.append((abs(s), "Waktu Belajar Tinggi" if s > 0 else "Waktu Belajar Rendah", s))
    s = g(["age"]);                    t.append((abs(s), "Usia Lebih Tua" if s > 0 else "Usia Lebih Muda", s))
    return t


def name_clusters(df_feat, labels, k):
    """Beri nama bermakna pada tiap cluster berdasarkan profil fitur asli.
    Cluster diasumsikan sudah terurut dari nilai tertinggi ke terendah."""
    mu, sd = df_feat.mean(), df_feat.std().replace(0, 1)
    tiers = _tier_names(k)
    out = []
    for j in range(k):
        sub = df_feat[labels == j]
        cm = sub.mean()
        z = ((cm - mu) / sd).to_dict()
        traits = sorted(_traits(z), key=lambda x: -x[0])
        main = traits[0] if traits and traits[0][0] >= 0.4 else None
        tier = tiers[j]
        name = f"{tier} – {main[1]}" if main else tier
        feats = sorted([(abs(v), c, v) for c, v in z.items()], reverse=True)[:4]
        ciri = []
        for _, c, v in feats:
            if abs(v) < 0.25:
                continue
            ciri.append(f"{FEATURE_INFO[c][0]} {'lebih tinggi' if v > 0 else 'lebih rendah'} "
                        f"(rata-rata {cm[c]:.2f} vs {mu[c]:.2f})")
        desc = (f"{len(sub)} siswa ({len(sub) / len(df_feat):.1%}). Rata-rata nilai akhir G3 = {cm['G3']:.2f} "
                f"(rata-rata seluruh siswa {mu['G3']:.2f}). "
                + ("Ciri menonjol: " + "; ".join(ciri) + "." if ciri else "Profil mendekati rata-rata seluruh siswa."))
        out.append({"id": j, "nama": name, "tier": tier, "deskripsi": desc,
                    "saran": TIER_ADVICE.get(tier, ""), "jumlah": int(len(sub))})
    # pastikan nama unik
    seen = {}
    for o in out:
        seen[o["nama"]] = seen.get(o["nama"], 0) + 1
    cnt = {}
    for o in out:
        if seen[o["nama"]] > 1:
            cnt[o["nama"]] = cnt.get(o["nama"], 0) + 1
            o["nama"] += f" (Kelompok {chr(64 + cnt[o['nama']])})"
    return out


# ----------------------------------------------------------------- pipeline
def run_pipeline(df, m=1.5, max_iter=500, tol=1e-5, n_init=5, var_thr=0.80,
                 k_min=2, k_max=8, k_mode="auto", k_manual=3, seed=42, n_stab=10):
    X_raw = df[FEATURES].astype(float)
    scaler = StandardScaler().fit(X_raw.values)
    X = scaler.transform(X_raw.values)

    pca_full = PCA(random_state=seed).fit(X)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_comp = int(np.argmax(cum >= var_thr) + 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(X)
    X_pca = pca.transform(X)
    pca2 = PCA(n_components=2, random_state=seed).fit(X)
    X_vis = pca2.transform(X)

    k_max = min(k_max, len(df) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        r = fcm(X_pca, k, m, max_iter, tol, n_init, seed)
        deg = is_degenerate(r["U"])
        rows.append({"k": k, "Silhouette": silhouette_score(X_pca, r["labels"]) if len(set(r["labels"])) > 1 else np.nan,
                     "Partition Coefficient": partition_coefficient(r["U"]),
                     "Partition Entropy": partition_entropy(r["U"]),
                     "Davies-Bouldin": davies_bouldin_score(X_pca, r["labels"]) if len(set(r["labels"])) > 1 else np.nan,
                     "J": r["J"], "Degenerat": deg})
    k_table = pd.DataFrame(rows).set_index("k")
    k_auto = int(k_table["Silhouette"].idxmax())
    K = k_auto if k_mode == "auto" else int(k_manual)

    res = fcm(X_pca, K, m, max_iter, tol, n_init, seed)
    # urutkan cluster dari nilai rata-rata G (G1,G2,G3) tertinggi -> terendah
    gmean = np.array([df.loc[res["labels"] == j, ["G1", "G2", "G3"]].values.mean()
                      if (res["labels"] == j).any() else -1 for j in range(K)])
    order = np.argsort(-gmean)
    U = res["U"][:, order]
    centers = res["centers"][order]
    labels = U.argmax(1)

    info = name_clusters(df[FEATURES], labels, K)
    names = [c["nama"] for c in info]

    # evaluasi
    uniq = len(set(labels)) > 1
    sil_p = silhouette_score(X_pca, labels) if uniq else np.nan
    sil_o = silhouette_score(X, labels) if uniq else np.nan
    sil_samples = silhouette_samples(X_pca, labels) if uniq else np.zeros(len(labels))
    metrics = {
        "Silhouette (ruang PCA)": sil_p,
        "Silhouette (ruang fitur asli)": sil_o,
        "Partition Coefficient": partition_coefficient(U),
        "Partition Entropy": partition_entropy(U),
        "Davies-Bouldin": davies_bouldin_score(X_pca, labels) if uniq else np.nan,
        "Calinski-Harabasz": calinski_harabasz_score(X_pca, labels) if uniq else np.nan,
        "Explained variance (komponen terpilih)": float(cum[n_comp - 1]),
        "Explained variance (PC1+PC2)": float(cum[1]),
        "Fungsi objektif J": res["J"],
        "Iterasi konvergen": res["n_iter"],
    }
    # stabilitas: ARI antar seed berbeda
    stab = []
    for s in range(n_stab):
        r2 = fcm(X_pca, K, m, max_iter, tol, max(1, n_init // 2), seed + 1000 + s)
        stab.append(adjusted_rand_score(labels, r2["labels"]))

    maxu = U.max(1)
    result_df = df[FEATURES].copy()
    result_df.insert(0, "Siswa", np.arange(1, len(df) + 1))
    result_df["Cluster"] = labels
    result_df["Nama Cluster"] = [names[l] for l in labels]
    for j in range(K):
        result_df[f"U{j}"] = U[:, j]
    result_df["Keanggotaan Tertinggi"] = maxu

    return {
        "params": dict(m=m, max_iter=max_iter, tol=tol, n_init=n_init, var_thr=var_thr,
                       k_min=k_min, k_max=k_max, k_mode=k_mode, seed=seed),
        "scaler": scaler, "pca": pca, "pca2": pca2, "X": X, "X_pca": X_pca, "X_vis": X_vis,
        "evr": evr, "cum": cum, "n_comp": n_comp, "loadings": pd.DataFrame(
            pca2.components_.T, index=FEATURES, columns=["PC1", "PC2"]),
        "k_table": k_table, "k_auto": k_auto, "K": K,
        "U": U, "centers": centers, "labels": labels, "hist": res["hist"],
        "cluster_info": info, "names": names, "metrics": metrics,
        "sil_samples": sil_samples, "stab": np.array(stab), "result_df": result_df,
        "degenerate": is_degenerate(U),
        "profile": df[FEATURES].assign(Cluster=labels).groupby("Cluster").mean(),
    }


def predict_one(res, values, m=None):
    """Hitung keanggotaan fuzzy satu siswa baru terhadap pusat cluster."""
    m = m or res["params"]["m"]
    x = np.array([[float(values[c]) for c in FEATURES]])
    z = res["scaler"].transform(x)
    zp = res["pca"].transform(z)
    u = memberships(zp, res["centers"], m)[0]
    zv = res["pca2"].transform(z)[0]
    return u, zv


# ----------------------------------------------------------------- tambahan: EDA & prediksi massal
def pca_suitability(df):
    """Uji kelayakan PCA: Bartlett sphericity & KMO. Mengembalikan dict."""
    from scipy import stats
    X = df[FEATURES].astype(float)
    n, p = X.shape
    R = np.corrcoef(X.values, rowvar=False)
    sign, logdet = np.linalg.slogdet(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    dof = p * (p - 1) / 2
    pval = float(stats.chi2.sf(chi2, dof))
    Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    partial = -Rinv / d
    np.fill_diagonal(R, 0); np.fill_diagonal(partial, 0)
    kmo_per = (R ** 2).sum(0) / ((R ** 2).sum(0) + (partial ** 2).sum(0))
    kmo = float((R ** 2).sum() / ((R ** 2).sum() + (partial ** 2).sum()))
    return {"chi2": float(chi2), "dof": int(dof), "p": pval, "kmo": kmo,
            "kmo_per": pd.Series(kmo_per, index=FEATURES)}


def predict_batch(res, df):
    """Tentukan cluster untuk banyak siswa baru (df sudah tervalidasi)."""
    z = res["scaler"].transform(df[FEATURES].astype(float).values)
    U = memberships(res["pca"].transform(z), res["centers"], res["params"]["m"])
    labels = U.argmax(1)
    out = df[FEATURES].copy().reset_index(drop=True)
    out.insert(0, "Siswa", np.arange(1, len(out) + 1))
    out["Cluster"] = labels
    out["Nama Cluster"] = [res["names"][l] for l in labels]
    for j in range(U.shape[1]):
        out[f"U{j}"] = U[:, j]
    out["Keanggotaan Tertinggi"] = U.max(1)
    return out
