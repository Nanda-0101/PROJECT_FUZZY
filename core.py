"""Logika komputasi Fuzzy C-Means + PCA (tanpa dependensi Streamlit)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

FEATURES = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health",
            "absences", "G1", "G2", "G3"]

# (label Indonesia, min, max, default, deskripsi)
FEATURE_INFO = {
    "age":        ("Usia", 15, 22, 17, "Usia siswa (tahun), rentang 15-22."),
    "Medu":       ("Pendidikan ibu", 0, 4, 2, "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi."),
    "Fedu":       ("Pendidikan ayah", 0, 4, 2, "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi."),
    "traveltime": ("Waktu tempuh ke sekolah", 1, 4, 1, "1 = <15 mnt, 2 = 15-30 mnt, 3 = 30-60 mnt, 4 = >60 mnt."),
    "studytime":  ("Waktu belajar mingguan", 1, 4, 2, "1 = <2 jam, 2 = 2-5 jam, 3 = 5-10 jam, 4 = >10 jam."),
    "failures":   ("Jumlah kegagalan kelas", 0, 4, 0, "Jumlah kegagalan kelas sebelumnya."),
    "famrel":     ("Kualitas hubungan keluarga", 1, 5, 4, "1 = sangat buruk sampai 5 = sangat baik."),
    "freetime":   ("Waktu luang setelah sekolah", 1, 5, 3, "1 = sangat sedikit sampai 5 = sangat banyak."),
    "goout":      ("Frekuensi bergaul/keluar", 1, 5, 3, "1 = sangat jarang sampai 5 = sangat sering."),
    "Dalc":       ("Konsumsi alkohol hari kerja", 1, 5, 1, "1 = sangat rendah sampai 5 = sangat tinggi."),
    "Walc":       ("Konsumsi alkohol akhir pekan", 1, 5, 1, "1 = sangat rendah sampai 5 = sangat tinggi."),
    "health":     ("Status kesehatan", 1, 5, 3, "1 = sangat buruk sampai 5 = sangat baik."),
    "absences":   ("Jumlah absensi", 0, 93, 4, "Jumlah ketidakhadiran di sekolah (0-93)."),
    "G1":         ("Nilai periode 1", 0, 20, 10, "Nilai periode pertama (skala 0-20)."),
    "G2":         ("Nilai periode 2", 0, 20, 10, "Nilai periode kedua (skala 0-20)."),
    "G3":         ("Nilai akhir", 0, 20, 10, "Nilai akhir (skala 0-20)."),
}


# ----------------------------------------------------------------- data
def read_dataset(file_or_path):
    """Baca CSV dengan deteksi separator & pembersihan format dobel-quote.

    Mendukung:
      - CSV standar (sep , atau ;)
      - CSV dengan seluruh baris dibungkus kutip ganda  "GP;""F"";18;..."
      - Campuran quote di dalam field
    """
    # 1. Baca mentah sebagai teks
    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
    else:
        with open(file_or_path, "r", encoding="utf-8-sig", errors="replace") as f:
            raw = f.read()

    # 2. Deteksi format "baris dikutip penuh": setiap baris diawali & diakhiri "
    lines = raw.splitlines()
    quoted_full = (
        len(lines) > 1
        and lines[0].startswith('"')
        and lines[0].endswith('"')
        and '""' in lines[0]
    )
    if quoted_full:
        # Buang kutip pembuka & penutup setiap baris, lalu ubah "" -> "
        cleaned = []
        for ln in lines:
            ln = ln.strip()
            if ln.startswith('"') and ln.endswith('"') and len(ln) >= 2:
                ln = ln[1:-1]
            ln = ln.replace('""', '"')
            cleaned.append(ln)
        raw = "\n".join(cleaned)

    # 3. Coba berbagai kombinasi (sep, quotechar, quoting) sampai ketemu
    #    yang menghasilkan >= len(FEATURES) kolom & tidak semuanya NaN.
    candidates = [
        dict(sep=";", quotechar='"', quoting=0),   # standar CSV ; dengan quote
        dict(sep=";", quotechar='"', quoting=3),   # abaikan quote sama sekali
        dict(sep=",", quotechar='"', quoting=0),   # standar CSV ,
        dict(sep=",", quotechar='"', quoting=3),
        dict(sep=None, engine="python"),           # auto-detect
    ]
    from io import StringIO
    last_err = None
    for kw in candidates:
        try:
            df = pd.read_csv(StringIO(raw), **kw)
            df.columns = [str(c).strip().strip('"') for c in df.columns]
            # validasi: harus punya kolom fitur, dan kolom numerik utama tidak semua NaN
            if any(c in df.columns for c in FEATURES):
                # cek apakah kolom numerik inti benar-benar terbaca
                sample = [c for c in ("age", "Medu", "G1", "G2", "G3") if c in df.columns]
                if sample:
                    conv = pd.to_numeric(df[sample[0]], errors="coerce")
                    if conv.notna().sum() > 0:
                        return df
            # kalau tidak ada kolom fitur sama sekali, tetap return (biar validate_dataset yang komplain)
            if len(df.columns) >= len(FEATURES):
                return df
        except Exception as e:
            last_err = e
            continue

    # fallback: kalau semua gagal, raise error informatif
    raise ValueError(
        f"Tidak dapat membaca CSV. Format file tidak dikenali. "
        f"Pesan terakhir: {last_err}"
    )


def validate_dataset(df):
    """Kembalikan (df_bersih, daftar_pesan[(level,teks)], ringkasan_dict)."""
    msgs = []

    # ---- Cabang A: file tidak punya header yang benar ----
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        # Cek apakah masalahnya karena seluruh baris terbaca sebagai 1 kolom
        if len(df.columns) == 1:
            msgs.append((
                "error",
                "File CSV hanya terbaca sebagai 1 kolom. Kemungkinan format "
                "baris dikutip penuh (\"...\") atau separator tidak dikenali. "
                "Periksa file dengan pembaca CSV standar (Excel/LibreOffice) "
                "lalu simpan ulang sebagai CSV (;)."
            ))
        elif len(df.columns) < len(FEATURES):
            msgs.append((
                "error",
                f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}. "
                f"Kolom terbaca hanya {len(df.columns)} dari minimal {len(FEATURES)}."
            ))
        else:
            msgs.append((
                "error",
                f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}. "
                f"Kolom terbaca: {', '.join(map(str, df.columns[:10]))}..."
            ))
        return None, msgs, {}

    d = df.copy()

    # ---- Cabang B: konversi numerik per kolom, laporkan kolom mana yang gagal ----
    gagal_konversi = {}
    for c in FEATURES:
        before = d[c].notna().sum()
        d[c] = pd.to_numeric(d[c], errors="coerce")
        after = d[c].notna().sum()
        if after < before:
            gagal_konversi[c] = before - after
    if gagal_konversi:
        rinci = ", ".join(f"{c} ({n})" for c, n in gagal_konversi.items())
        msgs.append((
            "warning",
            f"Beberapa kolom berisi nilai non-numerik dan diubah menjadi kosong: {rinci}."
        ))

    n0 = len(d)
    bad_na = d[FEATURES].isna().any(axis=1)
    bad_rng = pd.Series(False, index=d.index)

    # ---- Cabang C: laporkan per kolom mana yang menyumbang NaN ----
    if bad_na.sum():
        rinci_na = {}
        for c in FEATURES:
            n_na = int(d[c].isna().sum())
            if n_na:
                rinci_na[c] = n_na
        if rinci_na:
            detail = ", ".join(f"{c}: {n}" for c, n in rinci_na.items())
            msgs.append((
                "warning",
                f"{int(bad_na.sum())} baris memiliki nilai kosong/non-numerik pada fitur: {detail}."
            ))
        else:
            msgs.append((
                "warning",
                f"{int(bad_na.sum())} baris memiliki nilai kosong/non-numerik pada fitur dan dibuang."
            ))

    # ---- Cabang D: cek rentang valid per kolom ----
    rinci_rng = {}
    for c in FEATURES:
        _, lo, hi, _, _ = FEATURE_INFO[c]
        mask = (d[c] < lo) | (d[c] > hi)
        n_bad = int(mask.sum())
        if n_bad:
            rinci_rng[c] = (n_bad, float(d.loc[mask, c].min()), float(d.loc[mask, c].max()))
        bad_rng |= mask
    if rinci_rng:
        detail = "; ".join(
            f"{c}: {n} baris (rentang ditemukan {mn:.1f}-{mx:.1f})"
            for c, (n, mn, mx) in rinci_rng.items()
        )
        msgs.append((
            "warning",
            f"{int((bad_rng & ~bad_na).sum())} baris memiliki nilai di luar rentang valid: {detail}."
        ))

    bad = bad_na | bad_rng
    clean = d.loc[~bad].reset_index(drop=True)

    # ---- Cabang E: terlalu sedikit data ----
    if len(clean) < 30:
        if len(clean) == 0:
            msgs.append((
                "error",
                f"Data valid 0 baris dari {n0} baris awal. Penyebab paling umum: "
                f"parser CSV salah membaca (lihat pesan di atas), atau seluruh "
                f"nilai berada di luar rentang valid."
            ))
        else:
            msgs.append((
                "error",
                f"Data valid hanya {len(clean)} baris; minimal 30 baris dibutuhkan."
            ))
        return None, msgs, {}

    # ---- Cabang F: fitur konstan ----
    zero_var = [c for c in FEATURES if clean[c].std() == 0]
    if zero_var:
        msgs.append((
            "warning",
            f"Fitur tanpa variasi (konstan): {', '.join(zero_var)}."
        ))

    # ---- Cabang G: sukses ----
    if not bad.any():
        msgs.append((
            "success",
            f"Seluruh {len(clean)} baris lolos validasi "
            f"(kolom lengkap, numerik, dan dalam rentang)."
        ))

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
    U = rng.random((N, c))
    U /= U.sum(1, keepdims=True)
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
    "Berprestasi Tinggi": "Pertahankan dengan program pengayaan atau akselerasi dan peran tutor sebaya.",
    "Menengah Atas": "Potensial naik ke kelompok tertinggi; dorong konsistensi belajar dan target nilai lebih tinggi.",
    "Menengah": "Perlu pendampingan belajar rutin dan monitoring perkembangan nilai antarperiode.",
    "Menengah Bawah": "Perlu bimbingan belajar tambahan dan pemantauan kehadiran serta kebiasaan belajar.",
    "Berprestasi Rendah (Berisiko)": "Prioritas intervensi: konseling akademik, remedial, kolaborasi dengan orang tua, dan pemantauan absensi.",
    "Berprestasi Menengah": "Perlu pendampingan belajar rutin dan monitoring perkembangan nilai antarperiode.",
}


def _traits(z):
    """Skor sifat gabungan (z-score) menjadi list (abs skor, frasa, skor)."""
    g = lambda cols: float(np.mean([z[c] for c in cols]))
    t = []
    s = g(["failures"])
    t.append((abs(s), "Riwayat Kegagalan Tinggi" if s > 0 else "Jarang Gagal Kelas", s))
    s = g(["absences"])
    t.append((abs(s), "Sering Absen" if s > 0 else "Kehadiran Baik", s))
    s = g(["Dalc", "Walc", "goout"])
    t.append((abs(s), "Pergaulan dan Konsumsi Alkohol Tinggi" if s > 0 else "Gaya Hidup Tenang", s))
    s = g(["Medu", "Fedu"])
    t.append((abs(s), "Orang Tua Berpendidikan Tinggi" if s > 0 else "Orang Tua Berpendidikan Rendah", s))
    s = g(["studytime"])
    t.append((abs(s), "Waktu Belajar Tinggi" if s > 0 else "Waktu Belajar Rendah", s))
    s = g(["age"])
    t.append((abs(s), "Usia Lebih Tua" if s > 0 else "Usia Lebih Muda", s))
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
        name = f"{tier} - {main[1]}" if main else tier
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


# ----------------------------------------------------------------- kelayakan PCA: KMO dan Bartlett (detail)
def _corr(df):
    """Matriks korelasi Pearson antar fitur."""
    return np.corrcoef(df[FEATURES].astype(float).values, rowvar=False)


def _anti_image(R):
    """Matriks anti-image = -(R^-1) / sqrt(diag(R^-1) outer diag(R^-1)).
    Diagonal-nya adalah nilai KMO per fitur (MSA)."""
    Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    A = -Rinv / d
    return A, Rinv


def _kmo_matrix(R, A):
    """Matriks korelasi parsial (off-diagonal) dan hitung KMO per fitur dan total.
    KMO_j = sum r_jk^2 / (sum r_jk^2 + sum a_jk^2)
    KMO   = sum sum r_jk^2 / (sum sum r_jk^2 + sum sum a_jk^2)
    """
    Rp = R.copy()
    Ap = A.copy()
    np.fill_diagonal(Rp, 0)
    np.fill_diagonal(Ap, 0)
    num_per = (Rp ** 2).sum(axis=0)
    den_per = num_per + (Ap ** 2).sum(axis=0)
    kmo_per = num_per / den_per
    num_tot = (Rp ** 2).sum()
    den_tot = num_tot + (Ap ** 2).sum()
    kmo_tot = float(num_tot / den_tot)
    return kmo_per, kmo_tot, Rp, Ap


def _kmo_category(v):
    if v >= 0.90:
        return ("marvelous", "sangat baik (marvelous)")
    if v >= 0.80:
        return ("meritorious", "berjasa (meritorious)")
    if v >= 0.70:
        return ("middling", "cukup baik (middling)")
    if v >= 0.60:
        return ("mediocre", "sedang (mediocre)")
    if v >= 0.50:
        return ("miserable", "buruk (miserable)")
    return ("unacceptable", "tidak dapat diterima")


def _bartlett(R, n, p):
    """Uji Bartlett's Test of Sphericity.
    H0: matriks korelasi = identitas (tidak ada korelasi antar fitur).
    chi2 = -(n - 1 - (2p + 5) / 6) * ln det(R)
    df   = p (p - 1) / 2
    """
    from scipy import stats
    sign, logdet = np.linalg.slogdet(R)
    if logdet == 0 or not np.isfinite(logdet):
        eig = np.linalg.eigvalsh(R)
        logdet = float(np.sum(np.log(np.clip(eig, 1e-12, None))))
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    dof = p * (p - 1) / 2
    pval = float(stats.chi2.sf(chi2, dof))
    return {"chi2": float(chi2), "dof": int(dof), "p": pval,
            "logdet": float(logdet), "n": int(n), "p_feat": int(p)}


def pca_suitability(df):
    """Uji kelayakan PCA secara menyeluruh dan transparan.

    Mengembalikan dict berisi semua komponen perhitungan sehingga dapat
    diaudit atau ditampilkan langkah demi langkah di UI.
    """
    X = df[FEATURES].astype(float)
    n, p = X.shape

    # 1. Korelasi
    R = _corr(df)

    # 2. Anti-image dan matriks korelasi parsial
    A, Rinv = _anti_image(R)

    # 3. KMO total dan per fitur
    kmo_per, kmo_tot, Rp, Ap = _kmo_matrix(R, A)
    kmo_cat_code, kmo_cat_lbl = _kmo_category(kmo_tot)

    # 4. Bartlett
    bart = _bartlett(R, n, p)
    bart_sig = bart["p"] < 0.05

    # 5. Ringkasan tambahan
    Rvals = R[np.triu_indices(p, 1)]
    mean_abs = float(np.abs(Rvals).mean())
    max_pair_idx = int(np.argmax(np.abs(Rvals)))
    triu_i, triu_j = np.triu_indices(p, 1)
    max_pair = (FEATURES[triu_i[max_pair_idx]], FEATURES[triu_j[max_pair_idx]],
                float(Rvals[max_pair_idx]))
    eigenvalues = np.linalg.eigvalsh(R)[::-1]
    n_eig_gt1 = int((eigenvalues > 1).sum())

    return {
        # input
        "n": int(n), "p": int(p),
        # korelasi dan anti-image (lengkap, untuk ditabel atau diplot)
        "R": pd.DataFrame(R, index=FEATURES, columns=FEATURES),
        "Rinv": pd.DataFrame(Rinv, index=FEATURES, columns=FEATURES),
        "A": pd.DataFrame(A, index=FEATURES, columns=FEATURES),
        "Rp": pd.DataFrame(Rp, index=FEATURES, columns=FEATURES),
        "Ap": pd.DataFrame(Ap, index=FEATURES, columns=FEATURES),
        # KMO
        "kmo": kmo_tot,
        "kmo_per": pd.Series(kmo_per, index=FEATURES),
        "kmo_num": float((Rp ** 2).sum()),
        "kmo_den": float((Rp ** 2).sum() + (Ap ** 2).sum()),
        "kmo_cat": kmo_cat_lbl,
        "kmo_cat_code": kmo_cat_code,
        # Bartlett
        "bartlett": bart,
        "bartlett_sig": bart_sig,
        # ringkasan
        "mean_abs_corr": mean_abs,
        "max_pair": max_pair,
        "eigenvalues": pd.Series(eigenvalues, index=[f"lambda{i+1}" for i in range(p)]),
        "n_eig_gt1": n_eig_gt1,
    }


# ----------------------------------------------------------------- pipeline
def run_pipeline(df, m=1.5, max_iter=500, tol=1e-5, n_init=5, var_thr=0.80,
                 k_min=2, k_max=8, k_mode="auto", k_manual=3, seed=42):
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
        rows.append({"k": k,
                     "Silhouette": silhouette_score(X_pca, r["labels"]) if len(set(r["labels"])) > 1 else np.nan,
                     "Partition Coefficient": partition_coefficient(r["U"]),
                     "Degenerat": deg})
    k_table = pd.DataFrame(rows).set_index("k")
    k_auto = int(k_table["Silhouette"].idxmax())
    K = k_auto if k_mode == "auto" else int(k_manual)

    res = fcm(X_pca, K, m, max_iter, tol, n_init, seed)
    # urutkan cluster dari nilai rata-rata G (G1, G2, G3) tertinggi ke terendah
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
    sil_samples = silhouette_samples(X_pca, labels) if uniq else np.zeros(len(labels))
    # empat komponen evaluasi: KMO dan Bartlett sebelum ekstraksi,
    # explained variance sesudah ekstraksi, Silhouette, dan Fuzzy Partition Coefficient
    metrics = {
        "Silhouette (ruang PCA)": sil_p,
        "Partition Coefficient": partition_coefficient(U),
        "Explained variance (komponen terpilih)": float(cum[n_comp - 1]),
        "Explained variance (PC1+PC2)": float(cum[1]),
    }
    suit = pca_suitability(df)

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
        "evr": evr, "cum": cum, "n_comp": n_comp,
        "loadings": pd.DataFrame(pca2.components_.T, index=FEATURES, columns=["PC1", "PC2"]),
        "k_table": k_table, "k_auto": k_auto, "K": K,
        "U": U, "centers": centers, "labels": labels, "hist": res["hist"],
        "cluster_info": info, "names": names, "metrics": metrics,
        "sil_samples": sil_samples, "suit": suit, "n_iter": res["n_iter"], "J": res["J"],
        "result_df": result_df,
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