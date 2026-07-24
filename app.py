"""
=================================================================
  QUYU KAROTAJ DİAQRAMLARININ KOMPLEKS PETROFİZİKİ İNTERPRETASİYASI
  MODULYAR / XƏTAYA DAVAMLI + İNTERAKTİV (Plotly) VERSİYA
  ----------------------------------------------------------------
  YENİLİKLƏR (v5):
    1. Dinamik matrisa seçimi — Qumdaşı / Əhəngdaşı / Dolomit / Fərdi
       Δt_ma:  181.5 / 155.8 / 141.0 µs/m   (ρ_ma: 2.65 / 2.71 / 2.87)
    2. Net Pay üçün 3 dinamik cut-off sürgüsü (Cgil, Km, Knq)
    3. Plotly ilə interaktiv treklər — hover, zoom, pan
       (plotly quraşdırılmayıbsa avtomatik matplotlib-ə keçir)

  ƏSAS PRİNSİP: proqram HEÇ BİR əyrinin olmamasından dayanmır.
  Yalnız DEPTH məcburidir.
      Cgil:  GR (ΔIγ) → SP (αSP) → sabit
      Km:    DT (Wyllie) → RHOB → NPHI → NPHI+RHOB → sabit
      Ksu:   yalnız Km və Rt olduqda (Arçi-Daxnov)
  ----------------------------------------------------------------
      pip install streamlit matplotlib numpy pandas plotly lasio
      streamlit run app.py
=================================================================
"""

import io
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

try:
    import lasio
    LASIO_VAR = True
except ImportError:
    LASIO_VAR = False

try:
    import cv2
    CV2_VAR = True
except ImportError:
    CV2_VAR = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_VAR = True
except ImportError:
    PLOTLY_VAR = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =================================================================
# 0) SABİTLƏR
# =================================================================

# --- MATRİSA (süxur skeleti) kataloqu ---
#     dt: (µs/ft, µs/m) — interval müddəti Δt_ma
#     rho: skelet sıxlığı ρ_ma (q/sm³)
MATRISA = {
    "Qumdaşı":        {"dt": (55.3, 181.5), "rho": 2.65},
    "Əhəngdaşı":      {"dt": (47.5, 155.8), "rho": 2.71},
    "Dolomit":        {"dt": (43.0, 141.0), "rho": 2.87},
    "Fərdi (Custom)": {"dt": (55.0, 180.0), "rho": 2.65},
}

# Maye: Δt_f (µs/ft, µs/m) və ρ_f (q/sm³)
MAYE = {
    "Şirin su / lil məhlulu": {"dt": (189.0, 620.0), "rho": 1.00},
    "Duzlu su (şoran)":       {"dt": (185.0, 607.0), "rho": 1.10},
    "Neft":                   {"dt": (238.0, 781.0), "rho": 0.85},
}

GIL_DT = {"µs/ft": 100.0, "µs/m": 328.0}      # gil üçün tipik Δt
GIL_RHO = 2.45                                 # gil üçün tipik sıxlıq

# LAS-da avtomatik axtarış üçün standart mnemonikalar
ADLAR = {
    "DEPTH": ["DEPT", "DEPTH", "MD", "TVD", "DEPTH_M"],
    "GR":    ["GR", "GRD", "SGR", "CGR", "GAMMA", "GRC", "GRR"],
    "SP":    ["SP", "PS", "SPC", "SSP"],
    "CALI":  ["CALI", "CAL", "KV", "CALS", "DQ", "CALX"],
    "RT":    ["RT", "ILD", "LLD", "RESD", "RD", "RES", "RILD", "AT90", "LL7", "IK"],
    "DT":    ["DT", "DTC", "AC", "SONIC", "DT24", "DTCO"],
    "RHOB":  ["RHOB", "RHOZ", "DEN", "GGKP", "ZDEN", "RHO"],
    "NPHI":  ["NPHI", "NPOR", "TNPH", "NGK", "NEUT", "CNL"],
    # Rəqəmsallaşdırılmış (şəkildən oxunmuş) əyrilər
    "NKT":   ["NKT", "NNK", "NNKT"],
    "W":     ["W", "WPCT", "W_PCT"],
    "MZ":    ["MZ", "MIKROZOND", "MLL"],
}

ETIKET = {
    "GR": "GR (Qamma)", "SP": "SP (Quyu potensialı)", "CALI": "CALI (Kavernomer)",
    "RT": "Rt (Xüsusi müqavimət)", "DT": "DT (Akustik)",
    "RHOB": "RHOB (Sıxlıq)", "NPHI": "NPHI (Neytron)",
    "NKT": "NKT (Neytron karotajı)", "W": "W (%)", "MZ": "MZ (Mikrozond)",
}

OPSIONAL = ["GR", "SP", "CALI", "RT", "DT", "RHOB", "NPHI", "NKT", "W", "MZ"]
YOX = "(yoxdur)"

# Plotly / matplotlib üçün ortaq rənglər
RENG = {
    "GR": "#111111", "SP": "#8A2BE2", "CALI": "#2E8B57", "RT": "#B22222",
    "NKT": "#00008B", "W": "#00CED1", "MZ": "#228B22",
    "CGIL": "#8B4513", "KM": "#4169E1", "KSU": "#1E90FF", "KNQ": "#228B22",
    "QUM": "#FFD700", "GIL": "#808080", "NETPAY": "#228B22",
}


# =================================================================
# 1) LAS FAYLININ OXUNMASI
# =================================================================

def sade_las_parser(metn: str) -> pd.DataFrame:
    """lasio olmadan LAS 2.0 (WRAP = NO) faylını oxuyan minimal parser."""
    sutunlar, data_setirleri = [], []
    bolme, null_deyer = None, -999.25

    for setir in metn.splitlines():
        s = setir.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            bolme = s[1].upper()
            continue
        if bolme == "W" and s.upper().startswith("NULL"):
            tap = re.search(r"[-+]?\d*\.?\d+", s.split(":")[0].split(".", 1)[-1])
            if tap:
                null_deyer = float(tap.group())
            continue
        if bolme == "C":
            ad = s.split(".")[0].strip()
            if ad:
                sutunlar.append(ad.upper())
            continue
        if bolme == "A":
            data_setirleri.append(s.split())

    if not sutunlar or not data_setirleri:
        raise ValueError("LAS faylında ~C (əyrilər) və ya ~A (data) bölməsi tapılmadı.")

    en = min(len(sutunlar), len(data_setirleri[0]))
    df = pd.DataFrame([r[:en] for r in data_setirleri], columns=sutunlar[:en])
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace(null_deyer, np.nan)


def las_oxu(menbe) -> pd.DataFrame:
    """LAS faylını DataFrame-ə çevirir (fayl yolu və ya Streamlit upload obyekti)."""
    if hasattr(menbe, "read"):
        xam = menbe.read()
        metn = xam.decode("utf-8", errors="ignore") if isinstance(xam, bytes) else xam
    else:
        with open(menbe, "r", encoding="utf-8", errors="ignore") as f:
            metn = f.read()

    if LASIO_VAR:
        try:
            las = lasio.read(io.StringIO(metn))
            df = las.df().reset_index()
            df.columns = [str(c).upper() for c in df.columns]
            return df
        except Exception:
            pass
    return sade_las_parser(metn)


def sutun_tap(df: pd.DataFrame, acar: str):
    """Standart mnemonikalar üzrə sütunu tapır; tapmasa None qaytarır."""
    mövcud = {str(c).upper(): c for c in df.columns}
    for n in ADLAR.get(acar, []):
        if n in mövcud:
            return mövcud[n]
    for n in ADLAR.get(acar, []):
        if len(n) < 3:          # "W", "MZ" kimi qısa adlarda prefiks axtarışı yanlış nəticə verir
            continue
        for k, v in mövcud.items():
            if k.startswith(n):
                return v
    return None


def secim_indeksi(siyahi, tapilan):
    siyahi = list(siyahi)
    return siyahi.index(tapilan) if tapilan in siyahi else 0


def seriya_al(df: pd.DataFrame, sutun_adi: str) -> pd.Series:
    """Sütunu təhlükəsiz Series kimi qaytarır (təkrar mnemonika + mətn dəyər halı)."""
    s = df[sutun_adi]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def ort(x):
    """Bütün dəyərlər NaN olsa belə xəta verməyən orta."""
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 or np.all(np.isnan(x)) else float(np.nanmean(x))


def faiz(x, reqem=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{reqem}f} %"


def dolu_mu(seriya) -> bool:
    if seriya is None:
        return False
    return bool(np.isfinite(np.asarray(seriya, dtype=float)).any())


def dt_vahidi_tahmin(dt) -> str:
    """DT dəyərlərinə görə vahidi təxmin edir (µs/m dəyərləri xeyli böyükdür)."""
    orta = ort(dt)
    return "µs/m" if (orta is not None and not np.isnan(orta) and orta > 110) else "µs/ft"


# =================================================================
# 2) PETROFİZİKİ HESABLAMALAR
# =================================================================

def gamma_indeksi(gr, gr_min, gr_max):
    """ΔIγ = (Iγ − Iγmin) / (Iγmax − Iγmin)"""
    gr = np.asarray(gr, dtype=float)
    if abs(gr_max - gr_min) < 1e-9:
        return np.full_like(gr, np.nan)
    return np.clip((gr - gr_min) / (gr_max - gr_min), 0.0, 1.0)


def sp_indeksi(sp, sp_qum, sp_gil):
    """αSP = (SP − SP_qum) / (SP_gil − SP_qum)"""
    sp = np.asarray(sp, dtype=float)
    if abs(sp_gil - sp_qum) < 1e-9:
        return np.full_like(sp, np.nan)
    return np.clip((sp - sp_qum) / (sp_gil - sp_qum), 0.0, 1.0)


def gillilik(indeks, metod="Xətti (Cgil = ΔIγ)"):
    """Gillilik indeksi → Cgil (0…1)."""
    d = np.clip(np.asarray(indeks, dtype=float), 0.0, 1.0)
    if metod == "Larionov (gənc süxurlar)":
        c = 0.083 * (2.0 ** (3.7 * d) - 1.0)
    elif metod == "Larionov (köhnə süxurlar)":
        c = 0.33 * (2.0 ** (2.0 * d) - 1.0)
    elif metod == "Steiber":
        c = d / (3.0 - 2.0 * d)
    elif metod == "Clavier":
        c = 1.7 - np.sqrt(np.abs(3.38 - (d + 0.7) ** 2))
    else:
        c = d
    return np.clip(c, 0.0, 1.0)


def km_akustik(dt, dt_ma, dt_f, cgil=None, dt_gil=None, cp=1.0):
    """Wyllie: Km = (Δt − Δt_ma) / (Δt_f − Δt_ma)"""
    dt = np.asarray(dt, dtype=float)
    if abs(dt_f - dt_ma) < 1e-9:
        return np.full_like(dt, np.nan)
    km = (dt - dt_ma) / (dt_f - dt_ma) / max(cp, 1e-9)
    if cgil is not None and dt_gil is not None:
        km = km - np.asarray(cgil, dtype=float) * ((dt_gil - dt_ma) / (dt_f - dt_ma))
    return np.clip(km, 0.0, 1.0)


def km_sixliq(rhob, rho_ma, rho_f, cgil=None, rho_gil=None):
    """Sıxlıq: Km_D = (ρ_ma − ρ_b) / (ρ_ma − ρ_f)"""
    rhob = np.asarray(rhob, dtype=float)
    if abs(rho_ma - rho_f) < 1e-9:
        return np.full_like(rhob, np.nan)
    km = (rho_ma - rhob) / (rho_ma - rho_f)
    if cgil is not None and rho_gil is not None:
        km = km - np.asarray(cgil, dtype=float) * ((rho_ma - rho_gil) / (rho_ma - rho_f))
    return np.clip(km, 0.0, 1.0)


def km_neytron(nphi, cgil=None, nphi_gil=0.35):
    """Neytron məsaməliliyi; vahid avtomatik (>1.5 → faiz sayılır)."""
    n = np.asarray(nphi, dtype=float)
    if dolu_mu(n) and np.nanmax(n) > 1.5:
        n = n / 100.0
    if cgil is not None:
        n = n - np.asarray(cgil, dtype=float) * nphi_gil
    return np.clip(n, 0.0, 1.0)


def km_nkt(nkt, nkt_sifir, nkt_ust, km_ust=0.40):
    """
    NKT (neytron karotajı) üzrə məsaməlilik — iki nöqtəli kalibrləmə.
    Neytron göstəricisi məsaməlilik artdıqca AZALIR (hidrogen tutumu artır):
        Km = km_ust · (NKT_sıfır − NKT) / (NKT_sıfır − NKT_üst)
      nkt_sifir — sıx (məsaməsiz) süxur qarşısındakı NKT qiyməti
      nkt_ust   — km_ust məsaməliliyə uyğun NKT qiyməti
    """
    nkt = np.asarray(nkt, dtype=float)
    if abs(nkt_sifir - nkt_ust) < 1e-9:
        return np.full_like(nkt, np.nan)
    return np.clip(km_ust * (nkt_sifir - nkt) / (nkt_sifir - nkt_ust), 0.0, 1.0)


def km_w(w, faiz=True, emsal=1.0):
    """W əyrisi (su/məsaməlilik faizi) üzrə məsaməlilik: Km = əmsal · W / 100."""
    w = np.asarray(w, dtype=float)
    return np.clip(emsal * (w / 100.0 if faiz else w), 0.0, 1.0)


def km_birlesmis(km_n, km_d, qaz_duzelisi=False):
    """NPHI–RHOB birləşməsi: adi orta, qaz üçün kvadratik orta."""
    a, b = np.asarray(km_n, dtype=float), np.asarray(km_d, dtype=float)
    if qaz_duzelisi:
        return np.clip(np.sqrt((a ** 2 + b ** 2) / 2.0), 0.0, 1.0)
    return np.clip((a + b) / 2.0, 0.0, 1.0)


def mesamelik_parametri(km, a_n=1.0, n=2.0):
    """Arçi-Daxnov: Pm = a_n / Km^n"""
    km = np.asarray(km, dtype=float)
    return a_n / np.power(np.where(km > 1e-4, km, np.nan), n)


def sudoyumluluq(pm, rho_su, rt, m_sat=2.0):
    """ρ_sl = Pm·ρ_su ; Q = Rt/ρ_sl ; Ksu = (ρ_sl/Rt)^(1/m_sat)"""
    rt = np.asarray(rt, dtype=float)
    rho_sl = np.asarray(pm, dtype=float) * rho_su
    rt_temiz = np.where(rt > 1e-6, rt, np.nan)
    q = rt_temiz / rho_sl
    ksu = np.power(rho_sl / rt_temiz, 1.0 / m_sat)
    return rho_sl, q, np.clip(ksu, 0.0, 1.0)


def kaverna_tefsiri(dq, dn, tolerans=0.2):
    """dq < dn → kollektor · dq > dn → uçqun/gil · dq ≈ dn → bərk süxur"""
    dq = np.asarray(dq, dtype=float)
    netice = np.full(dq.shape, "Bərk süxur", dtype=object)
    netice[dq < dn - tolerans] = "Kollektor (gil qabığı)"
    netice[dq > dn + tolerans] = "Uçqun / gil"
    netice[~np.isfinite(dq)] = "—"
    return netice


def netpay_hesabla(data, var, cgil_cut, km_cut, knq_cut):
    """
    Cut-off sürgülərinə görə Net Pay maskası.
    Yalnız mövcud olan meyarlar tətbiq edilir (xətaya davamlılıq).
    Qaytarır: (maska, meyar_adlari)
    """
    meyarlar, adlar = [], []
    if var.get("CGIL"):
        meyarlar.append(np.nan_to_num(data["CGIL"].to_numpy(), nan=1.0) <= cgil_cut)
        adlar.append(f"Cgil ≤ {cgil_cut*100:.0f}%")
    if var.get("KM"):
        meyarlar.append(np.nan_to_num(data["KM"].to_numpy(), nan=-1.0) >= km_cut)
        adlar.append(f"Km ≥ {km_cut*100:.0f}%")
    if var.get("KSU"):
        meyarlar.append(np.nan_to_num(data["KNQ"].to_numpy(), nan=-1.0) >= knq_cut)
        adlar.append(f"Knq ≥ {knq_cut*100:.0f}%")

    if not meyarlar:
        return np.zeros(len(data), dtype=bool), []
    return np.logical_and.reduce(meyarlar), adlar


# =================================================================
# 3A) İNTERAKTİV QRAFİK — PLOTLY
# =================================================================

def _rgba(hex_reng, alfa):
    h = hex_reng.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alfa})"


# --- Paper (kağız) koordinatları ---------------------------------
# Plotly-də layout.xaxis.position YALNIZ [0, 1] aralığında ola bilər.
# Ona görə qrafik sahəsinin yuxarı sərhədini 0.85-ə endiririk və
# əlavə (twin) oxları onun ÜSTÜNDƏ, amma hələ də 1.0-dan aşağıda yerləşdiririk.
Y_DOMEN_UST = 0.85                      # y oxunun domeni: [0, 0.85]
ELAVE_POZ = [0.895, 0.945, 0.985]       # əlavə oxların mövqeləri — hamısı ≤ 1
BASLIQ_Y = 1.0                          # subplot başlıqları (annotation, paper)


def _poz(sira: int) -> float:
    """Əlavə oxun mövqeyi. Plotly [0,1] tələb edir — hər ehtimala qarşı kəsilir."""
    p = ELAVE_POZ[min(sira, len(ELAVE_POZ) - 1)]
    return float(np.clip(p, 0.0, 1.0))


def _ox_id(fig):
    """
    Sonuncu əlavə edilmiş trace-in REAL ox adlarını qaytarır.
    Ox adlarını əl ilə ('x2', 'y3') hesablamaq əvəzinə Plotly-nin özünün
    təyin etdiyi adları oxuyuruq — shared_yaxes rejimində bu daha etibarlıdır.
    """
    t = fig.data[-1]
    return (getattr(t, "xaxis", None) or "x"), (getattr(t, "yaxis", None) or "y")


def _trace_elave(fig, trace, ox_x=None, ox_y=None, row=None, col=None):
    """Trace-i ya subplot koordinatı (row/col), ya da konkret ox adı ilə əlavə edir."""
    if ox_x is not None:
        trace.update(xaxis=ox_x, yaxis=ox_y)
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def _araliqlar(maska):
    """
    Maskada ardıcıl True sahələrinin (başlanğıc, son) indekslərini qaytarır.
    Məs. [F,T,T,F,T] → [(1,3), (4,5)]
    """
    m = np.asarray(maska, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    kenar = np.diff(m.astype(np.int8))
    bas = list(np.flatnonzero(kenar == 1) + 1)
    son = list(np.flatnonzero(kenar == -1) + 1)
    if m[0]:
        bas.insert(0, 0)
    if m[-1]:
        son.append(m.size)
    return list(zip(bas, son))


def _zolaq_doldur(fig, x_sol, x_sag, depth, maska, reng, alfa, ad,
                  ox_x=None, ox_y=None, row=None, col=None):
    """
    Şaquli karotaj trekində düzgün sahə dolğusu.

    NƏ ÜÇÜN BELƏ:
      Dərinlik Y oxundadır. NaN boşluqları olan əyridə fill='tozerox'/'tonextx'
      istifadə edilsə, Plotly poliqonu boşluğun üstündən qapadır və trek boyu
      nəhəng diaqonal üçbucaqlar yaranır. Ona görə maskadakı hər ARDICIL sahə
      üçün ayrıca QAPALI poliqon (fill='toself') qurulur:
          sol sərhəd (yuxarıdan aşağı) + sağ sərhəd (aşağıdan yuxarı)
      Bu poliqon öz-özünə qapandığı üçün boşluqlar heç vaxt birləşmir.
    """
    x_sol = np.asarray(x_sol, dtype=float)
    x_sag = np.asarray(x_sag, dtype=float)
    # NaN olan nöqtələr dolğudan çıxarılır — üçbucaqların əsas mənbəyi budur
    maska = (np.asarray(maska, dtype=bool)
             & np.isfinite(x_sol) & np.isfinite(x_sag) & np.isfinite(depth))

    ilk = True
    for bas, son in _araliqlar(maska):
        if son - bas < 2:                     # tək nöqtə → görünməyən poliqon
            continue
        y = depth[bas:son]
        poly_x = np.concatenate([x_sol[bas:son], x_sag[bas:son][::-1]])
        poly_y = np.concatenate([y, y[::-1]])
        _trace_elave(fig, go.Scatter(
            x=poly_x, y=poly_y, mode="lines", fill="toself",
            fillcolor=_rgba(reng, alfa), line=dict(width=0),
            hoverinfo="skip", name=ad, legendgroup=ad,
            showlegend=ilk, connectgaps=False), ox_x, ox_y, row, col)
        ilk = False


def _netpay_zolaqlari(fig, depth, netpay, row, col, addim=None):
    """
    Net Pay treki: yalnız şərtləri ödəyən intervallar ÜFÜQİ YAŞIL ZOLAQ kimi.
    go.Bar(orientation='h') istifadə olunur — hər zolaq dəqiq düzbucaqlıdır,
    şərt ödənməyən dərinliklər tamamilə boş qalır (dolğu ümumiyyətlə çəkilmir).
    """
    netpay = np.asarray(netpay, dtype=bool)
    araliqlar = _araliqlar(netpay)
    if not araliqlar:
        return 0.0

    if addim is None or not np.isfinite(addim) or addim <= 0:
        addim = float(np.nanmedian(np.diff(depth))) if depth.size > 1 else 1.0

    merkez, en, metn = [], [], []
    umumi = 0.0
    for bas, son in araliqlar:
        ust = float(depth[bas]) - addim / 2.0        # zolağın yuxarı sərhədi
        alt = float(depth[son - 1]) + addim / 2.0    # aşağı sərhədi
        qalinliq = alt - ust
        umumi += qalinliq
        merkez.append((ust + alt) / 2.0)
        en.append(qalinliq)
        metn.append(f"{ust:.1f} – {alt:.1f} m  ({qalinliq:.1f} m)")

    fig.add_trace(go.Bar(
        x=np.ones(len(merkez)), y=merkez, width=en, base=0,
        orientation="h", marker=dict(color=RENG["NETPAY"],
                                     line=dict(width=0)),
        name="Xalis lay (Net Pay)", legendgroup="netpay",
        text=metn, hovertemplate="Xalis lay: %{text}<extra></extra>"),
        row=row, col=col)
    return umumi


def plotly_ciz(d, c, hundurluk=950):
    """Yalnız mövcud əyrilər üçün interaktiv trek qurur. Heç nə yoxdursa None."""
    var = c["var"]
    depth = d["DEPTH"].to_numpy(dtype=float)
    addim = float(np.nanmedian(np.diff(depth))) if depth.size > 1 else 1.0

    aktiv = []
    if any(var.get(k) for k in ("GR", "SP", "CALI")):
        aktiv.append(("LITO", 1.6))
    if any(var.get(k) for k in ("NKT", "W", "MZ")):
        aktiv.append(("DIGI", 1.3))
    if var.get("RT"):
        aktiv.append(("RT", 1.0))
    if var.get("CGIL") or var.get("KM"):
        aktiv.append(("PETRO", 1.3))
    if var.get("KSU"):
        aktiv.append(("SAT", 1.3))
    if var.get("NETPAY_VAR"):
        aktiv.append(("NP", 0.5))
    if not aktiv:
        return None

    basliqlar = {"LITO": "GR / SP / CALI", "DIGI": "NKT / W / MZ",
                 "RT": "Xüsusi müqavimət", "PETRO": "Km və Cgil",
                 "SAT": "Doyumluluq", "NP": "Net Pay"}
    fig = make_subplots(rows=1, cols=len(aktiv), shared_yaxes=True,
                        horizontal_spacing=0.035,
                        column_widths=[w for _, w in aktiv],
                        subplot_titles=[f"Track {i+1} — {basliqlar[a]}"
                                        for i, (a, _) in enumerate(aktiv)])

    elave_no = len(aktiv)      # əlavə oxların nömrələri subplot sayından sonra başlayır
    elave_oxlar = {}

    def yeni_ox(esas_ox, sira, ad, reng, aralik, log=False):
        """Mövcud oxun üstünə əlavə şaquli ox qurur (position həmişə [0,1])."""
        nonlocal elave_no
        elave_no += 1
        elave_oxlar[f"xaxis{elave_no}"] = dict(
            overlaying=esas_ox, side="top", anchor="free",
            position=_poz(sira), range=aralik,
            type="log" if log else "linear",
            title=dict(text=ad, font=dict(color=reng, size=11)),
            tickfont=dict(color=reng, size=9),
            showgrid=False, zeroline=False)
        return f"x{elave_no}"

    def xett(x, ad, reng, sablon, en=1.0, tire=None):
        """Standart əyri trace-i: boşluqlar HEÇ VAXT birləşdirilmir."""
        return go.Scatter(x=x, y=depth, name=ad, mode="lines",
                          line=dict(color=reng, width=en, dash=tire),
                          connectgaps=False, hovertemplate=sablon)

    for i, (acar, _) in enumerate(aktiv, start=1):
        esas_x = esas_y = None      # bu trekin əsas oxları (ilk trace-dən oxunur)
        sira = 0                    # neçənci əlavə ox

        # ---------- TRACK: LİTOLOGİYA ----------
        if acar == "LITO":
            if var.get("GR"):
                gr = d["GR"].to_numpy(dtype=float)
                ust = float(c["gr_ox_max"])
                # Qum: 0 → GR   |   Gil: GR → sağ kənar   (hər biri qapalı poliqon)
                _zolaq_doldur(fig, np.zeros_like(gr), gr, depth, gr < c["kesim"],
                              RENG["QUM"], 0.85, "Qum", row=1, col=i)
                _zolaq_doldur(fig, gr, np.full_like(gr, ust), depth, gr >= c["kesim"],
                              RENG["GIL"], 0.70, "Gil", row=1, col=i)
                fig.add_trace(xett(gr, "GR", RENG["GR"],
                                   "GR: %{x:.1f} API<extra></extra>"), row=1, col=i)
                esas_x, esas_y = _ox_id(fig)
                fig.update_xaxes(range=[0, ust], title_text="GR (API)", row=1, col=i)

            if var.get("SP"):
                sp = d["SP"].to_numpy(dtype=float)
                lim = np.nanmax(np.abs(sp)) * 1.15
                lim = 100.0 if not np.isfinite(lim) or lim == 0 else float(lim)
                trace = xett(sp, "SP", RENG["SP"], "SP: %{x:.1f} mV<extra></extra>")
                if esas_x is None:                       # GR yoxdur → əsas ox SP olur
                    fig.add_trace(trace, row=1, col=i)
                    esas_x, esas_y = _ox_id(fig)
                    fig.update_xaxes(range=[-lim, lim], title_text="SP (mV)", row=1, col=i)
                else:
                    ox = yeni_ox(esas_x, sira, "SP (mV)", RENG["SP"], [-lim, lim])
                    sira += 1
                    _trace_elave(fig, trace, ox, esas_y)

            if var.get("CALI"):
                cali = d["CALI"].to_numpy(dtype=float)
                alt_q, ust_q = np.nanmin(cali), np.nanmax(cali)
                aralik = [float(alt_q) * 0.85, float(ust_q) * 1.15] \
                    if np.isfinite(alt_q) and np.isfinite(ust_q) and ust_q > alt_q else None
                trace = xett(cali, "CALI (dq)", RENG["CALI"],
                             "dq: %{x:.2f}<extra></extra>", en=1.1)
                if esas_x is None:                       # yalnız CALI var
                    fig.add_trace(trace, row=1, col=i)
                    esas_x, esas_y = _ox_id(fig)
                    fig.update_xaxes(range=aralik, title_text="CALI (dq)", row=1, col=i)
                    ox = None
                else:
                    ox = yeni_ox(esas_x, sira, "CALI (dq)", RENG["CALI"], aralik)
                    sira += 1
                    _trace_elave(fig, trace, ox, esas_y)

                dn = c.get("dn")
                if dn is not None:
                    dn = float(dn)
                    # gil qabığı (dq < dn) və uçqun (dq > dn) zonaları
                    _zolaq_doldur(fig, cali, np.full_like(cali, dn), depth, cali < dn,
                                  "#90EE90", 0.55, "Gil qabığı", ox, esas_y,
                                  row=(1 if ox is None else None),
                                  col=(i if ox is None else None))
                    _zolaq_doldur(fig, np.full_like(cali, dn), cali, depth, cali > dn,
                                  "#CD853F", 0.45, "Uçqun", ox, esas_y,
                                  row=(1 if ox is None else None),
                                  col=(i if ox is None else None))
                    dn_trace = xett(np.full_like(depth, dn), "dn (nominal)",
                                    "darkred", "dn: %{x:.2f}<extra></extra>",
                                    tire="dashdot")
                    if ox is None:
                        fig.add_trace(dn_trace, row=1, col=i)
                    else:
                        _trace_elave(fig, dn_trace, ox, esas_y)

        # ---------- TRACK: RƏQƏMSALLAŞDIRILMIŞ ƏYRİLƏR ----------
        elif acar == "DIGI":
            for ad in ("NKT", "W", "MZ"):
                if not var.get(ad):
                    continue
                v = d[ad].to_numpy(dtype=float)
                alt_v, ust_v = np.nanmin(v), np.nanmax(v)
                if not (np.isfinite(alt_v) and np.isfinite(ust_v)) or ust_v <= alt_v:
                    alt_v, ust_v = 0.0, 1.0
                pad = (ust_v - alt_v) * 0.05
                aralik = [float(alt_v - pad), float(ust_v + pad)]
                vahid = " %" if ad == "W" else ""
                trace = xett(v, ad, RENG[ad],
                             ad + ": %{x:.2f}" + vahid + "<extra></extra>", en=1.1)
                if esas_x is None:
                    fig.add_trace(trace, row=1, col=i)
                    esas_x, esas_y = _ox_id(fig)
                    fig.update_xaxes(range=aralik, title_text=ETIKET[ad], row=1, col=i)
                else:
                    ox = yeni_ox(esas_x, sira, ETIKET[ad], RENG[ad], aralik)
                    sira += 1
                    _trace_elave(fig, trace, ox, esas_y)

        # ---------- TRACK: MÜQAVİMƏT ----------
        elif acar == "RT":
            rt = d["RT"].to_numpy(dtype=float)
            fig.add_trace(xett(rt, "Rt", RENG["RT"],
                               "Rt: %{x:.2f} Ω·m<extra></extra>"), row=1, col=i)
            esas_x, esas_y = _ox_id(fig)
            # loqarifmik oxda range LOQARİFM vahidində verilir: 10^-1 … 10^3
            fig.update_xaxes(type="log", range=[-1, 3], title_text="Rt (Ω·m)",
                             row=1, col=i)

        # ---------- TRACK: PETROFİZİKA ----------
        elif acar == "PETRO":
            if var.get("CGIL"):
                cgil = d["CGIL"].to_numpy(dtype=float) * 100.0
                _zolaq_doldur(fig, np.zeros_like(cgil), cgil, depth,
                              np.isfinite(cgil), RENG["CGIL"], 0.16, "Cgil sahəsi",
                              row=1, col=i)
                fig.add_trace(xett(cgil, "Cgil", RENG["CGIL"],
                                   "Cgil: %{x:.1f} %<extra></extra>"), row=1, col=i)
                esas_x, esas_y = _ox_id(fig)
                fig.update_xaxes(range=[0, 100], title_text="Cgil (%)", row=1, col=i)

            if var.get("KM"):
                km = d["KM"].to_numpy(dtype=float) * 100.0
                trace = xett(km, "Km", RENG["KM"],
                             "Km: %{x:.1f} %<extra></extra>", en=1.2)
                if esas_x is None:
                    _zolaq_doldur(fig, np.zeros_like(km), km, depth,
                                  np.isfinite(km), RENG["KM"], 0.18, "Km sahəsi",
                                  row=1, col=i)
                    fig.add_trace(trace, row=1, col=i)
                    esas_x, esas_y = _ox_id(fig)
                    fig.update_xaxes(range=[0, 100], title_text="Km (%)", row=1, col=i)
                else:
                    ox = yeni_ox(esas_x, sira, "Km (%)", RENG["KM"], [0, 100])
                    sira += 1
                    _zolaq_doldur(fig, np.zeros_like(km), km, depth,
                                  np.isfinite(km), RENG["KM"], 0.18, "Km sahəsi",
                                  ox, esas_y)
                    _trace_elave(fig, trace, ox, esas_y)

        # ---------- TRACK: DOYUMLULUQ ----------
        elif acar == "SAT":
            ksu = d["KSU"].to_numpy(dtype=float) * 100.0
            knq = d["KNQ"].to_numpy(dtype=float) * 100.0
            hedd = float(c["knq_cut"]) * 100.0
            _zolaq_doldur(fig, np.zeros_like(ksu), ksu, depth, np.isfinite(ksu),
                          RENG["KSU"], 0.18, "Su doyumu", row=1, col=i)
            # Neftli zona: yalnız Knq ≥ cut-off olan intervallarda, Knq → 100 %
            _zolaq_doldur(fig, knq, np.full_like(knq, 100.0), depth, knq >= hedd,
                          RENG["KNQ"], 0.45, "Neftli zona", row=1, col=i)
            fig.add_trace(xett(ksu, "Ksu", RENG["KSU"],
                               "Ksu: %{x:.1f} %<extra></extra>"), row=1, col=i)
            esas_x, esas_y = _ox_id(fig)
            fig.add_trace(xett(knq, "Knq", RENG["KNQ"],
                               "Knq: %{x:.1f} %<extra></extra>", en=1.2), row=1, col=i)
            fig.add_trace(xett(np.full_like(depth, hedd), "Knq cut-off", "black",
                               "cut-off: %{x:.0f} %<extra></extra>", tire="dash"),
                          row=1, col=i)
            fig.update_xaxes(range=[0, 100], title_text="Doyumluluq (%)", row=1, col=i)

        # ---------- TRACK: NET PAY ----------
        elif acar == "NP":
            netpay = np.asarray(d["NETPAY"].to_numpy(), dtype=bool)
            _netpay_zolaqlari(fig, depth, netpay, row=1, col=i, addim=addim)
            fig.update_xaxes(range=[0, 1], showticklabels=False,
                             showgrid=False, title_text="Net Pay", row=1, col=i)

    # --- Əlavə oxlar layout-a yazılır ---
    if elave_oxlar:
        fig.update_layout(**elave_oxlar)

    # --- Oxlar və ümumi görünüş ---
    fig.update_yaxes(domain=[0.0, Y_DOMEN_UST], autorange="reversed",
                     showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Dərinlik (m)", row=1, col=1)
    fig.update_xaxes(side="top", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_layout(
        height=hundurluk,
        margin=dict(t=160, b=40, l=70, r=30),
        hoverlabel=dict(namelength=-1),
        dragmode="zoom",
        barmode="overlay",          # Net Pay zolaqları üçün
        bargap=0,
        legend=dict(orientation="h", yanchor="top", y=-0.02, x=0),
        plot_bgcolor="white",
    )
    # "y unified" yalnız plotly ≥ 5.0-da var — köhnə versiyalarda "closest"
    try:
        fig.update_layout(hovermode="y unified")
    except Exception:
        fig.update_layout(hovermode="closest")

    # Subplot başlıqlarını əlavə oxların üstünə qaldır
    for annot in fig.layout.annotations:
        annot.update(y=BASLIQ_Y, yanchor="bottom", font=dict(size=11))
    return fig


# =================================================================
# 3B) STATİK QRAFİK — MATPLOTLIB (ehtiyat mühərrik)
# =================================================================

def _ox_hazirla(ax):
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.55)


def _elave_ox(ax, mesafe):
    yeni = ax.twiny()
    yeni.xaxis.set_label_position("top")
    yeni.xaxis.tick_top()
    yeni.spines["top"].set_position(("outward", mesafe))
    return yeni


def mpl_track_litologiya(ax, d, c):
    depth = d["DEPTH"].to_numpy()
    mesafe, basliq = 0, []

    if c["var"].get("GR"):
        gr = d["GR"].to_numpy()
        ust = c["gr_ox_max"]
        ax.plot(gr, depth, color="black", linewidth=0.75, zorder=4)
        ax.fill_betweenx(depth, 0, gr, where=(gr < c["kesim"]),
                         facecolor="gold", interpolate=True, zorder=2)
        ax.fill_betweenx(depth, gr, ust, where=(gr >= c["kesim"]),
                         facecolor="grey", interpolate=True, zorder=2)
        ax.axvline(c["kesim"], color="red", linestyle="--", linewidth=1, zorder=5)
        ax.set_xlim(0, ust)
        ax.set_xlabel("GR (API)")
        ax.legend(handles=[Patch(facecolor="gold", label="Qum"),
                           Patch(facecolor="grey", label="Gil")],
                  loc="lower left", fontsize=7.5, framealpha=0.9)
        basliq.append("GR")

    if c["var"].get("SP"):
        sp = d["SP"].to_numpy()
        hedef = ax if not basliq else _elave_ox(ax, mesafe := mesafe + 26)
        hedef.plot(sp, depth, color="darkviolet", linewidth=0.85)
        lim = np.nanmax(np.abs(sp)) * 1.15
        lim = 100.0 if not np.isfinite(lim) or lim == 0 else lim
        hedef.set_xlim(-lim, lim)
        hedef.set_xlabel("SP (mV)", color="darkviolet")
        hedef.tick_params(axis="x", colors="darkviolet")
        if hedef is ax:
            _ox_hazirla(ax)
        basliq.append("SP")

    if c["var"].get("CALI"):
        cali = d["CALI"].to_numpy()
        hedef = ax if not basliq else _elave_ox(ax, mesafe := mesafe + 26)
        hedef.plot(cali, depth, color="seagreen", linewidth=0.9)
        dn = c.get("dn")
        if dn is not None:
            hedef.axvline(dn, color="darkred", linestyle="-.", linewidth=1.1)
            hedef.fill_betweenx(depth, dn, cali, where=(cali > dn),
                                facecolor="peru", alpha=0.35)
            hedef.fill_betweenx(depth, cali, dn, where=(cali < dn),
                                facecolor="lightgreen", alpha=0.45)
        alt, ust = np.nanmin(cali), np.nanmax(cali)
        if np.isfinite(alt) and np.isfinite(ust) and ust > alt:
            hedef.set_xlim(alt * 0.85, ust * 1.15)
        hedef.set_xlabel("CALI  dq / dn", color="seagreen")
        hedef.tick_params(axis="x", colors="seagreen")
        if hedef is ax:
            _ox_hazirla(ax)
        basliq.append("CALI")

    if not basliq:
        ax.set_xticks([])
    return " / ".join(basliq) if basliq else "—"


def mpl_track_digi(ax, d, c):
    """Rəqəmsallaşdırılmış NKT / W / MZ əyriləri (statik mühərrik)."""
    depth = d["DEPTH"].to_numpy()
    etiketler, mesafe = [], 0
    for ad, reng in (("NKT", "darkblue"), ("W", "darkturquoise"), ("MZ", "forestgreen")):
        if not c["var"].get(ad):
            continue
        v = d[ad].to_numpy(dtype=float)
        hedef = ax if not etiketler else _elave_ox(ax, mesafe := mesafe + 26)
        hedef.plot(v, depth, color=reng, linewidth=0.9)
        alt_v, ust_v = np.nanmin(v), np.nanmax(v)
        if np.isfinite(alt_v) and np.isfinite(ust_v) and ust_v > alt_v:
            pad = (ust_v - alt_v) * 0.05
            hedef.set_xlim(alt_v - pad, ust_v + pad)
        hedef.set_xlabel(ad, color=reng)
        hedef.tick_params(axis="x", colors=reng)
        if hedef is ax:
            _ox_hazirla(ax)
        etiketler.append(ad)
    if not etiketler:
        ax.set_xticks([])
    return " / ".join(etiketler) if etiketler else "—"


def mpl_track_muqavimet(ax, d, c):
    ax.plot(d["RT"].to_numpy(), d["DEPTH"].to_numpy(), color="firebrick", linewidth=0.85)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_xlabel("Rt (Ω·m)", color="firebrick")
    ax.tick_params(axis="x", colors="firebrick")
    return "Xüsusi müqavimət"


def mpl_track_petrofizika(ax, d, c):
    depth = d["DEPTH"].to_numpy()
    etiketler = []

    if c["var"].get("CGIL"):
        cgil = d["CGIL"].to_numpy() * 100.0
        ax.plot(cgil, depth, color="saddlebrown", linewidth=0.85)
        ax.fill_betweenx(depth, 0, cgil, facecolor="saddlebrown", alpha=0.16)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Cgil (%)", color="saddlebrown")
        ax.tick_params(axis="x", colors="saddlebrown")
        etiketler.append("Cgil")

    if c["var"].get("KM"):
        km = d["KM"].to_numpy() * 100.0
        hedef = ax if not etiketler else _elave_ox(ax, 26)
        hedef.plot(km, depth, color="royalblue", linewidth=0.95)
        hedef.fill_betweenx(depth, 0, km, facecolor="royalblue", alpha=0.18)
        hedef.set_xlim(0, 100)
        hedef.set_xlabel("Km (%)", color="royalblue")
        hedef.tick_params(axis="x", colors="royalblue")
        if hedef is ax:
            _ox_hazirla(ax)
        etiketler.append("Km")

    if not etiketler:
        ax.set_xticks([])
    return " və ".join(etiketler) if etiketler else "—"


def mpl_track_doyumluluq(ax, d, c):
    depth = d["DEPTH"].to_numpy()
    ksu = d["KSU"].to_numpy() * 100.0
    knq = d["KNQ"].to_numpy() * 100.0
    hedd = c["knq_cut"] * 100.0
    ax.plot(ksu, depth, color="dodgerblue", linewidth=0.9, label="Ksu")
    ax.plot(knq, depth, color="green", linewidth=0.9, label="Knq")
    ax.fill_betweenx(depth, 0, ksu, facecolor="dodgerblue", alpha=0.18)
    ax.fill_betweenx(depth, knq, 100, where=(knq >= hedd),
                     facecolor="limegreen", alpha=0.55, interpolate=True)
    ax.axvline(hedd, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Doyumluluq (%)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    return "Doyumluluq (Ksu / Knq)"


def mpl_track_netpay(ax, d, c):
    depth = d["DEPTH"].to_numpy()
    netpay = d["NETPAY"].to_numpy().astype(bool)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    if c["var"].get("KSU"):
        knq = d["KNQ"].to_numpy()
        ax.fill_betweenx(depth, 0, 1, where=(knq >= c["knq_cut"]),
                         facecolor="honeydew", interpolate=True)
    ax.fill_betweenx(depth, 0, 1, where=netpay,
                     facecolor="forestgreen", interpolate=True)
    ax.legend(handles=[Patch(facecolor="forestgreen", label="Xalis lay")],
              loc="lower center", fontsize=7, framealpha=0.95)
    return "Net Pay"


TREK_REYESTRI = [
    ("LITO", lambda v: any(v.get(k) for k in ("GR", "SP", "CALI")), mpl_track_litologiya, 1.5),
    ("DIGI", lambda v: any(v.get(k) for k in ("NKT", "W", "MZ")), mpl_track_digi, 1.3),
    ("RT", lambda v: v.get("RT"), mpl_track_muqavimet, 1.0),
    ("PETRO", lambda v: v.get("CGIL") or v.get("KM"), mpl_track_petrofizika, 1.2),
    ("SAT", lambda v: v.get("KSU"), mpl_track_doyumluluq, 1.2),
    ("NP", lambda v: v.get("NETPAY_VAR"), mpl_track_netpay, 0.55),
]


def mpl_ciz(d, c):
    """Matplotlib ilə statik treklər (ehtiyat mühərrik / PNG ixracı üçün)."""
    aktiv = [(a, f, w) for a, sert, f, w in TREK_REYESTRI if sert(c["var"])]
    if not aktiv:
        return None

    fig, oxlar = plt.subplots(
        1, len(aktiv), figsize=(3.4 * len(aktiv) + 0.6, 12), sharey=True,
        gridspec_kw={"width_ratios": [w for _, _, w in aktiv]})
    oxlar = np.atleast_1d(oxlar)
    fig.subplots_adjust(top=0.80, wspace=0.10, left=0.07, right=0.98)

    for i, ((acar, cek, _), ax) in enumerate(zip(aktiv, oxlar), start=1):
        _ox_hazirla(ax)
        ad = cek(ax, d, c)
        ax.set_title(f"Track {i}\n{ad}", pad=46, fontsize=10, fontweight="bold")

    oxlar[0].set_ylabel("Dərinlik (m)")
    depth = d["DEPTH"].to_numpy()
    oxlar[0].set_ylim(np.nanmax(depth), np.nanmin(depth))
    return fig


# =================================================================
# 3C) KAROTAJ DİAQRAM ŞƏKLİNİN RƏQƏMSALLAŞDIRILMASI (DIGITIZER)
# =================================================================
#  Şəkildəki əyri seçilmiş rəngə görə maskalanır, hər piksel sətri
#  (y) bir dərinliyə, tapılan piksel sütunu (x) isə əyrinin qiymətinə
#  çevrilir:
#       dərinlik = ust_d + (y + 0.5)/H · (alt_d − ust_d)
#       qiymət   = v_min + (x + 0.5)/W · (v_max − v_min)     [xətti]
#       qiymət   = 10^( lg v_min + (x + 0.5)/W · lg(v_max/v_min) )  [log]
# =================================================================

# HSV aralıqları (OpenCV konvensiyası: H = 0…179)
RENG_ARALIQ = {
    "Qırmızı": [((0, 80, 50), (10, 255, 255)), ((166, 80, 50), (179, 255, 255))],
    "Yaşıl":   [((36, 60, 40), (86, 255, 255))],
    "Mavi":    [((90, 60, 40), (135, 255, 255))],
    "Qara":    None,          # xüsusi hal: aşağı parlaqlıq + aşağı doyum
}

# Rəqəmsallaşdırıla bilən əyrilər və onların ilkin şkalaları
DIGI_EGRILER = {
    "GR":   {"ad": "GR (Qamma)",            "min": 0.0,   "max": 150.0, "log": False},
    "RT":   {"ad": "Rt (Xüsusi müqavimət)", "min": 0.2,   "max": 200.0, "log": True},
    "DT":   {"ad": "DT (Akustik)",          "min": 40.0,  "max": 240.0, "log": False},
    "SP":   {"ad": "SP (Quyu potensialı)",  "min": -80.0, "max": 20.0,  "log": False},
    "CALI": {"ad": "CALI (Kavernomer)",     "min": 6.0,   "max": 16.0,  "log": False},
    "RHOB": {"ad": "RHOB (Sıxlıq)",         "min": 1.95,  "max": 2.95,  "log": False},
    "NPHI": {"ad": "NPHI (Neytron)",        "min": 0.0,   "max": 60.0,  "log": False},
}


def sekil_oxu(bayt) -> np.ndarray:
    """Yüklənmiş baytları RGB massivinə çevirir (cv2 varsa cv2, yoxdursa PIL)."""
    xam = np.frombuffer(bayt, dtype=np.uint8)
    if CV2_VAR:
        bgr = cv2.imdecode(xam, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Şəkil oxuna bilmədi (format dəstəklənmir).")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image                       # ehtiyat yol
    return np.array(Image.open(io.BytesIO(bayt)).convert("RGB"))


def rgb_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV (OpenCV miqyası: H 0-179, S/V 0-255). cv2 yoxdursa NumPy ilə."""
    if CV2_VAR:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    import matplotlib.colors as mcolors
    hsv = mcolors.rgb_to_hsv(rgb.astype(np.float32) / 255.0)
    hsv[..., 0] *= 179.0
    hsv[..., 1] *= 255.0
    hsv[..., 2] *= 255.0
    return hsv.astype(np.uint8)


def reng_maskasi(rgb: np.ndarray, reng_ad: str, doyum_min: int = 80,
                 parlaqliq_min: int = 50, qara_hedd: int = 90) -> np.ndarray:
    """Seçilmiş rəngdəki pikselləri tapan boolean maska."""
    if reng_ad == "Qara":
        # qara/tünd əyri: parlaqlıq aşağı, rəng doyumu aşağı
        hsv = rgb_hsv(rgb)
        return (hsv[..., 2] < qara_hedd) & (hsv[..., 1] < 120)

    hsv = rgb_hsv(rgb)
    maska = np.zeros(rgb.shape[:2], dtype=bool)
    for (h1, s1, v1), (h2, s2, v2) in RENG_ARALIQ[reng_ad]:
        s1 = max(s1, doyum_min)
        v1 = max(v1, parlaqliq_min)
        maska |= ((hsv[..., 0] >= h1) & (hsv[..., 0] <= h2) &
                  (hsv[..., 1] >= s1) & (hsv[..., 2] >= v1))
    return maska


def egri_cixart(maska: np.ndarray, ust_d: float, alt_d: float,
                v_min: float, v_max: float, log_miqyas: bool = False,
                usul: str = "median"):
    """
    Maskadan (H×W) əyrini çıxarır: hər sətir üçün bir qiymət.
    Qaytarır: (dərinliklər, qiymətlər) — piksel olmayan sətirlərdə NaN.
    """
    H, W = maska.shape
    depth = ust_d + (np.arange(H) + 0.5) / H * (alt_d - ust_d)
    x_piksel = np.full(H, np.nan, dtype=float)

    for r in range(H):
        sutunlar = np.flatnonzero(maska[r])
        if sutunlar.size:
            if usul == "sol":
                x_piksel[r] = sutunlar[0]
            elif usul == "sağ":
                x_piksel[r] = sutunlar[-1]
            elif usul == "orta":
                x_piksel[r] = sutunlar.mean()
            else:
                x_piksel[r] = np.median(sutunlar)

    nisbi = (x_piksel + 0.5) / W
    if log_miqyas:
        v_min = max(float(v_min), 1e-6)
        v_max = max(float(v_max), v_min * 1.0000001)
        qiymet = 10.0 ** (np.log10(v_min) + nisbi * (np.log10(v_max) - np.log10(v_min)))
    else:
        qiymet = v_min + nisbi * (v_max - v_min)
    qiymet[~np.isfinite(x_piksel)] = np.nan
    return depth, qiymet


def setre_yerlesdir(depth, qiymet, grid, hamarlama: int = 0,
                    maks_bosluq: float = None) -> np.ndarray:
    """
    Piksel sətirlərindən alınan əyrini nizamlı dərinlik şəbəkəsinə köçürür.

    DİQQƏT: np.interp istənilən uzunluqda boşluğu səssizcə "körpüləyir".
    Ona görə maks_bosluq (m) parametri ilə yalnız KİÇİK boşluqlar doldurulur;
    daha uzun boşluqlar (məs. əyrilərin uzun müddət üst-üstə düşdüyü zonalar)
    NaN qalır və sonra idarə olunan şəkildə df.interpolate() ilə doldurulur.
    """
    etibarli = np.isfinite(qiymet)
    if etibarli.sum() < 2:
        return np.full(len(grid), np.nan)

    d_e, q_e = depth[etibarli], qiymet[etibarli]
    sira = np.argsort(d_e)
    d_e, q_e = d_e[sira], q_e[sira]

    if hamarlama and hamarlama > 1:
        q_e = pd.Series(q_e).rolling(int(hamarlama), center=True,
                                     min_periods=1).median().to_numpy()

    netice = np.interp(grid, d_e, q_e, left=np.nan, right=np.nan)
    netice[(grid < d_e[0]) | (grid > d_e[-1])] = np.nan

    if maks_bosluq is not None and maks_bosluq > 0:
        yer = np.searchsorted(d_e, grid)
        sol = np.clip(yer - 1, 0, d_e.size - 1)
        sag = np.clip(yer, 0, d_e.size - 1)
        mesafe = np.minimum(np.abs(grid - d_e[sol]), np.abs(d_e[sag] - grid))
        netice[mesafe > maks_bosluq] = np.nan
    return netice


def maska_onizleme(rgb: np.ndarray, maska: np.ndarray) -> np.ndarray:
    """Tapılan piksellər qırmızı ilə vurğulanmış önizləmə şəkli."""
    onizleme = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    onizleme[maska] = np.array([255, 0, 0], dtype=np.uint8)
    return onizleme


# -----------------------------------------------------------------
#  ÇOXRƏNGLİ (MULTI-COLOR) ÇIXARILMA — 4 əyri eyni anda
# -----------------------------------------------------------------
#  Tipik köhnə karotaj diaqramlarında 4 rəngli əyri olur:
#     GK  — qırmızı        (qamma karotajı)
#     NKT — tünd mavi      (neytron karotajı)
#     W   — açıq mavi / firuzəyi (rütubət / məsaməlilik, %)
#     MZ  — yaşıl          (mikrozond)
#  Hər rəng ayrıca HSV aralığı ilə maskalanır və ayrı sütun kimi
#  vahid DataFrame-ə yığılır.
# -----------------------------------------------------------------

COX_EGRI = {
    "GK":  {"ad": "GK — Qamma (qırmızı)",              "reng": "#D62728",
            "h": (0, 10), "h2": (166, 179),
            "min": 1.5, "max": 9.5,  "log": False, "hedef": "GR"},
    "NKT": {"ad": "NKT — Neytron (tünd mavi)",          "reng": "#00008B",
            "h": (105, 135), "h2": None,
            "min": 1.2, "max": 4.4,  "log": False, "hedef": "NKT"},
    "W":   {"ad": "W % — Rütubət (açıq mavi/firuzəyi)", "reng": "#00CED1",
            "h": (78, 104), "h2": None,
            "min": -30.0, "max": 50.0, "log": False, "hedef": "W"},
    "MZ":  {"ad": "MZ — Mikrozond (yaşıl)",             "reng": "#228B22",
            "h": (36, 86), "h2": None,
            "min": 1.2, "max": 2.8,  "log": False, "hedef": "MZ"},
}


def hsv_maska(rgb: np.ndarray, h_aralik, h_aralik2=None,
              s_min: int = 70, v_min: int = 40, v_max: int = 255) -> np.ndarray:
    """Verilmiş H (rəng çaları) aralığına düşən pikselləri tapır."""
    hsv = rgb_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    maska = ((h >= h_aralik[0]) & (h <= h_aralik[1]) &
             (s >= s_min) & (v >= v_min) & (v <= v_max))
    if h_aralik2 is not None:
        maska |= ((h >= h_aralik2[0]) & (h <= h_aralik2[1]) &
                  (s >= s_min) & (v >= v_min) & (v <= v_max))
    return maska


def coxrengli_cixart(rgb: np.ndarray, konfiqler: dict, ust_d: float, alt_d: float,
                     grid: np.ndarray, s_min: int = 70, v_min: int = 40,
                     hamarlama: int = 3, usul: str = "median",
                     doldur: bool = True, interp_limit: int = 0):
    """
    Şəkildəki bütün seçilmiş rəngli əyriləri BİR keçiddə çıxarır.

    konfiqler: {acar: {"h":(a,b), "h2":(a,b)|None, "min":.., "max":.., "log":bool}}
    Qaytarır: (DataFrame, statistika, maskalar)

    Xətlərin kəsişdiyi yerlərdə piksellər itir (üstdəki xətt altdakını örtür).
    Bu boşluqlar df.interpolate(method="linear") ilə doldurulur —
    yalnız DAXİLİ boşluqlar (limit_area="inside"), yəni əyrinin əvvəli/sonu
    süni şəkildə uzadılmır. interp_limit = 0 → limitsiz, >0 → ən çoxu bu
    qədər ardıcıl nöqtə doldurulur (uzun boşluqlar NaN qalır).
    """
    setirler = {"DEPTH": grid}
    statistika, maskalar = {}, {}
    # şəbəkə addımının 3 misli — bundan uzun boşluqlar df.interpolate-ə buraxılır
    kicik_bosluq = 3.0 * float(np.median(np.diff(grid))) if grid.size > 1 else None

    for acar, kf in konfiqler.items():
        maska = hsv_maska(rgb, kf["h"], kf.get("h2"), s_min, v_min)
        maskalar[acar] = maska
        dep_px, qiymet_px = egri_cixart(maska, ust_d, alt_d,
                                        kf["min"], kf["max"], kf.get("log", False), usul)
        setirler[acar] = setre_yerlesdir(dep_px, qiymet_px, grid, hamarlama,
                                         maks_bosluq=kicik_bosluq)
        statistika[acar] = {
            "piksel": int(maska.sum()),
            "xam_ortuk": float(np.isfinite(qiymet_px).mean() * 100),
            "ortuk": float(np.isfinite(setirler[acar]).mean() * 100),
        }

    df = pd.DataFrame(setirler)

    # --- Kəsişmə (overlap) səbəbindən yaranan boşluqların doldurulması ---
    sutunlar = [c for c in df.columns if c != "DEPTH"]
    if sutunlar and doldur:
        df[sutunlar] = df[sutunlar].interpolate(
            method="linear", limit_area="inside",
            limit=(int(interp_limit) if interp_limit and interp_limit > 0 else None))
    if sutunlar:
        for acar in sutunlar:
            statistika[acar]["son_ortuk"] = float(np.isfinite(df[acar]).mean() * 100)
            statistika[acar]["doldurulan"] = (statistika[acar]["son_ortuk"] -
                                              statistika[acar]["ortuk"])
    return df, statistika, maskalar


def cox_onizleme(rgb: np.ndarray, maskalar: dict) -> np.ndarray:
    """Bütün maskalar öz rəngləri ilə vurğulanmış önizləmə."""
    onizleme = (rgb.astype(np.float32) * 0.30).astype(np.uint8)
    for acar, maska in maskalar.items():
        h = COX_EGRI[acar]["reng"].lstrip("#")
        rgb_reng = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                            dtype=np.uint8)
        onizleme[maska] = rgb_reng
    return onizleme


# =================================================================
### UI_BASLANGIC ###
# 4) STREAMLIT İNTERFEYSİ
# =================================================================

st.set_page_config(page_title="Petrofiziki İnterpretasiya", page_icon="🛢️", layout="wide")
st.title("🛢️ Kompleks Petrofiziki İnterpretasiya")
st.caption("Dinamik matrisa · dinamik cut-off · interaktiv treklər")

# ---------------- Məlumat mənbəyi: LAS və ya şəkil ----------------
st.sidebar.header("📂 Məlumat mənbəyi")
REJIM_LAS = "LAS Faylı Yüklə"
REJIM_SEKIL = "Karotaj Diaqram Şəkli Yüklə (Digitizer)"
rejim = st.sidebar.radio("Rejim", [REJIM_LAS, REJIM_SEKIL])


def digitizer_paneli():
    """
    Şəkil rəqəmsallaşdırma paneli — iki alt rejim:
      A) 4 rəngli standart diaqram: GK (qırmızı), NKT (tünd mavi),
         W (açıq mavi/firuzəyi), MZ (yaşıl) — HAMISI BİR KEÇİDDƏ oxunur.
      B) Tək əyri (fərdi rəng və şkala) — məs. Rt və ya DT ayrıca şəkildən.
    Nəticələr st.session_state-də toplanır, ona görə müxtəlif şəkillərdən
    gələn əyriləri birləşdirmək mümkündür.
    """
    if "digi" not in st.session_state:
        st.session_state.digi = {"grid": None, "egriler": {}, "qeyd": {}}
    anbar = st.session_state.digi

    MULTI = "4 rəngli standart diaqram (GK/NKT/W/MZ)"
    TEK = "Tək əyri (fərdi rəng)"
    alt_rejim = st.sidebar.radio("Diaqram tipi", [MULTI, TEK])

    sekil_fayl = st.sidebar.file_uploader("Karotaj şəkli (PNG / JPG)",
                                          type=["png", "jpg", "jpeg", "bmp"])

    st.sidebar.subheader("📏 Miqyaslama")
    ust_d = st.sidebar.number_input("Başlanğıc dərinlik — Top (m)", value=1000.0, step=10.0)
    alt_d = st.sidebar.number_input("Bitiş dərinliyi — Bottom (m)", value=1500.0, step=10.0)
    grid_addim = st.sidebar.number_input("Dərinlik addımı (m)", value=0.5,
                                         min_value=0.01, step=0.05, format="%.2f")

    st.sidebar.subheader("✂️ Qrafik sahəsi (ROI, %)")
    x_aralik = st.sidebar.slider("Üfüqi (sol → sağ)", 0, 100, (0, 100))
    y_aralik = st.sidebar.slider("Şaquli (yuxarı → aşağı)", 0, 100, (0, 100))

    st.sidebar.subheader("🎛️ Piksel filtri")
    doyum = st.sidebar.slider("Rəng doyumu həddi (S)", 0, 255, 70, 5)
    parlaqliq = st.sidebar.slider("Parlaqlıq həddi (V)", 0, 255, 40, 5)
    usul = st.sidebar.selectbox("Qalın xətdə nöqtə seçimi",
                                ["median", "orta", "sol", "sağ"])
    hamarlama = st.sidebar.slider("Hamarlama (nöqtə)", 0, 25, 3, 1)

    # ---------------- Şkala panelləri ----------------
    konfiqler, sec_edilmis = {}, []
    if alt_rejim == MULTI:
        st.sidebar.subheader("📊 Şkalalar (hər əyri üçün Min / Max)")
        for acar, kf in COX_EGRI.items():
            aktiv = st.sidebar.checkbox(kf["ad"], value=True, key=f"akt_{acar}")
            s1, s2 = st.sidebar.columns(2)
            v_min = s1.number_input("Min", value=float(kf["min"]), step=0.1,
                                    format="%.2f", key=f"min_{acar}")
            v_max = s2.number_input("Max", value=float(kf["max"]), step=0.1,
                                    format="%.2f", key=f"max_{acar}")
            if aktiv:
                sec_edilmis.append(acar)
                konfiqler[acar] = {"h": kf["h"], "h2": kf["h2"],
                                   "min": v_min, "max": v_max, "log": kf["log"]}

        with st.sidebar.expander("🔬 Rəng çalarının dəqiqləşdirilməsi (HSV)"):
            st.caption("Şəkildəki rənglər standartdan fərqlənirsə H aralığını dəyişin "
                       "(OpenCV: H = 0…179).")
            for acar in sec_edilmis:
                h_var = st.slider(f"{acar} — H aralığı", 0, 179,
                                  COX_EGRI[acar]["h"], key=f"h_{acar}")
                konfiqler[acar]["h"] = h_var

        st.sidebar.subheader("🔗 Kəsişmələr")
        doldur = st.sidebar.checkbox("Boşluqları interpolyasiya ilə doldur "
                                     "(df.interpolate)", value=True)
        interp_limit = st.sidebar.slider("Maksimum doldurulan ardıcıl nöqtə "
                                         "(0 = limitsiz)", 0, 200, 40, 5)
        mz_rt = st.sidebar.checkbox("MZ-ni Rt (müqavimət) kimi istifadə et", value=False,
                                    help="MZ mikrozonddur — dayaz zona. Ksu yalnız "
                                         "təxmini xarakter daşıyır.")
    else:
        st.sidebar.subheader("📈 Əyri və şkala")
        egri_acar = st.sidebar.selectbox("Əyri növü", list(DIGI_EGRILER.keys()),
                                         format_func=lambda k: DIGI_EGRILER[k]["ad"])
        ilkin = DIGI_EGRILER[egri_acar]
        v_min = st.sidebar.number_input(f"{egri_acar} — şkala MIN",
                                        value=float(ilkin["min"]), step=1.0, format="%.3f")
        v_max = st.sidebar.number_input(f"{egri_acar} — şkala MAX",
                                        value=float(ilkin["max"]), step=1.0, format="%.3f")
        log_miqyas = st.sidebar.checkbox("Loqarifmik şkala", value=bool(ilkin["log"]))
        reng_ad = st.sidebar.selectbox("İzlənəcək rəng", list(RENG_ARALIQ.keys()))
        doldur, interp_limit, mz_rt = True, 40, False

    # ---------------- Əsas sahə ----------------
    st.subheader("🖼️ Şəkil rəqəmsallaşdırma (Digitizer)")
    if not CV2_VAR:
        st.info("OpenCV tapılmadı — PIL ilə işləyir. Tam funksionallıq üçün: "
                "`pip install opencv-python-headless`")

    if sekil_fayl is None:
        st.info("👈 Karotaj diaqramının şəklini yükləyin, dərinlik və şkala "
                "parametrlərini verin, sonra çıxarma düyməsini basın.")
    else:
        try:
            rgb = sekil_oxu(sekil_fayl.getvalue())
        except Exception as e:
            st.error(f"Şəkil oxunmadı: {e}")
            rgb = None

        if rgb is not None:
            H, W = rgb.shape[:2]
            x0, x1 = int(W * x_aralik[0] / 100), max(int(W * x_aralik[1] / 100), 1)
            y0, y1 = int(H * y_aralik[0] / 100), max(int(H * y_aralik[1] / 100), 1)
            kesik = rgb[y0:y1, x0:x1]

            if kesik.size == 0:
                st.error("ROI boşdur — sahə sürgülərini düzəldin.")
            elif alt_rejim == MULTI:
                # ---------- 4 rəngli rejim ----------
                if not konfiqler:
                    st.warning("Ən azı bir əyri seçin.")
                else:
                    grid = np.arange(min(ust_d, alt_d), max(ust_d, alt_d) + 1e-9,
                                     float(grid_addim))
                    cerceve, stat, maskalar = coxrengli_cixart(
                        kesik, konfiqler, ust_d, alt_d, grid,
                        s_min=doyum, v_min=parlaqliq, hamarlama=hamarlama,
                        usul=usul, doldur=doldur, interp_limit=interp_limit)

                    s1, s2 = st.columns(2)
                    s1.image(kesik, caption=f"Seçilmiş sahə — {kesik.shape[1]}×"
                                            f"{kesik.shape[0]} px",
                             use_container_width=True)
                    s2.image(cox_onizleme(kesik, maskalar),
                             caption="Tapılan piksellər (hər əyri öz rəngi ilə)",
                             use_container_width=True)

                    st.markdown("**Çıxarılma statistikası**")
                    st.table(pd.DataFrame([{
                        "Əyri": a,
                        "Rəng": COX_EGRI[a]["ad"].split("(")[-1].rstrip(")"),
                        "Piksel": f"{stat[a]['piksel']:,}",
                        "Xam örtük": f"{stat[a]['ortuk']:.1f} %",
                        "İnterp. sonra": f"{stat[a].get('son_ortuk', 0):.1f} %",
                        "Doldurulan": f"{stat[a].get('doldurulan', 0):+.1f} %",
                        "Şkala": f"{konfiqler[a]['min']:g} … {konfiqler[a]['max']:g}",
                    } for a in konfiqler]))

                    zeif = [a for a in konfiqler if stat[a]["piksel"] == 0]
                    if zeif:
                        st.warning("Bu rənglərdə piksel tapılmadı: " + ", ".join(zeif) +
                                   ". HSV aralığını və ya doyum/parlaqlıq həddini dəyişin.")

                    if st.button("➕ Bütün əyriləri əlavə et", type="primary"):
                        if alt_d == ust_d:
                            st.error("Başlanğıc və bitiş dərinliyi eyni ola bilməz.")
                        else:
                            anbar_grid_yenile(anbar, grid)
                            for a in konfiqler:
                                hedef = COX_EGRI[a]["hedef"]
                                anbar["egriler"][hedef] = cerceve[a].to_numpy()
                                anbar["qeyd"][hedef] = (
                                    f"{a} · {konfiqler[a]['min']:g}–{konfiqler[a]['max']:g}"
                                    f" · örtük {stat[a].get('son_ortuk', 0):.0f} %")
                            if mz_rt and "MZ" in konfiqler:
                                anbar["egriler"]["RT"] = cerceve["MZ"].to_numpy()
                                anbar["qeyd"]["RT"] = "MZ əyrisindən (dayaz zona!)"
                            st.success(f"{len(konfiqler)} əyri əlavə edildi.")
            else:
                # ---------- tək əyri rejimi ----------
                maska = reng_maskasi(kesik, reng_ad, doyum, parlaqliq)
                dep_px, qiymet_px = egri_cixart(maska, ust_d, alt_d,
                                                v_min, v_max, log_miqyas, usul)
                ortuk = float(np.isfinite(qiymet_px).mean() * 100)

                s1, s2 = st.columns(2)
                s1.image(kesik, caption=f"Seçilmiş sahə — {kesik.shape[1]}×"
                                        f"{kesik.shape[0]} px", use_container_width=True)
                s2.image(maska_onizleme(kesik, maska),
                         caption=f"Tapılan piksellər: {int(maska.sum()):,} · "
                                 f"sətir örtüyü: {ortuk:.1f} %", use_container_width=True)

                if maska.sum() == 0:
                    st.warning("Seçilmiş rəngdə piksel tapılmadı — rəngi dəyişin və ya "
                               "doyum/parlaqlıq həddini azaldın.")
                elif ortuk < 50:
                    st.warning(f"Sətirlərin yalnız {ortuk:.0f} %-ində əyri tapıldı.")

                if st.button(f"➕ «{egri_acar}» əyrisini əlavə et", type="primary"):
                    if alt_d == ust_d:
                        st.error("Başlanğıc və bitiş dərinliyi eyni ola bilməz.")
                    else:
                        grid = np.arange(min(ust_d, alt_d), max(ust_d, alt_d) + 1e-9,
                                         float(grid_addim))
                        anbar_grid_yenile(anbar, grid)
                        anbar["egriler"][egri_acar] = setre_yerlesdir(
                            dep_px, qiymet_px, anbar["grid"], hamarlama,
                            maks_bosluq=3.0 * float(grid_addim))
                        anbar["qeyd"][egri_acar] = (f"{reng_ad} · {v_min:g}–{v_max:g}"
                                                    f"{' (log)' if log_miqyas else ''} · "
                                                    f"örtük {ortuk:.0f} %")
                        st.success(f"«{egri_acar}» əlavə edildi.")

    # ---------------- Toplanmış əyrilər ----------------
    if anbar["egriler"]:
        st.markdown("**Toplanmış (rəqəmsallaşdırılmış) əyrilər**")
        st.table(pd.DataFrame(
            [{"Sütun": k, "Mənbə": anbar["qeyd"].get(k, ""),
              "Etibarlı nöqtə": int(np.isfinite(v).sum()),
              "Min": f"{np.nanmin(v):.2f}" if np.isfinite(v).any() else "—",
              "Max": f"{np.nanmax(v):.2f}" if np.isfinite(v).any() else "—"}
             for k, v in anbar["egriler"].items()]))

        if st.button("🗑️ Hamısını təmizlə"):
            st.session_state.digi = {"grid": None, "egriler": {}, "qeyd": {}}
            st.rerun()

        cerceve = pd.DataFrame({"DEPTH": anbar["grid"], **anbar["egriler"]})
        st.download_button("⬇️ Rəqəmsallaşdırılmış datanı CSV kimi yüklə",
                           cerceve.to_csv(index=False).encode("utf-8"),
                           file_name="reqemsallasdirilmis_karotaj.csv", mime="text/csv")
        return cerceve

    return None


def anbar_grid_yenile(anbar, grid):
    """Dərinlik şəbəkəsi dəyişibsə, əvvəlki əyriləri yeni şəbəkəyə köçürür."""
    if anbar["grid"] is None:
        anbar["grid"] = grid
        return
    if len(anbar["grid"]) == len(grid) and np.allclose(anbar["grid"], grid):
        return
    for k, v in list(anbar["egriler"].items()):
        anbar["egriler"][k] = np.interp(grid, anbar["grid"], v,
                                        left=np.nan, right=np.nan)
    anbar["grid"] = grid


if rejim == REJIM_LAS:
    yuklenen = st.sidebar.file_uploader("LAS faylını yüklə", type=["las", "LAS"])
    yol = st.sidebar.text_input("və ya fayl yolu", value="WA1.LAS")

    menbe = yuklenen if yuklenen is not None else (yol if os.path.exists(yol) else None)
    if menbe is None:
        st.info("👈 Yan paneldən LAS faylını yükləyin və ya düzgün fayl yolunu yazın.")
        st.stop()
    try:
        df = las_oxu(menbe)
    except Exception as e:
        st.error(f"Fayl oxunmadı: {e}")
        st.stop()
else:
    df = digitizer_paneli()
    if df is None or df.empty:
        st.stop()
    st.success(f"Rəqəmsallaşdırılmış data hesablamalara ötürüldü: "
               f"{len(df)} nöqtə · əyrilər: {', '.join(c for c in df.columns if c != 'DEPTH')}")

sut = list(df.columns)
if not sut:
    st.error("Heç bir sütun tapılmadı.")
    st.stop()

# ---------------- Əyrilərin təyini ----------------
st.sidebar.header("🧭 Əyrilərin təyini")
depth_col = st.sidebar.selectbox("Dərinlik (məcburi)", sut,
                                 index=secim_indeksi(sut, sutun_tap(df, "DEPTH")))
secimler = {}
for acar in OPSIONAL:
    secimler[acar] = st.sidebar.selectbox(
        ETIKET[acar], [YOX] + sut,
        index=secim_indeksi([YOX] + sut, sutun_tap(df, acar)))

# ---------------- Data hazırlığı (dropna yalnız dərinliyə görə) ----------------
if depth_col not in df.columns:
    st.error(f"Seçilmiş `{depth_col}` sütunu faylda yoxdur.")
    st.stop()

xam = df.dropna(subset=[depth_col])
if xam.empty:
    st.error(f"`{depth_col}` sütununda etibarlı dərinlik dəyəri yoxdur.")
    st.stop()

secilmis = {"DEPTH": depth_col}
for acar in OPSIONAL:
    if secimler[acar] != YOX and secimler[acar] in xam.columns:
        secilmis[acar] = secimler[acar]

data = pd.DataFrame({daxili: seriya_al(xam, las_adi).to_numpy()
                     for daxili, las_adi in secilmis.items()})
data = data.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
if data.empty:
    st.error("Dərinlik sütununda ədədi dəyər yoxdur.")
    st.stop()

var = {}
for acar in OPSIONAL:
    var[acar] = acar in data.columns and dolu_mu(data[acar])
    if acar in data.columns and not var[acar]:
        st.sidebar.warning(f"`{secimler[acar]}` sütunu tamamilə boşdur — istifadə edilmir.")

# ---------------- Vəziyyət paneli ----------------
mövcud_ad = [ETIKET[a].split(" ")[0] for a in OPSIONAL if var[a]]
eksik_ad = [ETIKET[a].split(" ")[0] for a in OPSIONAL if not var[a]]
with st.expander(f"🔎 Faylda tapılan əyrilər: {len(mövcud_ad)}/{len(OPSIONAL)}",
                 expanded=bool(eksik_ad)):
    s1, s2 = st.columns(2)
    s1.success("✅ Mövcud: " + (", ".join(mövcud_ad) if mövcud_ad else "yalnız dərinlik"))
    s2.warning("⚠️ Əskik: " + (", ".join(eksik_ad) if eksik_ad else "yoxdur"))
    st.caption("Əskik əyrilər üçün hesablamalar alternativ mənbədən aparılır və ya "
               "ötürülür — proqram dayanmır.")

P, qeydler = {}, []

# =================================================================
#  SÜXUR LİTOLOGİYASI (MATRİSA) — dinamik seçim
# =================================================================
st.sidebar.header("🪨 Süxur litologiyası (Matrisa)")
P["litologiya"] = st.sidebar.selectbox("Matrisa tipi", list(MATRISA.keys()), index=0)

# DT vahidi: mövcud datadan təxmin edilir, istifadəçi dəyişə bilər
tahmin = dt_vahidi_tahmin(data["DT"]) if var["DT"] else "µs/m"
P["vahid"] = st.sidebar.radio("Δt vahidi", ["µs/ft", "µs/m"],
                              index=0 if tahmin == "µs/ft" else 1, horizontal=True)
j = 0 if P["vahid"] == "µs/ft" else 1

if P["litologiya"] == "Fərdi (Custom)":
    P["dt_ma"] = st.sidebar.number_input(f"Δt_matrix ({P['vahid']})",
                                         value=float(MATRISA["Fərdi (Custom)"]["dt"][j]),
                                         min_value=1.0, step=0.5, format="%.1f")
    P["rho_ma"] = st.sidebar.number_input("ρ_matrix (q/sm³)", value=2.650,
                                          min_value=1.0, step=0.01, format="%.3f")
else:
    P["dt_ma"] = float(MATRISA[P["litologiya"]]["dt"][j])
    P["rho_ma"] = float(MATRISA[P["litologiya"]]["rho"])
    st.sidebar.caption(f"Δt_matrix = **{P['dt_ma']:.1f} {P['vahid']}** · "
                       f"ρ_matrix = **{P['rho_ma']:.2f} q/sm³**")

P["maye"] = st.sidebar.selectbox("Maye (flüid)", list(MAYE.keys()))
P["dt_f"] = st.sidebar.number_input(f"Δt_fluid ({P['vahid']})",
                                    value=float(MAYE[P["maye"]]["dt"][j]), step=1.0)
P["rho_f"] = st.sidebar.number_input("ρ_fluid (q/sm³)",
                                     value=float(MAYE[P["maye"]]["rho"]), step=0.01)

# =================================================================
#  GİLLİLİK
# =================================================================
st.sidebar.header("🟫 Gillilik (Cgil)")
cgil_variantlari = []
if var["GR"]:
    cgil_variantlari.append("GR (ΔIγ)")
if var["SP"]:
    cgil_variantlari.append("SP (αSP)")
cgil_variantlari.append("Sabit ilkin dəyər")
cgil_menbe = st.sidebar.selectbox("Mənbə", cgil_variantlari, index=0)

if cgil_menbe == "GR (ΔIγ)":
    avto = st.sidebar.checkbox("Iγmin / Iγmax avtomatik (P5 / P95)", value=True)
    p5, p95 = float(np.nanpercentile(data["GR"], 5)), float(np.nanpercentile(data["GR"], 95))
    if avto:
        P["gr_min"], P["gr_max"] = p5, p95
        st.sidebar.caption(f"LAS-dan: Iγmin = {p5:.1f}, Iγmax = {p95:.1f} API")
    else:
        P["gr_min"] = st.sidebar.number_input("Iγmin — təmiz qum (API)",
                                              value=round(p5, 1), step=1.0)
        P["gr_max"] = st.sidebar.number_input("Iγmax — təmiz gil (API)",
                                              value=round(p95, 1), step=1.0)
elif cgil_menbe == "SP (αSP)":
    sp5 = float(np.nanpercentile(data["SP"], 5))
    sp95 = float(np.nanpercentile(data["SP"], 95))
    P["sp_qum"] = st.sidebar.number_input("SP — təmiz qum (mV)", value=round(sp5, 1), step=1.0)
    P["sp_gil"] = st.sidebar.number_input("SP — gil xətti (mV)", value=round(sp95, 1), step=1.0)
    qeydler.append("GR olmadığı üçün gillilik SP əyrisindən (αSP) hesablandı.")
else:
    P["cgil_sabit"] = st.sidebar.slider("Sabit Cgil (%)", 0, 100, 20, 5) / 100.0
    qeydler.append(f"GR və SP olmadığı üçün gillilik sabit "
                   f"{P['cgil_sabit']*100:.0f} % qəbul edildi.")

P["gil_metod"] = st.sidebar.selectbox(
    "Cgil modeli", ["Xətti (Cgil = ΔIγ)", "Larionov (gənc süxurlar)",
                    "Larionov (köhnə süxurlar)", "Steiber", "Clavier"])
P["kesim"] = st.sidebar.slider("GR litoloji kəsimi (API)", 0.0, 200.0, 75.0, 1.0) \
    if var["GR"] else 75.0

# =================================================================
#  MƏSAMƏLİLİK
# =================================================================
st.sidebar.header("🔵 Məsaməlilik (Km)")
km_variantlari = []
if var["DT"]:
    km_variantlari.append("DT — Wyllie")
if var["RHOB"]:
    km_variantlari.append("RHOB — sıxlıq")
if var["NPHI"]:
    km_variantlari.append("NPHI — neytron")
if var["NPHI"] and var["RHOB"]:
    km_variantlari.append("NPHI + RHOB ortası")
if var["NKT"]:
    km_variantlari.append("NKT — neytron (kalibrləmə)")
if var["W"]:
    km_variantlari.append("W — məsaməlilik (%)")
km_variantlari.append("Sabit ilkin dəyər")
km_menbe = st.sidebar.selectbox("Mənbə", km_variantlari, index=0)

if km_menbe != "Sabit ilkin dəyər":
    P["gil_duzelis"] = st.sidebar.checkbox("Gilliliyə görə düzəliş", value=False)

if km_menbe == "DT — Wyllie":
    P["cp"] = st.sidebar.number_input("Kompaksiya əmsalı Cp (≥1)", value=1.0,
                                      min_value=1.0, step=0.05)
    P["dt_gil"] = st.sidebar.number_input(f"Δt_gil ({P['vahid']})",
                                          value=float(GIL_DT[P["vahid"]]), step=1.0) \
        if P.get("gil_duzelis") else None
elif km_menbe in ("RHOB — sıxlıq", "NPHI + RHOB ortası"):
    P["rho_gil"] = st.sidebar.number_input("ρ_gil (q/sm³)", value=GIL_RHO, step=0.01) \
        if P.get("gil_duzelis") else None
    if km_menbe == "NPHI + RHOB ortası":
        P["qaz"] = st.sidebar.checkbox("Qaz düzəlişi (kvadratik orta)", value=False)
    qeydler.append("Məsaməlilik sıxlıq karotajından hesablandı.")
elif km_menbe == "NPHI — neytron":
    P["nphi_gil"] = st.sidebar.number_input("NPHI_gil (hissə)", value=0.35, step=0.05) \
        if P.get("gil_duzelis") else 0.35
    qeydler.append("Məsaməlilik neytron karotajından götürüldü.")
elif km_menbe == "NKT — neytron (kalibrləmə)":
    nkt_p5 = float(np.nanpercentile(data["NKT"].dropna(), 5)) if var["NKT"] else 1.2
    nkt_p95 = float(np.nanpercentile(data["NKT"].dropna(), 95)) if var["NKT"] else 4.4
    st.sidebar.caption("İki nöqtəli kalibrləmə: NKT artdıqca məsaməlilik azalır.")
    P["nkt_sifir"] = st.sidebar.number_input("NKT — sıx süxur (Km ≈ 0)",
                                             value=round(nkt_p95, 2), step=0.05, format="%.2f")
    P["nkt_ust"] = st.sidebar.number_input("NKT — məsaməli süxur",
                                           value=round(nkt_p5, 2), step=0.05, format="%.2f")
    P["km_ust"] = st.sidebar.slider("Həmin nöqtədə Km (%)", 1, 60, 35, 1) / 100.0
    qeydler.append("Məsaməlilik NKT əyrisindən iki nöqtəli kalibrləmə ilə hesablandı.")
elif km_menbe == "W — məsaməlilik (%)":
    P["w_faiz"] = st.sidebar.checkbox("W dəyərləri faizlə (%) verilib", value=True)
    P["w_emsal"] = st.sidebar.number_input("Miqyas əmsalı", value=1.00, step=0.05, format="%.2f")
    qeydler.append("Məsaməlilik W (rütubət/məsaməlilik %) əyrisindən götürüldü.")
else:
    P["km_sabit"] = st.sidebar.slider("Sabit Km (%)", 0, 40, 20, 1) / 100.0
    qeydler.append(f"DT / RHOB / NPHI olmadığı üçün məsaməlilik sabit "
                   f"{P['km_sabit']*100:.0f} % qəbul edildi.")

# =================================================================
#  ARÇİ-DAXNOV
# =================================================================
st.sidebar.header("⚡ Arçi-Daxnov (doyumluluq)")
if var["RT"]:
    P["a_n"] = st.sidebar.number_input("a_n (tortuozluq)", value=1.00, step=0.05, format="%.2f")
    P["n"] = st.sidebar.number_input("n (sementləşmə)", value=2.00, step=0.10, format="%.2f")
    P["m_sat"] = st.sidebar.number_input("Doyumluluq eksponenti", value=2.00,
                                         step=0.10, format="%.2f")
    P["rho_su"] = st.sidebar.number_input("ρ_su — lay suyu (Ω·m)", value=0.050,
                                          min_value=0.001, step=0.005, format="%.3f")
else:
    st.sidebar.info("Rt yoxdur → doyumluluq hesablanmır, digər treklər çəkilir.")
    qeydler.append("Rt olmadığı üçün Arçi-Daxnov doyumluluğu ötürüldü.")

# =================================================================
#  NET PAY CUT-OFF SÜRGÜLƏRİ
# =================================================================
st.sidebar.header("🎯 Net Pay cut-off")
P["cgil_cut"] = st.sidebar.slider("Maksimum gillilik — Cgil ≤ (%)", 0, 100, 30, 1,
                                  help="Bu həddən gilli laylar kollektor sayılmır") / 100.0
P["km_cut"] = st.sidebar.slider("Minimum məsaməlilik — Km ≥ (%)", 0, 40, 10, 1,
                                help="Bu həddən aşağı məsaməlilik sıx süxur sayılır") / 100.0
P["knq_cut"] = st.sidebar.slider("Minimum neftdoyumluluq — Knq ≥ (%)", 0, 100, 50, 1,
                                 help="Dərslik qaydası: Knq > 50% → neftli-qazlı lay") / 100.0

# ---------------- Kavernometriya ----------------
st.sidebar.header("📐 Kavernometriya")
if var["CALI"]:
    P["cali_vahid"] = st.sidebar.radio("Diametr vahidi", ["düym", "mm"], horizontal=True)
    P["dn"] = st.sidebar.number_input(f"Nominal diametr dn ({P['cali_vahid']})",
                                      value=8.5 if P["cali_vahid"] == "düym" else 216.0,
                                      step=0.5)
    P["tolerans"] = st.sidebar.number_input(f"Tolerans ± ({P['cali_vahid']})",
                                            value=0.2 if P["cali_vahid"] == "düym" else 5.0,
                                            step=0.1)
else:
    P["dn"] = None
    st.sidebar.info("CALI yoxdur → kaverna təhlili aparılmır.")

# ---------------- İnterval və qrafik ----------------
st.sidebar.header("📏 İnterval və qrafik")
d_min, d_max = float(data["DEPTH"].min()), float(data["DEPTH"].max())
if d_max > d_min:
    interval = st.sidebar.slider("Dərinlik (m)", d_min, d_max, (d_min, d_max))
    data = data[(data["DEPTH"] >= interval[0]) &
                (data["DEPTH"] <= interval[1])].reset_index(drop=True)

muherrik_secimler = (["Plotly (interaktiv)", "Matplotlib (statik)"] if PLOTLY_VAR
                     else ["Matplotlib (statik)"])
P["muherrik"] = st.sidebar.radio("Qrafik mühərriki", muherrik_secimler)
if not PLOTLY_VAR:
    st.sidebar.caption("İnteraktiv rejim üçün: `pip install plotly`")
P["hundurluk"] = st.sidebar.slider("Qrafik hündürlüyü (px)", 600, 1600, 950, 50)

# Çox uzun loqlar üçün seyrəkləşdirmə (interaktivlik sürətli qalsın)
if len(data) > 15000:
    addim_say = int(np.ceil(len(data) / 15000))
    st.sidebar.caption(f"⚡ {len(data):,} nöqtə → qrafik üçün hər {addim_say}-ci nöqtə")
else:
    addim_say = 1

# =================================================================
# 5) HESABLAMALAR
# =================================================================

n_setir = len(data)

# --- Gillilik ---
if cgil_menbe == "GR (ΔIγ)":
    data["INDEKS"] = gamma_indeksi(data["GR"].to_numpy(), P["gr_min"], P["gr_max"])
    data["CGIL"] = gillilik(data["INDEKS"], P["gil_metod"])
elif cgil_menbe == "SP (αSP)":
    data["INDEKS"] = sp_indeksi(data["SP"].to_numpy(), P["sp_qum"], P["sp_gil"])
    data["CGIL"] = gillilik(data["INDEKS"], P["gil_metod"])
else:
    data["INDEKS"] = np.nan
    data["CGIL"] = np.full(n_setir, P["cgil_sabit"])

var["CGIL"] = dolu_mu(data["CGIL"])
cgil_ded = data["CGIL"].to_numpy() if (var["CGIL"] and P.get("gil_duzelis")) else None

# --- Məsaməlilik (matrisa parametrləri dinamik) ---
if km_menbe == "DT — Wyllie":
    data["KM"] = km_akustik(data["DT"].to_numpy(), P["dt_ma"], P["dt_f"],
                            cgil=cgil_ded, dt_gil=P.get("dt_gil"), cp=P["cp"])
elif km_menbe == "RHOB — sıxlıq":
    data["KM"] = km_sixliq(data["RHOB"].to_numpy(), P["rho_ma"], P["rho_f"],
                           cgil=cgil_ded, rho_gil=P.get("rho_gil"))
elif km_menbe == "NPHI — neytron":
    data["KM"] = km_neytron(data["NPHI"].to_numpy(), cgil=cgil_ded,
                            nphi_gil=P.get("nphi_gil", 0.35))
elif km_menbe == "NKT — neytron (kalibrləmə)":
    data["KM"] = km_nkt(data["NKT"].to_numpy(), P["nkt_sifir"], P["nkt_ust"], P["km_ust"])
elif km_menbe == "W — məsaməlilik (%)":
    data["KM"] = km_w(data["W"].to_numpy(), P.get("w_faiz", True), P.get("w_emsal", 1.0))
elif km_menbe == "NPHI + RHOB ortası":
    kn = km_neytron(data["NPHI"].to_numpy(), cgil=cgil_ded)
    kd = km_sixliq(data["RHOB"].to_numpy(), P["rho_ma"], P["rho_f"],
                   cgil=cgil_ded, rho_gil=P.get("rho_gil"))
    data["KM"] = km_birlesmis(kn, kd, P.get("qaz", False))
else:
    data["KM"] = np.full(n_setir, P["km_sabit"])

var["KM"] = dolu_mu(data["KM"])
if var["KM"]:
    data["KM_EFF"] = np.clip(data["KM"] * (1.0 - data["CGIL"]), 0, 1) \
        if var["CGIL"] else data["KM"]
    data["PM"] = mesamelik_parametri(data["KM"].to_numpy(), P.get("a_n", 1.0), P.get("n", 2.0))

# --- Doyumluluq ---
var["KSU"] = bool(var["KM"] and var["RT"])
if var["KSU"]:
    rho_sl, q_ems, ksu = sudoyumluluq(data["PM"].to_numpy(), P["rho_su"],
                                      data["RT"].to_numpy(), P["m_sat"])
    data["RHO_SL"], data["Q"], data["KSU"] = rho_sl, q_ems, ksu
    data["KNQ"] = np.clip(1.0 - ksu, 0.0, 1.0)
    var["KSU"] = dolu_mu(data["KSU"])

# --- Kavernometriya ---
if var["CALI"]:
    data["KAVERNA"] = kaverna_tefsiri(data["CALI"].to_numpy(), P["dn"], P["tolerans"])

# --- Net Pay (cut-off sürgülərinə görə dinamik) ---
maska, meyar_adlari = netpay_hesabla(data, var, P["cgil_cut"], P["km_cut"], P["knq_cut"])
data["NETPAY"] = maska
var["NETPAY_VAR"] = bool(meyar_adlari)

if meyar_adlari and len(meyar_adlari) < 3:
    qeydler.append("Net Pay yalnız mövcud meyarlarla təyin edildi: " + ", ".join(meyar_adlari))

# --- Təyinat ---
if var["KSU"]:
    data["TEYINAT"] = np.where(data["CGIL"] > P["cgil_cut"], "Gil",
                               np.where(data["KNQ"] >= P["knq_cut"], "Neftli-Qazlı", "Sulu"))
elif var["CGIL"]:
    data["TEYINAT"] = np.where(data["CGIL"] > P["cgil_cut"], "Gil", "Kollektor (?)")
else:
    data["TEYINAT"] = "—"

# --- Qalınlıqlar ---
addim = float(np.nanmedian(np.diff(data["DEPTH"].to_numpy()))) if n_setir > 1 else 0.0
net_pay_qalinliq = float(np.nansum(data["NETPAY"].to_numpy().astype(float)) * addim)
brutto = float(data["DEPTH"].max() - data["DEPTH"].min())

# =================================================================
# 6) METRİKLƏR
# =================================================================

m = st.columns(5)
m[0].metric("Ümumi interval", f"{brutto:.1f} m",
            help=f"{data['DEPTH'].min():.1f} – {data['DEPTH'].max():.1f} m")
m[1].metric("Orta Cgil", faiz(ort(data["CGIL"])) if var["CGIL"] else "—")
m[2].metric("Orta Km", faiz(ort(data["KM"])) if var["KM"] else "—")
m[3].metric("Orta Knq", faiz(ort(data["KNQ"])) if var["KSU"] else "—")
m[4].metric("Net Pay qalınlığı", f"{net_pay_qalinliq:.1f} m" if meyar_adlari else "—",
            delta=(f"NTG {net_pay_qalinliq/brutto*100:.1f} %"
                   if meyar_adlari and brutto > 0 else None))

if qeydler:
    st.info("ℹ️ " + "  \n".join("• " + q for q in qeydler))

# =================================================================
# 7) QRAFİK VƏ CƏDVƏL
# =================================================================

konf = {"var": var, "kesim": P["kesim"], "knq_cut": P["knq_cut"], "dn": P.get("dn"),
        "gr_ox_max": float(max(150.0, np.nanmax(data["GR"]) * 1.05)) if var["GR"] else 150.0}

sekme1, sekme2, sekme3 = st.tabs(["📈 Karotaj diaqramı", "📋 Cədvəl", "🧮 Düsturlar"])

with sekme1:
    if not var["RT"]:
        st.warning("⚠️ **Rt əyrisi yoxdur** — doyumluluq treki (Ksu / Knq) çəkilmir və "
                   "Arçi-Daxnov hesablaması aparılmır. Litologiya, rəqəmsallaşdırılmış "
                   "əyrilər və məsaməlilik/gillilik trekləri normal göstərilir.")
    qrafik_data = data.iloc[::addim_say].reset_index(drop=True) if addim_say > 1 else data

    if P["muherrik"].startswith("Plotly") and PLOTLY_VAR:
        try:
            fig = plotly_ciz(qrafik_data, konf, hundurluk=P["hundurluk"])
            if fig is None:
                st.warning("Çəkiləcək əyri yoxdur.")
            else:
                st.plotly_chart(fig, use_container_width=True,
                                config={"scrollZoom": True, "displaylogo": False,
                                        "toImageButtonOptions": {"format": "png",
                                                                 "filename": "karotaj",
                                                                 "scale": 2}})
                st.caption("🖱️ Zoom: sahə seçin · Pan: sağ düymə · İkiqat klik: ilkin görünüş · "
                           "Hover: dərinlik üzrə bütün göstəricilər")
        except Exception as e:
            st.warning(f"Plotly qrafiki qurula bilmədi ({e}) — matplotlib rejiminə keçildi.")
            fig = mpl_ciz(qrafik_data, konf)
            if fig is not None:
                st.pyplot(fig)
    else:
        fig = mpl_ciz(qrafik_data, konf)
        if fig is None:
            st.warning("Çəkiləcək əyri yoxdur.")
        else:
            st.pyplot(fig)

with sekme2:
    cedvel = pd.DataFrame({"DEPTH (m)": data["DEPTH"].round(2)})
    for acar, ad, reqem in [("GR", "GR (API)", 1), ("SP", "SP (mV)", 1),
                            ("CALI", "CALI (dq)", 2), ("RT", "Rt (Ω·m)", 2),
                            ("DT", "DT", 1), ("RHOB", "RHOB", 3), ("NPHI", "NPHI", 3)]:
        if var[acar]:
            cedvel[ad] = data[acar].round(reqem)
    if var["CALI"]:
        cedvel["Kaverna"] = data["KAVERNA"]
    if var["CGIL"]:
        cedvel["Cgil (%)"] = (data["CGIL"] * 100).round(1)
    if var["KM"]:
        cedvel["Km (%)"] = (data["KM"] * 100).round(1)
        cedvel["Pm"] = data["PM"].round(1)
    if var["KSU"]:
        cedvel["Ksu (%)"] = (data["KSU"] * 100).round(1)
        cedvel["Knq (%)"] = (data["KNQ"] * 100).round(1)
    if meyar_adlari:
        cedvel["Net Pay"] = np.where(data["NETPAY"], "✅", "")
    cedvel["Təyinat"] = data["TEYINAT"]

    st.dataframe(cedvel, use_container_width=True, height=620)
    st.download_button("⬇️ Nəticələri CSV kimi yüklə",
                       cedvel.to_csv(index=False).encode("utf-8"),
                       file_name="petrofiziki_interpretasiya.csv", mime="text/csv")

with sekme3:
    st.latex(r"\Delta I_\gamma=\frac{I_\gamma-I_{\gamma\min}}{I_{\gamma\max}-I_{\gamma\min}}"
             r"\qquad \alpha_{SP}=\frac{SP-SP_{qum}}{SP_{gil}-SP_{qum}}")
    st.latex(r"K_m=\frac{\Delta t-\Delta t_{ma}}{\Delta t_{f}-\Delta t_{ma}}"
             r"\qquad K_{m}^{D}=\frac{\rho_{ma}-\rho_{b}}{\rho_{ma}-\rho_{f}}")
    st.latex(r"P_m=\frac{a_n}{K_m^{\,n}}\qquad \rho_{sl}=P_m\rho_{su}"
             r"\qquad K_{su}=\sqrt{\frac{\rho_{sl}}{R_t}}\qquad K_{nq}=1-K_{su}")

    st.markdown(f"""
**Matrisa:** {P['litologiya']} — Δt_ma = **{P['dt_ma']:.1f} {P['vahid']}**, ρ_ma = **{P['rho_ma']:.2f} q/sm³**
**Maye:** {P['maye']} — Δt_f = {P['dt_f']:.1f} {P['vahid']}, ρ_f = {P['rho_f']:.2f} q/sm³

- Gillilik mənbəyi: **{cgil_menbe}**, model: **{P['gil_metod']}**
- Məsaməlilik mənbəyi: **{km_menbe}**
- Doyumluluq: **{"Arçi-Daxnov" if var['KSU'] else "hesablanmadı"}**
- Net Pay meyarları: **{", ".join(meyar_adlari) if meyar_adlari else "meyar yoxdur"}**
- Dərinlik addımı: **{addim:.2f} m** · Xalis qalınlıq: **{net_pay_qalinliq:.1f} m**

**Kavernometriya:** dq < dn → kollektor (gil qabığı) · dq > dn → uçqun/gil · dq ≈ dn → bərk süxur
""")