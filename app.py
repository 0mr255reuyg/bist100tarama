import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BIST 100 Strateji Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0d0f14; color: #e2e8f0; }
    .main-header { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem; font-family: 'JetBrains Mono', monospace; }
    .metric-card { background: #161b27; border: 1px solid #1e2535; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; }
    .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
    .metric-pass { color: #22c55e !important; }
    .metric-fail { color: #ef4444 !important; }
    .section-title { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace; margin: 1.2rem 0 0.6rem 0; border-bottom: 1px solid #1e2535; padding-bottom: 0.4rem; }
    div[data-testid="stButton"] button { border-radius: 8px; font-weight: 500; font-size: 0.85rem; }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 0.8rem !important; }
    .search-result-card { background: #12172a; border: 1px solid #2d3748; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .search-result-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; }
    .crit-row { display: flex; justify-content: space-between; align-items: center; padding: 0.25rem 0; border-bottom: 1px solid #1e2535; font-size: 0.8rem; }
    .crit-name { color: #94a3b8; }
    .crit-pass { color: #22c55e; font-family: 'JetBrains Mono', monospace; }
    .crit-fail { color: #ef4444; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── BIST 100 TAM LİSTE ──────────────────────────────────────────────────────
BIST100_TICKERS = [
    "ACSEL","ADEL","ADESE","AFYON","AGESA","AGHOL","AHGAZ","AKBNK","AKCNS","AKFGY",
    "AKGRT","AKSA","AKSEL","AKSEN","AKSGY","AKSUE","AKYHO","ALARK","ALBRK","ALCAR",
    "ALCTL","ALFAS","ALGYO","ALKIM","ALKLC","ALMAD","AEFES","ANSGR","ARCLK","ARDYZ",
    "ARENA","ARKAS","ASELS","ASTOR","ASUZU","ATAGY","ATAKP","ATEKS","AVGYO","AVHOL",
    "AVOD","AYCES","AYEN","AYGAZ","AZGYO","BAGFS","BAKAB","BANVT","BASGZ","BERA",
    "BFREN","BIENY","BIGCH","BIMAS","BINBN","BIOEN","BIRGY","BIZIM","BJKAS","BKFIN",
    "BNTAS","BOBET","BOSSA","BRISA","BRKSN","BRLSM","BRSAN","BRYAT","BSOKE","BTCIM",
    "BUCIM","BURCE","BURVA","BVSAN","CCOLA","CEMAS","CEMTS","CEOMT","CIMSA","CLEBI",
    "CMBTN","CMENT","CONSE","COSMO","CRFSA","CUSAN","DAGHL","DARDL","DENGE","DESA",
    "DEVA","DGATE","DGKLB","DITAS","DMSAS","DOAS","DOBUR","DOCO","DOFER","DOGUB",
]

BIST100_YF = [t + ".IS" for t in BIST100_TICKERS]

# ── VERİ ÇEKME ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers: list, period: str, interval: str) -> dict:
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 30:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
        except Exception:
            pass
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(period: str, interval: str) -> pd.DataFrame:
    df = yf.download("XU100.IS", period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

# ── İNDİKATÖRLER ─────────────────────────────────────────────────────────────
def sma(s, n):   return s.rolling(n).mean()
def ema(s, n):   return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def stoch(h, l, c, k=14, smooth=3, d=3):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    raw = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    ks = raw.rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def macd(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow); sg = ema(m, sig)
    return m, sg, m - sg

def atr(h, l, c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_ratio(v, n=20):
    avg = v.rolling(n).mean().iloc[-1]
    return v.iloc[-1] / avg if avg and not np.isnan(avg) else np.nan

def rs_vs_bm(c, bm_c, days=63):
    sr = c.pct_change(days).iloc[-1]
    br = bm_c.pct_change(days).iloc[-1]
    return ((1+sr)/(1+br) - 1) if br and not np.isnan(br) else np.nan

def prox_52w(c):
    hi = c.rolling(min(252, len(c))).max().iloc[-1]
    return ((c.iloc[-1] / hi) - 1) * 100

# ── SKORLAMA ──────────────────────────────────────────────────────────────────
def score_rs(df, bm_df):
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();  v = df['Volume'].squeeze()
    bm = bm_df['Close'].squeeze()

    s50 = sma(c, 50).iloc[-1]
    k, d_ = stoch(h, lo, c)
    k_now, k_prev = k.iloc[-1], k.iloc[-2]
    vr = vol_ratio(v)
    rs = rs_vs_bm(c, bm)
    price = c.iloc[-1]

    criteria = {
        "SMA 50 Üzerinde":         (price > s50,                          f"{price:.2f} > {s50:.2f}"),
        "Stoch < 40'tan Yukarı Dön":(k_now > k_prev and k_prev < 40,     f"%K={k_now:.1f} (önceki {k_prev:.1f})"),
        "Hacim Ort. Üstünde":      (not np.isnan(vr) and vr > 1.0,       f"{vr:.2f}x"),
        "RS > BIST100 (3 ay)":     (rs is not None and not np.isnan(rs) and rs > 0,
                                    f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A"),
    }
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "SMA 50": f"{s50:.2f}",
        "Stoch %K": f"{k_now:.1f}",
        "Stoch %D": f"{d_.iloc[-1]:.1f}",
        "Hacim Oranı": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        "RS (3 ay)": f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A",
    }
    sc = sum(1 for v2, _ in criteria.values() if v2)
    return sc, len(criteria), criteria, details

def score_momentum(df, bm_df):
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();  v = df['Volume'].squeeze()

    _, _, hist = macd(c)
    a14 = atr(h, lo, c)
    atr_now = a14.iloc[-1]; atr_avg = a14.rolling(60).mean().iloc[-1]
    prox = prox_52w(c)
    hn, hp = hist.iloc[-1], hist.iloc[-2]
    price = c.iloc[-1]
    vr = vol_ratio(v)
    s50 = sma(c, 50).iloc[-1]
    s200 = sma(c, 200).iloc[-1]

    criteria = {
        "SMA 200 Üzerinde":             (price > s200,                   f"{price:.2f} > {s200:.2f}"),
        "SMA 50 Üzerinde":              (price > s50,                    f"{price:.2f} > {s50:.2f}"),
        "52H Yükseğe Yakın (≥ -5%)":   (prox >= -5,                     f"{prox:.1f}%"),
        "MACD Hist Pozitif & Artıyor":  (hn > 0 and hn > hp,            f"{hn:.4f}"),
        "ATR Genişliyor":               (atr_now > atr_avg,              f"{atr_now:.2f} > ort {atr_avg:.2f}"),
        "Hacim Ort. Üstünde (>1.2x)":  (not np.isnan(vr) and vr > 1.2, f"{vr:.2f}x"),
    }
    _, sig_l, hist_l = macd(c)
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "SMA 50": f"{s50:.2f}",
        "SMA 200": f"{s200:.2f}",
        "52H Yakınlık": f"{prox:.1f}%",
        "MACD": f"{(ema(c,12)-ema(c,26)).iloc[-1]:.4f}",
        "MACD Hist": f"{hn:.4f}",
        "ATR (14)": f"{atr_now:.2f}",
        "ATR Ort (60)": f"{atr_avg:.2f}",
        "Hacim Oranı": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
    }
    sc = sum(1 for v2, _ in criteria.values() if v2)
    return sc, len(criteria), criteria, details

# ── GRAFİK ───────────────────────────────────────────────────────────────────
def build_chart(df, ticker, strategy, interval):
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();  v = df['Volume'].squeeze()
    o = df['Open'].squeeze()

    s50 = sma(c, 50); s200 = sma(c, 200)
    k_line, d_line = stoch(h, lo, c)
    ml, sl, hl = macd(c)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.50, 0.18, 0.18, 0.14],
                        vertical_spacing=0.03)

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=lo, close=c,
        name="Fiyat",
        increasing_fillcolor="#22c55e", increasing_line_color="#22c55e",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
        line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50",
        line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s200, name="SMA 200",
        line=dict(color="#818cf8", width=1.5, dash="dash")), row=1, col=1)
    if strategy == "momentum":
        win = min(252, len(c))
        fig.add_trace(go.Scatter(x=df.index, y=c.rolling(win).max(),
            name="52H Yüksek", line=dict(color="#f97316", width=1, dash="longdash")), row=1, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=k_line, name="%K",
        line=dict(color="#38bdf8", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=d_line, name="%D",
        line=dict(color="#f97316", width=1.2, dash="dot")), row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.07, row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="#22c55e", opacity=0.07, row=2, col=1)
    fig.add_hline(y=80, line=dict(color="#ef4444", width=0.5, dash="dot"), row=2, col=1)
    fig.add_hline(y=20, line=dict(color="#22c55e", width=0.5, dash="dot"), row=2, col=1)

    # MACD
    bar_colors = ["#22c55e" if x >= 0 else "#ef4444" for x in hl]
    fig.add_trace(go.Bar(x=df.index, y=hl, name="Histogram",
        marker_color=bar_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ml, name="MACD",
        line=dict(color="#38bdf8", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sl, name="Signal",
        line=dict(color="#f97316", width=1.2, dash="dot")), row=3, col=1)

    # Hacim
    vcols = ["#22c55e" if cv >= ov else "#ef4444" for cv, ov in zip(c, o)]
    fig.add_trace(go.Bar(x=df.index, y=v, name="Hacim",
        marker_color=vcols, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=v.rolling(20).mean(), name="Hacim MA20",
        line=dict(color="#f59e0b", width=1.2)), row=4, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker.replace('.IS','')}</b>  —  {interval}",
                   font=dict(color="#f1f5f9", size=14), x=0.01),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1e2535", showgrid=True, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2535", showgrid=True, zeroline=False, row=i, col=1)
    return fig

# ── DETAY PANELİ ─────────────────────────────────────────────────────────────
def render_detail(result, strategy, interval, mode="single"):
    """Grafik + kriter kartları. mode='single': tam ekran / mode='search': içinde"""
    ticker_label = result["ticker"].replace(".IS", "")
    st.markdown(f"### {ticker_label}")

    criteria = result["criteria"]; details = result["details"]
    c1, c2 = st.columns(2)
    for i, (k, (passed, val_str)) in enumerate(criteria.items()):
        icon = "✅" if passed else "❌"
        color = "metric-pass" if passed else "metric-fail"
        html = f"""<div class="metric-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value {color}">{icon} {val_str}</div>
        </div>"""
        (c1 if i % 2 == 0 else c2).markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
    dc = st.columns(3)
    for i, (k, v) in enumerate(details.items()):
        dc[i % 3].markdown(
            f'<div class="metric-card"><div class="metric-label">{k}</div>'
            f'<div class="metric-value" style="font-size:0.95rem">{v}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Grafik</div>', unsafe_allow_html=True)
    fig = build_chart(result["df"], result["ticker"], strategy, interval)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">📊 BIST 100 Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y %H:%M")} · {len(BIST100_TICKERS)} hisse · yfinance</div>',
            unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")

    interval = st.selectbox("Zaman Dilimi", ["1d", "4h", "1wk"],
                            format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    period_map = {"1d": "6mo", "4h": "60d", "1wk": "2y"}
    period = period_map[interval]

    st.markdown("---")
    st.markdown("### 🔍 Hisse Ara")
    search_query = st.text_input("Ticker gir (örn: GARAN)", placeholder="GARAN, THYAO...").upper().strip()
    search_btn = st.button("Ara", use_container_width=True)

    st.markdown("---")
    st.markdown("**Rölatif Güç**: SMA50 · Stoch < 40 dönüş · RS > BIST100 · Hacim")
    st.markdown("**Momentum Kırılım**: SMA200 · SMA50 · 52H yakın · MACD artan · ATR genişleme · Hacim")

# ── STRATEJİ BUTONLARI ───────────────────────────────────────────────────────
col_b1, col_b2, col_b3 = st.columns([2.2, 2.2, 4])
with col_b1:
    btn_rs = st.button("🔵 Rölatif Güç Rotasyonu", use_container_width=True, type="primary")
with col_b2:
    btn_mo = st.button("🟣 Momentum Kırılımcısı", use_container_width=True)

if "strategy" not in st.session_state: st.session_state.strategy = None
if "selected_ticker" not in st.session_state: st.session_state.selected_ticker = None
if "scan_done" not in st.session_state: st.session_state.scan_done = False
if "results" not in st.session_state: st.session_state.results = []

if btn_rs:
    st.session_state.strategy = "rs"
    st.session_state.selected_ticker = None
    st.session_state.scan_done = False
if btn_mo:
    st.session_state.strategy = "momentum"
    st.session_state.selected_ticker = None
    st.session_state.scan_done = False

# ── ARAMA MODU ───────────────────────────────────────────────────────────────
if search_btn and search_query:
    ticker_yf = search_query + ".IS"
    st.markdown("---")
    st.markdown(f"## 🔍 {search_query} — İkili Analiz")

    with st.spinner(f"{search_query} ve BIST100 verisi çekiliyor..."):
        single_data = fetch_data([ticker_yf], period=period, interval=interval)
        bm_df = fetch_benchmark(period=period, interval=interval)

    if ticker_yf not in single_data:
        st.error(f"❌ {search_query} için veri çekilemedi. Ticker'ı kontrol et.")
    else:
        df = single_data[ticker_yf]

        # Her iki stratejiyi paralel göster
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🔵 Rölatif Güç Kriterleri")
            try:
                sc, mx, crit, det = score_rs(df, bm_df)
                score_color = "#22c55e" if sc == mx else ("#f59e0b" if sc >= mx-1 else "#ef4444")
                st.markdown(f"**Skor:** <span style='color:{score_color};font-size:1.2rem;font-weight:700'>{sc}/{mx}</span>", unsafe_allow_html=True)
                for k, (passed, val) in crit.items():
                    icon = "✅" if passed else "❌"
                    color = "metric-pass" if passed else "metric-fail"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{k}</div>'
                        f'<div class="metric-value {color}">{icon} {val}</div></div>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Hata: {e}")

        with col_right:
            st.markdown("### 🟣 Momentum Kırılım Kriterleri")
            try:
                sc2, mx2, crit2, det2 = score_momentum(df, bm_df)
                score_color2 = "#22c55e" if sc2 == mx2 else ("#f59e0b" if sc2 >= mx2-1 else "#ef4444")
                st.markdown(f"**Skor:** <span style='color:{score_color2};font-size:1.2rem;font-weight:700'>{sc2}/{mx2}</span>", unsafe_allow_html=True)
                for k, (passed, val) in crit2.items():
                    icon = "✅" if passed else "❌"
                    color = "metric-pass" if passed else "metric-fail"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{k}</div>'
                        f'<div class="metric-value {color}">{icon} {val}</div></div>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Hata: {e}")

        # Grafik — aktif stratejiye göre, yoksa "rs" default
        active_strat = st.session_state.strategy or "rs"
        st.markdown("---")
        st.markdown(f"**Grafik · {interval.upper()}** — SMA 50 🟡 / SMA 200 🟣")
        fig = build_chart(df, ticker_yf, active_strat, interval)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.stop()

# ── ANA TARAMA AKIŞI ─────────────────────────────────────────────────────────
if st.session_state.strategy is None:
    st.markdown("---")
    st.info("👆 Bir strateji seç ve taramayı başlat.")
    st.stop()

strategy = st.session_state.strategy
strategy_label = "Rölatif Güç Rotasyonu" if strategy == "rs" else "Momentum Kırılımcısı"
st.markdown(f"**Aktif:** `{strategy_label}` · `{interval.upper()}` · `{len(BIST100_TICKERS)} hisse`")
st.markdown("---")

tara_btn = st.button("🔍 Tara — Tüm BIST 100", type="primary")

if tara_btn or not st.session_state.scan_done:
    with st.spinner(f"BIST 100 taranıyor ({len(BIST100_TICKERS)} hisse, {interval})..."):
        bm_df = fetch_benchmark(period=period, interval=interval)
        stock_data = fetch_data(BIST100_YF, period=period, interval=interval)
        st.session_state.benchmark_df = bm_df

    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 30:
            continue
        try:
            if strategy == "rs":
                sc, mx, crit, det = score_rs(df, bm_df)
            else:
                sc, mx, crit, det = score_momentum(df, bm_df)
            results.append({"ticker": ticker, "score": sc, "max_score": mx,
                            "criteria": crit, "details": det, "df": df})
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.results = results
    st.session_state.scan_done = True

results = st.session_state.get("results", [])
if not results:
    st.warning("Veri yok. Tara butonuna bas.")
    st.stop()

strategy = st.session_state.strategy
pass_r = [r for r in results if r["score"] == r["max_score"]]
near_r = [r for r in results if r["score"] < r["max_score"]]  # hepsi

# ── LİSTE + DETAY ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="large")

with left_col:
    st.markdown(f'<div class="section-title">✅ Tüm Kriterleri Geçenler ({len(pass_r)})</div>',
                unsafe_allow_html=True)
    if not pass_r:
        st.markdown("*Şu an geçen hisse yok.*")
    for r in pass_r:
        lbl = r["ticker"].replace(".IS", "")
        if st.button(f"✅ {lbl}  {r['score']}/{r['max_score']}", key=f"p_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

    st.markdown(f'<div class="section-title">⚠️ Kriterlere Yaklaşanlar — Tüm Liste ({len(near_r)})</div>',
                unsafe_allow_html=True)
    if not near_r:
        st.markdown("*Hepsi zaten geçiyor!*")
    for r in near_r:
        lbl = r["ticker"].replace(".IS", "")
        miss = r["max_score"] - r["score"]
        emoji = "🟡" if miss == 1 else "🟠" if miss == 2 else "🔴"
        if st.button(f"{emoji} {lbl}  {r['score']}/{r['max_score']}", key=f"n_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

with right_col:
    sel = st.session_state.get("selected_ticker")
    if sel is None:
        st.markdown("### ← Soldan bir hisse seç")
        st.markdown("Tıkladığında grafik ve kriter detayları burada görünür.")
        if pass_r:
            st.markdown(f"**{len(pass_r)}** hisse tüm kriterleri geçiyor, **{len(near_r)}** hisse yaklaşıyor.")
    else:
        sel_result = next((r for r in results if r["ticker"] == sel), None)
        if sel_result is None:
            st.warning("Hisse verisi bulunamadı.")
        else:
            render_detail(sel_result, strategy, interval)
