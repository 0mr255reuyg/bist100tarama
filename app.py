import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from backtest_engine import run_backtest, calc_stats, build_perf_chart, STRAT_LABELS, STRAT_COLORS

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
    .stSelectbox label { color: #94a3b8 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── GERÇEK BIST 100 LİSTESİ ──────────────────────────────────────────────────
BIST100_TICKERS = [
    "ACSEL","AEFES","AGESA","AGHOL","AKBNK","AKCNS","AKGRT","AKSA","AKSEN","AKSGY",
    "ALARK","ALBRK","ALFAS","ALGYO","ALKIM","ANACM","ANSGR","ARCLK","ARDYZ","ARENA",
    "ASELS","ASTOR","ASUZU","ATAGY","AYGAZ","BAGFS","BERA","BIMAS","BIOEN","BIRGY",
    "BIZIM","BNTAS","BOBET","BRISA","BRSAN","BRYAT","BSOKE","BTCIM","BUCIM","BURCE",
    "CCOLA","CEMTS","CIMSA","CLEBI","CMBTN","CMENT","CONSE","CRFSA","DOAS","DOHOL",
    "ECILC","EGEEN","EKGYO","ENKAI","EREGL","FROTO","GARAN","GUBRF","HALKB","ISCTR",
    "ISGYO","ISGSY","KARSN","KCHOL","KLGYO","KONTR","KORDS","KOZAA","KOZAL","KRDMD",
    "LOGO","MAVI","MGROS","NETAS","ODAS","OTKAR","OYAKC","PGSUS","PETKM","PTOFS",
    "SAHOL","SASA","SELEC","SISE","SKBNK","SOKM","TAVHL","TCELL","THYAO","TKFEN",
    "TOASO","TSKB","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VESTL","YKBNK",
]

BIST100_YF = [t + ".IS" for t in BIST100_TICKERS]

# ── VERİ ÇEKME ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, period, interval):
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
def fetch_benchmark(period, interval):
    df = yf.download("XU100.IS", period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

# ── İNDİKATÖRLER ─────────────────────────────────────────────────────────────
def sma(s, n):  return s.rolling(n).mean()
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def rsi_calc(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def stoch(h, l, c, k=14, smooth=3, d=3):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    raw = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    ks = raw.rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def macd_calc(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow); sg = ema(m, sig)
    return m, sg, m - sg

def atr_calc(h, l, c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_ratio(v, n=20):
    avg = v.rolling(n).mean().iloc[-1]
    last = v.iloc[-1]
    return last / avg if avg and not np.isnan(avg) and avg > 0 else np.nan

def rs_vs_bm(c, bm_c, days=63):
    if len(c) < days or len(bm_c) < days: return np.nan
    sr = c.pct_change(days).iloc[-1]
    br = bm_c.pct_change(days).iloc[-1]
    if np.isnan(br) or br == 0: return np.nan
    return ((1+sr)/(1+br) - 1)

# ── SKORLAMA FONKSİYONLARI ───────────────────────────────────────────────────

def score_rs(df, bm_df):
    """Rölatif Güç Rotasyonu — RSI 50-70 filtresi eklendi, hacim eşiği 0.99"""
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze(); v = df['Volume'].squeeze()
    bm = bm_df['Close'].squeeze()

    s50 = sma(c, 50).iloc[-1]
    k_line, d_line = stoch(h, lo, c)
    k_now = k_line.iloc[-1]; k_prev = k_line.iloc[-2]
    vr = vol_ratio(v)
    rs = rs_vs_bm(c, bm)
    rsi_val = rsi_calc(c).iloc[-1]
    price = c.iloc[-1]

    criteria = {
        "SMA 50 Üzerinde":           (price > s50,
                                       f"{price:.2f} > {s50:.2f}"),
        "RSI 50–70 Arasında":        (50 <= rsi_val <= 70,
                                       f"RSI={rsi_val:.1f}"),
        "Stoch < 40'tan Yukarı Dön": (k_now > k_prev and k_prev < 40,
                                       f"%K={k_now:.1f} (önceki {k_prev:.1f})"),
        "Hacim ≥ 0.99x Ort.":        (not np.isnan(vr) and vr >= 0.99,
                                       f"{vr:.2f}x"),
        "RS > BIST100 (3 ay)":       (rs is not None and not np.isnan(rs) and rs > 0,
                                       f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A"),
    }
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "SMA 50": f"{s50:.2f}",
        "RSI (14)": f"{rsi_val:.1f}",
        "Stoch %K": f"{k_now:.1f}",
        "Stoch %D": f"{d_line.iloc[-1]:.1f}",
        "Hacim Oranı": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        "RS (3 ay)": f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A",
    }
    sc = sum(1 for passed, _ in criteria.values() if passed)
    return sc, len(criteria), criteria, details


def score_momentum(df, bm_df):
    """Momentum Kırılımcısı — SMA200 ve 52H kaldırıldı"""
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze(); v = df['Volume'].squeeze()

    s50 = sma(c, 50).iloc[-1]
    ml, sl, hl = macd_calc(c)
    hn = hl.iloc[-1]; hp = hl.iloc[-2]
    a14 = atr_calc(h, lo, c)
    atr_now = a14.iloc[-1]
    atr_avg = a14.rolling(60).mean().iloc[-1]
    vr = vol_ratio(v)
    price = c.iloc[-1]
    rsi_val = rsi_calc(c).iloc[-1]

    criteria = {
        "SMA 50 Üzerinde":              (price > s50,
                                          f"{price:.2f} > {s50:.2f}"),
        "MACD Hist Pozitif & Artıyor":  (hn > 0 and hn > hp,
                                          f"{hn:.4f}"),
        "ATR Genişliyor":               (not np.isnan(atr_avg) and atr_now > atr_avg,
                                          f"{atr_now:.2f} > ort {atr_avg:.2f}"),
        "Hacim > 1.2x Ort.":           (not np.isnan(vr) and vr > 1.2,
                                          f"{vr:.2f}x"),
    }
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "SMA 50": f"{s50:.2f}",
        "RSI (14)": f"{rsi_val:.1f}",
        "MACD": f"{ml.iloc[-1]:.4f}",
        "MACD Signal": f"{sl.iloc[-1]:.4f}",
        "MACD Hist": f"{hn:.4f}",
        "ATR (14)": f"{atr_now:.2f}",
        "ATR Ort (60)": f"{atr_avg:.2f}" if not np.isnan(atr_avg) else "N/A",
        "Hacim Oranı": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
    }
    sc = sum(1 for passed, _ in criteria.values() if passed)
    return sc, len(criteria), criteria, details


def score_trend(df, bm_df):
    """Trend Sürücüsü — EMA 20/50/200 hizalanmış + RSI 50-65 + Hacim"""
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze(); v = df['Volume'].squeeze()

    e20 = ema(c, 20).iloc[-1]
    e50 = ema(c, 50).iloc[-1]
    e200 = ema(c, 200).iloc[-1]
    rsi_val = rsi_calc(c).iloc[-1]
    vr = vol_ratio(v)
    price = c.iloc[-1]

    # Ay kapanışı EMA20 üstünde (son değeri kullan)
    above_e20 = price > e20

    criteria = {
        "EMA 20 > EMA 50 > EMA 200":  (e20 > e50 > e200,
                                         f"{e20:.2f} > {e50:.2f} > {e200:.2f}"),
        "Fiyat EMA 20 Üstünde":        (above_e20,
                                         f"{price:.2f} > {e20:.2f}"),
        "RSI 50–65 Arasında":          (50 <= rsi_val <= 65,
                                         f"RSI={rsi_val:.1f}"),
        "Hacim ≥ 1.2x Ort.":          (not np.isnan(vr) and vr >= 1.2,
                                         f"{vr:.2f}x"),
    }
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "EMA 20": f"{e20:.2f}",
        "EMA 50": f"{e50:.2f}",
        "EMA 200": f"{e200:.2f}" if not np.isnan(e200) else "N/A",
        "RSI (14)": f"{rsi_val:.1f}",
        "Hacim Oranı": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
    }
    sc = sum(1 for passed, _ in criteria.values() if passed)
    return sc, len(criteria), criteria, details


STRATEGY_FN = {
    "rs":       (score_rs,       "Rölatif Güç Rotasyonu"),
    "momentum": (score_momentum, "Momentum Kırılımcısı"),
    "trend":    (score_trend,    "Trend Sürücüsü"),
}

# ── GRAFİK ───────────────────────────────────────────────────────────────────
def build_chart(df, ticker, strategy, interval):
    c = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze(); v = df['Volume'].squeeze()
    o = df['Open'].squeeze()

    s50 = sma(c, 50)
    e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
    k_line, d_line = stoch(h, lo, c)
    ml, sl, hl_s = macd_calc(c)
    rsi_line = rsi_calc(c)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48, 0.18, 0.18, 0.16],
                        vertical_spacing=0.03)

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=lo, close=c,
        name="Fiyat",
        increasing_fillcolor="#22c55e", increasing_line_color="#22c55e",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
        line=dict(width=1)), row=1, col=1)

    if strategy == "trend":
        fig.add_trace(go.Scatter(x=df.index, y=e20, name="EMA 20",
            line=dict(color="#22c55e", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=e50, name="EMA 50",
            line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=e200, name="EMA 200",
            line=dict(color="#818cf8", width=1.5, dash="dash")), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50",
            line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)

    # RSI (row 2 — Stoch yerine RSI göster trend'de)
    if strategy == "trend":
        fig.add_trace(go.Scatter(x=df.index, y=rsi_line, name="RSI",
            line=dict(color="#38bdf8", width=1.5)), row=2, col=1)
        fig.add_hrect(y0=65, y1=100, fillcolor="#ef4444", opacity=0.07, row=2, col=1)
        fig.add_hrect(y0=0, y1=50, fillcolor="#64748b", opacity=0.05, row=2, col=1)
        fig.add_hline(y=65, line=dict(color="#ef4444", width=0.7, dash="dot"), row=2, col=1)
        fig.add_hline(y=50, line=dict(color="#22c55e", width=0.7, dash="dot"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=k_line, name="%K",
            line=dict(color="#38bdf8", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=d_line, name="%D",
            line=dict(color="#f97316", width=1.2, dash="dot")), row=2, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.07, row=2, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="#22c55e", opacity=0.07, row=2, col=1)
        fig.add_hline(y=80, line=dict(color="#ef4444", width=0.5, dash="dot"), row=2, col=1)
        fig.add_hline(y=20, line=dict(color="#22c55e", width=0.5, dash="dot"), row=2, col=1)

    # MACD
    bar_colors = ["#22c55e" if x >= 0 else "#ef4444" for x in hl_s]
    fig.add_trace(go.Bar(x=df.index, y=hl_s, name="MACD Hist",
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

    strat_label = STRATEGY_FN[strategy][1]
    fig.update_layout(
        height=720, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker.replace('.IS','')}</b>  —  {interval.upper()}  ·  {strat_label}",
                   font=dict(color="#f1f5f9", size=14), x=0.01),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1e2535", showgrid=True, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2535", showgrid=True, zeroline=False, row=i, col=1)
    return fig

# ── DETAY PANELİ ─────────────────────────────────────────────────────────────
def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS", "")
    st.markdown(f"### {ticker_label}  `{result['score']}/{result['max_score']}`")
    c1, c2 = st.columns(2)
    for i, (k, (passed, val_str)) in enumerate(result["criteria"].items()):
        icon = "✅" if passed else "❌"
        color = "metric-pass" if passed else "metric-fail"
        html = (f'<div class="metric-card">'
                f'<div class="metric-label">{k}</div>'
                f'<div class="metric-value {color}">{icon} {val_str}</div></div>')
        (c1 if i % 2 == 0 else c2).markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
    dc = st.columns(3)
    for i, (k, v) in enumerate(result["details"].items()):
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
    search_query = st.text_input("Ticker (örn: GARAN)", placeholder="GARAN...").upper().strip()
    search_btn = st.button("Ara", use_container_width=True)

    st.markdown("---")
    perf_btn = st.button("📊 Performans", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**🔵 Rölatif Güç:** SMA50 · RSI 50-70 · Stoch dönüş · Hacim ≥0.99 · RS>BIST")
    st.markdown("**🟣 Momentum:** SMA50 · MACD artan · ATR genişleme · Hacim >1.2x")
    st.markdown("**🟢 Trend:** EMA 20>50>200 · Fiyat>EMA20 · RSI 50-65 · Hacim ≥1.2x")

# ── STRATEJİ BUTONLARI ───────────────────────────────────────────────────────
col_b1, col_b2, col_b3, col_b4 = st.columns([2, 2, 2, 3])
with col_b1:
    btn_rs   = st.button("🔵 Rölatif Güç", use_container_width=True, type="primary")
with col_b2:
    btn_mo   = st.button("🟣 Momentum", use_container_width=True)
with col_b3:
    btn_tr   = st.button("🟢 Trend Sürücüsü", use_container_width=True)

# Session state init
for key, default in [("strategy", None), ("selected_ticker", None),
                      ("scan_done", False), ("results", []),
                      ("page", "scanner"), ("bt_done", False), ("bt_results", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

if btn_rs:
    st.session_state.update(strategy="rs", selected_ticker=None, scan_done=False, page="scanner")
if btn_mo:
    st.session_state.update(strategy="momentum", selected_ticker=None, scan_done=False, page="scanner")
if btn_tr:
    st.session_state.update(strategy="trend", selected_ticker=None, scan_done=False, page="scanner")
if perf_btn:
    st.session_state.update(page="perf")

# ── PERFORMANS SAYFASI ───────────────────────────────────────────────────────
if st.session_state.page == "perf":
    st.markdown("## 📊 Strateji Performans Karşılaştırması")
    st.markdown("Her ayın ilk borsa günü tam puan alan hisseler alınır · Günlük · 2 Yıl · 100.000 ₺")
    st.markdown("---")

    run_bt_btn = st.button("▶️ Backtest'i Çalıştır (2 yıl, tüm BIST 100)", type="primary")

    if run_bt_btn or not st.session_state.bt_done:
        with st.spinner("2 yıllık veri çekiliyor (bu işlem 1-2 dakika sürebilir)..."):
            bm_df_bt   = fetch_benchmark("2y", "1d")
            stock_bt   = fetch_data(BIST100_YF, "2y", "1d")

        bt_results = {}
        strat_keys = ["rs", "momentum", "trend"]
        prog = st.progress(0, text="Backtest hesaplanıyor...")
        for i, sk in enumerate(strat_keys):
            prog.progress((i+1)/len(strat_keys), text=f"{STRAT_LABELS[sk]} hesaplanıyor...")
            pv, bm_n, trades, active, monthly = run_backtest(sk, stock_bt, bm_df_bt)
            bt_results[sk] = {
                "pv": pv, "bm": bm_n, "trades": trades, "active": active,
                "monthly": monthly,
                "stats": calc_stats(pv, bm_n, 100_000) if pv is not None else {}
            }
        prog.empty()
        st.session_state.bt_results = bt_results
        st.session_state.bt_done = True

    bt_results = st.session_state.bt_results
    if not bt_results:
        st.info("▶️ Backtest'i çalıştır butonuna bas.")
        st.stop()

    # ── AKTİF POZİSYONLAR (bu ayın hisseleri) ──────────────────────────────
    st.markdown("### 📌 Bu Ay Aktif Portföyler")
    col1, col2, col3 = st.columns(3)
    for col, sk in zip([col1, col2, col3], ["rs", "momentum", "trend"]):
        with col:
            label = STRAT_LABELS[sk]
            color = STRAT_COLORS[sk]
            st.markdown(f"**{label}**")
            active = bt_results[sk].get("active", [])
            if not active:
                st.markdown("*Şu an pozisyon yok*")
            else:
                for pos in sorted(active, key=lambda x: x['pnl_pct'], reverse=True):
                    pnl = pos['pnl_pct']
                    pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                    st.markdown(
                        f'<div class="metric-card" style="border-color:{color}33">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="font-weight:700;color:#f1f5f9;font-size:0.95rem">{pos["ticker"]}</span>'
                        f'<span style="color:{pnl_color};font-family:JetBrains Mono;font-weight:600">{pnl:+.1f}%</span>'
                        f'</div>'
                        f'<div style="font-size:0.72rem;color:#64748b;font-family:JetBrains Mono;margin-top:0.3rem">'
                        f'Alış: {pos["buy_date"]} · ₺{pos["buy_price"]:.2f} → ₺{pos["current_price"]:.2f}'
                        f'</div></div>',
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── PERFORMANS GRAFİĞİ ───────────────────────────────────────────────────
    st.markdown("### 📈 Portföy Performansı vs BIST 100")
    perf_map = {sk: (bt_results[sk]["pv"], bt_results[sk]["bm"])
                for sk in ["rs","momentum","trend"] if bt_results[sk]["pv"] is not None}
    if perf_map:
        fig_perf = build_perf_chart(perf_map, 100_000)
        st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})

    # ── İSTATİSTİK TABLOLARI ─────────────────────────────────────────────────
    st.markdown("### 📐 Özet İstatistikler")
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    for col, sk in zip([stat_col1, stat_col2, stat_col3], ["rs", "momentum", "trend"]):
        with col:
            label = STRAT_LABELS[sk]
            color = STRAT_COLORS[sk]
            st.markdown(f"**{label}**")
            stats = bt_results[sk].get("stats", {})
            for k, v in stats.items():
                is_pos = "+" in str(v) and "Alpha" not in k and "Drawdown" not in k
                is_neg = "-" in str(v) or "Drawdown" in k
                val_color = ("#22c55e" if is_pos else "#ef4444" if is_neg else "#f1f5f9")
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{k}</div>'
                    f'<div class="metric-value" style="color:{val_color};font-size:1rem">{v}</div>'
                    f'</div>', unsafe_allow_html=True)

    # ── AYLIK TABLO ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Aylık Portföy Detayı — Top 5 Hisseler")
    st.markdown("Her ay başında en yüksek puanlı 5 hisse · Skor = kaç kriter geçildi")
    monthly_tabs = st.tabs([STRAT_LABELS["rs"], STRAT_LABELS["momentum"], STRAT_LABELS["trend"]])
    for tab, sk in zip(monthly_tabs, ["rs", "momentum", "trend"]):
        with tab:
            mdf = bt_results[sk].get("monthly")
            if mdf is not None and len(mdf) > 0:
                # Aylık P&L renklendir
                def color_pnl(val):
                    if isinstance(val, str) and val.startswith('+'):
                        return 'color: #22c55e'
                    elif isinstance(val, str) and val.startswith('-'):
                        return 'color: #ef4444'
                    return ''
                styled = mdf.style.map(color_pnl, subset=['Aylık P&L'] if 'Aylık P&L' in mdf.columns else [])
                st.dataframe(styled, use_container_width=True, height=600)
            else:
                st.markdown("*Veri yok.*")

    # ── İŞLEM LOGU ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 İşlem Geçmişi")
    log_tabs = st.tabs([STRAT_LABELS["rs"], STRAT_LABELS["momentum"], STRAT_LABELS["trend"]])
    for tab, sk in zip(log_tabs, ["rs", "momentum", "trend"]):
        with tab:
            trades = bt_results[sk].get("trades")
            if trades is not None and len(trades) > 0:
                # Sadece satışları göster (P&L olan)
                sales = trades[trades["İşlem"] == "SATIŞ"] if "İşlem" in trades.columns else trades
                if len(sales) > 0:
                    st.dataframe(sales.reset_index(drop=True),
                                 use_container_width=True, height=300)
                else:
                    st.markdown("*Henüz kapanan işlem yok.*")
            else:
                st.markdown("*İşlem verisi yok.*")

    st.stop()

# ── ARAMA MODU ───────────────────────────────────────────────────────────────
if search_btn and search_query:
    ticker_yf = search_query + ".IS"
    st.markdown("---")
    st.markdown(f"## 🔍 {search_query} — Üç Strateji Analizi")

    with st.spinner(f"{search_query} verisi çekiliyor..."):
        single_data = fetch_data([ticker_yf], period=period, interval=interval)
        bm_df = fetch_benchmark(period=period, interval=interval)

    if ticker_yf not in single_data:
        st.error(f"❌ {search_query} için veri çekilemedi.")
    else:
        df = single_data[ticker_yf]
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        labels = ["🔵 Rölatif Güç", "🟣 Momentum Kırılım", "🟢 Trend Sürücüsü"]
        keys   = ["rs", "momentum", "trend"]

        for col, label, key in zip(cols, labels, keys):
            with col:
                st.markdown(f"### {label}")
                try:
                    fn = STRATEGY_FN[key][0]
                    sc, mx, crit, det = fn(df, bm_df)
                    color = "#22c55e" if sc == mx else ("#f59e0b" if sc >= mx-1 else "#ef4444")
                    st.markdown(f"**Skor:** <span style='color:{color};font-size:1.3rem;font-weight:700'>{sc}/{mx}</span>",
                                unsafe_allow_html=True)
                    for k, (passed, val) in crit.items():
                        icon = "✅" if passed else "❌"
                        c_cls = "metric-pass" if passed else "metric-fail"
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">{k}</div>'
                            f'<div class="metric-value {c_cls}">{icon} {val}</div></div>',
                            unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Hata: {e}")

        st.markdown("---")
        active = st.session_state.strategy or "rs"
        st.markdown(f"**Grafik · {interval.upper()}**")
        fig = build_chart(df, ticker_yf, active, interval)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.stop()

# ── ANA TARAMA ───────────────────────────────────────────────────────────────
if st.session_state.strategy is None:
    st.markdown("---")
    st.info("👆 Bir strateji seç ve taramayı başlat.")
    st.stop()

strategy = st.session_state.strategy
strategy_label = STRATEGY_FN[strategy][1]
score_fn       = STRATEGY_FN[strategy][0]

st.markdown(f"**Aktif:** `{strategy_label}` · `{interval.upper()}` · `{len(BIST100_TICKERS)} hisse`")
st.markdown("---")

tara_btn = st.button("🔍 Tara — Tüm BIST 100", type="primary")

if tara_btn or not st.session_state.scan_done:
    with st.spinner(f"BIST 100 taranıyor ({len(BIST100_TICKERS)} hisse, {interval})..."):
        bm_df      = fetch_benchmark(period=period, interval=interval)
        stock_data = fetch_data(BIST100_YF, period=period, interval=interval)
        st.session_state.benchmark_df = bm_df

    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 30: continue
        try:
            sc, mx, crit, det = score_fn(df, bm_df)
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

pass_r = [r for r in results if r["score"] == r["max_score"]]
near_r = [r for r in results if r["score"] < r["max_score"]]

# ── LİSTE + DETAY ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="large")

with left_col:
    st.markdown(f'<div class="section-title">✅ Tüm Kriterleri Geçenler ({len(pass_r)})</div>',
                unsafe_allow_html=True)
    if not pass_r:
        st.markdown("*Şu an geçen hisse yok.*")
    for r in pass_r:
        lbl = r["ticker"].replace(".IS", "")
        if st.button(f"✅ {lbl}  {r['score']}/{r['max_score']}",
                     key=f"p_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

    st.markdown(f'<div class="section-title">⚠️ Kriterlere Yaklaşanlar ({len(near_r)})</div>',
                unsafe_allow_html=True)
    for r in near_r:
        lbl = r["ticker"].replace(".IS", "")
        miss = r["max_score"] - r["score"]
        emoji = "🟡" if miss == 1 else "🟠" if miss == 2 else "🔴"
        if st.button(f"{emoji} {lbl}  {r['score']}/{r['max_score']}",
                     key=f"n_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

with right_col:
    sel = st.session_state.get("selected_ticker")
    if sel is None:
        st.markdown("### ← Soldan bir hisse seç")
        st.markdown(f"**{len(pass_r)}** hisse tüm kriterleri geçiyor · **{len(near_r)}** hisse yaklaşıyor.")
    else:
        sel_result = next((r for r in results if r["ticker"] == sel), None)
        if sel_result:
            render_detail(sel_result, strategy, interval)
        else:
            st.warning("Hisse bulunamadı.")
