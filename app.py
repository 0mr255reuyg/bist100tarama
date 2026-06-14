import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BIST 100 Strateji Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0d0f14; color: #e2e8f0; }

    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #f1f5f9;
        letter-spacing: -0.03em; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .strategy-btn-active {
        background: linear-gradient(135deg, #1e3a5f, #0f4c75) !important;
        border: 1px solid #3b82f6 !important;
        color: #93c5fd !important;
        border-radius: 8px; padding: 0.6rem 1.2rem;
        font-weight: 600; font-size: 0.85rem;
    }

    .metric-card {
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.7rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.1rem; font-weight: 600; color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-pass { color: #22c55e !important; }
    .metric-fail { color: #ef4444 !important; }
    .metric-warn { color: #f59e0b !important; }

    .stock-row-pass {
        background: #0d1f0d; border: 1px solid #166534;
        border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem;
        cursor: pointer; transition: all 0.15s;
    }
    .stock-row-near {
        background: #1c1709; border: 1px solid #92400e;
        border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem;
        cursor: pointer;
    }

    .section-title {
        font-size: 0.75rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        margin: 1.2rem 0 0.6rem 0; border-bottom: 1px solid #1e2535;
        padding-bottom: 0.4rem;
    }

    .score-badge {
        display: inline-block; padding: 0.15rem 0.5rem;
        border-radius: 4px; font-size: 0.7rem; font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .score-high { background: #14532d; color: #4ade80; }
    .score-mid  { background: #451a03; color: #fb923c; }
    .score-low  { background: #1c1917; color: #78716c; }

    div[data-testid="stButton"] button {
        border-radius: 8px; font-weight: 500; font-size: 0.85rem;
        transition: all 0.15s;
    }

    .stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# --- BIST 100 Ticker Listesi ---
BIST100 = [
    "AKBNK","AKFEN","AKSA","AKSEN","AEFES","ARCLK","ASELS","ASTOR","BIMAS","BRSAN",
    "CCOLA","CIMSA","DOHOL","EKGYO","ENKAI","EREGL","FROTO","GARAN","GUBRF","HALKB",
    "ISCTR","KARSN","KCHOL","KORDS","KOZAA","KOZAL","KRDMD","MAVI","MGROS","ODAS",
    "OTKAR","OYAKC","PETKM","PGSUS","PTOFS","SAHOL","SASA","SELEC","SISE","SKBNK",
    "SOKM","TAVHL","TCELL","THYAO","TKFEN","TOASO","TSKB","TTKOM","TUPRS","TURSG",
    "ULKER","VAKBN","VESTL","YKBNK","ZOREN","AGHOL","ALARK","ALBRK","ALCAR","ALGYO",
    "ALKIM","ANSGR","APCOA","ARKAS","ATAGY","ATAKP","BFREN","BIRGY","BIZIM","BNTAS",
    "BOBET","BRYAT","BSOKE","BTCIM","BUCIM","CEMTS","CEOMT","CIMSA","CLEBI","CMBTN",
    "CMENT","CONSE","CRFSA","CUSAN","DAGHL","DESA","DGKLB","DMSAS","DOAS","DOKTA",
    "DURDO","DYOBY","ECILC","EGEEN","EGGUB","EGPRO","EGSER","ELITE","EMKEL","EMNIS",
]
# Duplikaları temizle, .IS ekle
BIST100 = list(dict.fromkeys(BIST100))
BIST100_YF = [t + ".IS" for t in BIST100]
BIST100_YF = BIST100_YF[:60]  # İlk 60 ile çalış (hız için)

# --- Veri Çekme ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, period="6mo", interval="1d"):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 50:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
        except Exception:
            pass
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(period="6mo", interval="1d"):
    df = yf.download("XU100.IS", period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

# --- İndikatör Hesaplama ---
def calc_sma(series, n):
    return series.rolling(n).mean()

def calc_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def calc_rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch(high, low, close, k=14, d=3, smooth=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    k_raw = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    k_smooth = k_raw.rolling(smooth).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth, d_line

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd = ema_fast - ema_slow
    sig = calc_ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def calc_atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_rs(close, benchmark_close, period=63):
    """Rölatif güç: hissenin son N günlük getirisi / endeksin getirisi"""
    stock_ret = close.pct_change(period).iloc[-1]
    bm_ret = benchmark_close.pct_change(period).iloc[-1]
    if bm_ret == 0 or np.isnan(bm_ret):
        return np.nan
    return (1 + stock_ret) / (1 + bm_ret) - 1  # RS farkı

def calc_52w_proximity(close):
    """52 haftanın yüksek seviyesine yakınlık (%)"""
    window = min(252, len(close))
    high_52w = close.rolling(window).max().iloc[-1]
    current = close.iloc[-1]
    return ((current / high_52w) - 1) * 100  # negatif = yüksekten ne kadar uzakta

def calc_vol_ratio(volume, n=20):
    """Son günün hacmi / N günlük ortalama hacim"""
    avg = volume.rolling(n).mean().iloc[-1]
    last = volume.iloc[-1]
    if avg == 0 or np.isnan(avg):
        return np.nan
    return last / avg

# --- Strateji Skorlama ---
def score_relative_strength(df, benchmark_df):
    """Rölatif Güç Rotasyonu kriterleri"""
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()
    bm    = benchmark_df['Close'].squeeze()

    sma50  = calc_sma(close, 50).iloc[-1]
    k, d   = calc_stoch(high, low, close)
    k_now  = k.iloc[-1]
    d_now  = d.iloc[-1]
    k_prev = k.iloc[-2]
    vol_r  = calc_vol_ratio(vol)
    rs     = calc_rs(close, bm, 63)  # 3 aylık ~63 gün
    price  = close.iloc[-1]

    criteria = {
        "SMA50 Üzerinde":      (price > sma50,         f"{price:.2f} > {sma50:.2f}"),
        "Stoch < 40'tan Dönüş":(k_now > k_prev and k_prev < 40, f"%K={k_now:.1f} (önceki: {k_prev:.1f})"),
        "Hacim Oranı > 1.0":   (vol_r > 1.0,           f"{vol_r:.2f}x"),
        "RS Pozitif (BIST100+)":(rs is not None and not np.isnan(rs) and rs > 0,
                                 f"{rs*100:.1f}% fark" if rs is not None and not np.isnan(rs) else "N/A"),
    }
    score = sum(1 for v, _ in criteria.values() if v)
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "SMA 50": f"{sma50:.2f}",
        "Stoch %K": f"{k_now:.1f}",
        "Stoch %D": f"{d_now:.1f}",
        "Hacim Oranı": f"{vol_r:.2f}x",
        "RS (3 ay)": f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A",
    }
    return score, len(criteria), criteria, details

def score_momentum_breakout(df, benchmark_df):
    """Momentum Kırılımcısı kriterleri"""
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()

    macd, sig, hist = calc_macd(close)
    atr14   = calc_atr(high, low, close)
    atr_now = atr14.iloc[-1]
    atr_avg = atr14.rolling(60).mean().iloc[-1]
    prox    = calc_52w_proximity(close)
    hist_now  = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    price     = close.iloc[-1]
    vol_r     = calc_vol_ratio(vol)
    sma50     = calc_sma(close, 50).iloc[-1]

    criteria = {
        "52H Yükseğe Yakın (≥-5%)":(prox >= -5,        f"{prox:.1f}%"),
        "MACD Hist Pozitif & Artıyor": (hist_now > 0 and hist_now > hist_prev,
                                         f"{hist_now:.4f}"),
        "ATR Genişliyor":          (atr_now > atr_avg,  f"{atr_now:.2f} > avg {atr_avg:.2f}"),
        "Hacim Oranı > 1.2":       (vol_r > 1.2,        f"{vol_r:.2f}x"),
    }
    score = sum(1 for v, _ in criteria.values() if v)
    macd_v, sig_v = macd.iloc[-1], sig.iloc[-1]
    details = {
        "Fiyat": f"{price:.2f} ₺",
        "52H Yakınlık": f"{prox:.1f}%",
        "MACD": f"{macd_v:.4f}",
        "MACD Signal": f"{sig_v:.4f}",
        "MACD Hist": f"{hist_now:.4f}",
        "ATR (14)": f"{atr_now:.2f}",
        "ATR Ort.": f"{atr_avg:.2f}",
        "Hacim Oranı": f"{vol_r:.2f}x",
        "SMA 50": f"{sma50:.2f}",
    }
    return score, len(criteria), criteria, details

# --- Grafik ---
def build_chart(df, ticker, strategy, interval):
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()
    opens = df['Open'].squeeze()

    sma50 = calc_sma(close, 50)
    k, d  = calc_stoch(high, low, close)
    macd_line, sig_line, hist_line = calc_macd(close)
    rsi14 = calc_rsi(close)
    atr14 = calc_atr(high, low, close)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        vertical_spacing=0.03,
        subplot_titles=["", "Stochastic (14,3,3)", "MACD (12,26,9)", "Hacim"]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=opens, high=high, low=low, close=close,
        name="Fiyat",
        increasing_fillcolor="#22c55e", increasing_line_color="#22c55e",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
        line=dict(width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50",
                             line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)

    # 52H yüksek çizgisi (Kırılımcı için)
    if strategy == "momentum":
        window = min(252, len(close))
        high52 = close.rolling(window).max()
        fig.add_trace(go.Scatter(x=df.index, y=high52, name="52H Yüksek",
                                 line=dict(color="#818cf8", width=1, dash="dash")), row=1, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=k, name="%K",
                             line=dict(color="#38bdf8", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=d, name="%D",
                             line=dict(color="#f97316", width=1.2, dash="dot")), row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.07, row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="#22c55e", opacity=0.07, row=2, col=1)
    fig.add_hline(y=80, line=dict(color="#ef4444", width=0.5, dash="dot"), row=2, col=1)
    fig.add_hline(y=20, line=dict(color="#22c55e", width=0.5, dash="dot"), row=2, col=1)

    # MACD
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist_line]
    fig.add_trace(go.Bar(x=df.index, y=hist_line, name="Histogram",
                         marker_color=colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                             line=dict(color="#38bdf8", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sig_line, name="Signal",
                             line=dict(color="#f97316", width=1.2, dash="dot")), row=3, col=1)

    # Hacim
    vol_colors = ["#22c55e" if c >= o else "#ef4444"
                  for c, o in zip(close, opens)]
    fig.add_trace(go.Bar(x=df.index, y=vol, name="Hacim",
                         marker_color=vol_colors, opacity=0.6), row=4, col=1)
    # Hacim MA
    vol_ma = vol.rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, name="Hacim MA20",
                             line=dict(color="#f59e0b", width=1.2)), row=4, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker.replace('.IS','')}</b>  —  {interval}",
                   font=dict(color="#f1f5f9", size=14), x=0.01),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#1e2535", showgrid=True,
            zeroline=False, row=i, col=1
        )
        fig.update_yaxes(
            gridcolor="#1e2535", showgrid=True,
            zeroline=False, row=i, col=1
        )

    return fig

# ===================== STREAMLIT UI =====================
st.markdown('<div class="main-header">📊 BIST 100 Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Son güncelleme: {datetime.now().strftime("%d.%m.%Y %H:%M")} · yfinance</div>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d", "1wk"], index=0,
                            format_func=lambda x: "Günlük (1D)" if x == "1d" else "Haftalık (1W)")
    period_map = {"1d": "6mo", "1wk": "2y"}
    period = period_map[interval]
    top_n = st.slider("Tarama Limiti (ticker sayısı)", 20, 60, 40, 5)
    st.markdown("---")
    st.markdown("**Strateji Açıklamaları**")
    st.markdown("🔵 **Rölatif Güç**: SMA50 + Stoch dönüş + RS > BIST100 + Hacim")
    st.markdown("🟣 **Momentum Kırılım**: 52H yakın + MACD artan + ATR genişleme + Hacim")

# Strateji seçim butonları
col_b1, col_b2, col_b3 = st.columns([2, 2, 5])
with col_b1:
    btn_rs = st.button("🔵 Rölatif Güç Rotasyonu", use_container_width=True, type="primary")
with col_b2:
    btn_mo = st.button("🟣 Momentum Kırılımcısı", use_container_width=True)

# Strateji state
if "strategy" not in st.session_state:
    st.session_state.strategy = None
if btn_rs:
    st.session_state.strategy = "rs"
    st.session_state.selected_ticker = None
    st.session_state.scan_done = False
if btn_mo:
    st.session_state.strategy = "momentum"
    st.session_state.selected_ticker = None
    st.session_state.scan_done = False

if st.session_state.strategy is None:
    st.markdown("---")
    st.info("👆 Yukarıdan bir strateji seç ve taramayı başlat.")
    st.stop()

strategy = st.session_state.strategy
strategy_label = "Rölatif Güç Rotasyonu" if strategy == "rs" else "Momentum Kırılımcısı"
st.markdown(f"**Aktif Strateji:** `{strategy_label}` · Zaman dilimi: `{interval.upper()}`")
st.markdown("---")

# Tarama
tickers_to_scan = BIST100_YF[:top_n]

if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

do_scan = st.button("🔍 Tara", type="primary") or not st.session_state.get("scan_done", False)

if do_scan:
    with st.spinner(f"BIST 100 taranıyor ({len(tickers_to_scan)} hisse)..."):
        benchmark_df = fetch_benchmark(period=period, interval=interval)
        stock_data   = fetch_data(tickers_to_scan, period=period, interval=interval)

    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 60:
            continue
        try:
            if strategy == "rs":
                score, max_score, criteria, details = score_relative_strength(df, benchmark_df)
            else:
                score, max_score, criteria, details = score_momentum_breakout(df, benchmark_df)
            results.append({
                "ticker": ticker,
                "score": score,
                "max_score": max_score,
                "criteria": criteria,
                "details": details,
                "df": df,
            })
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.results = results
    st.session_state.scan_done = True
    st.session_state.benchmark_df = benchmark_df

results = st.session_state.get("results", [])
if not results:
    st.warning("Veri çekilemedi. Biraz bekleyip tekrar dene.")
    st.stop()

pass_results = [r for r in results if r["score"] == r["max_score"]]
near_results = [r for r in results if r["score"] == r["max_score"] - 1][:10]

# Ana layout: sol liste, sağ detay
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    # GEÇEN HİSSELER
    st.markdown(f'<div class="section-title">✅ Tüm Kriterleri Geçenler ({len(pass_results)})</div>',
                unsafe_allow_html=True)
    if not pass_results:
        st.markdown("*Şu an tüm kriterleri geçen hisse yok.*")
    for r in pass_results:
        label = r["ticker"].replace(".IS", "")
        score_html = f'<span class="score-badge score-high">{r["score"]}/{r["max_score"]}</span>'
        clicked = st.button(f"✅ {label}  {r['score']}/{r['max_score']}", key=f"pass_{r['ticker']}",
                            use_container_width=True)
        if clicked:
            st.session_state.selected_ticker = r["ticker"]

    # YAKLAŞANLAR
    st.markdown(f'<div class="section-title">⚠️ En Yakın Olanlar ({len(near_results)})</div>',
                unsafe_allow_html=True)
    if not near_results:
        st.markdown("*Yakın hisse yok.*")
    for r in near_results:
        label = r["ticker"].replace(".IS", "")
        clicked = st.button(f"⚠️ {label}  {r['score']}/{r['max_score']}", key=f"near_{r['ticker']}",
                            use_container_width=True)
        if clicked:
            st.session_state.selected_ticker = r["ticker"]

with right_col:
    sel = st.session_state.get("selected_ticker", None)
    if sel is None:
        st.markdown("### ← Soldan bir hisse seç")
        st.markdown("Hisseye tıkladığında grafik ve kriter detayları burada görünecek.")
    else:
        # Seçili hissenin datasını bul
        sel_result = next((r for r in results if r["ticker"] == sel), None)
        if sel_result is None:
            st.warning("Hisse verisi bulunamadı.")
        else:
            ticker_label = sel.replace(".IS", "")
            st.markdown(f"### {ticker_label}")

            # Üst satır: kriter kartları
            criteria = sel_result["criteria"]
            details  = sel_result["details"]
            score    = sel_result["score"]
            max_s    = sel_result["max_score"]

            # Kriterler (2 sütun)
            crit_keys = list(criteria.keys())
            c1, c2 = st.columns(2)
            for i, k in enumerate(crit_keys):
                passed, val_str = criteria[k]
                icon  = "✅" if passed else "❌"
                color = "metric-pass" if passed else "metric-fail"
                card_html = f"""
                <div class="metric-card">
                    <div class="metric-label">{k}</div>
                    <div class="metric-value {color}">{icon} {val_str}</div>
                </div>"""
                if i % 2 == 0:
                    c1.markdown(card_html, unsafe_allow_html=True)
                else:
                    c2.markdown(card_html, unsafe_allow_html=True)

            # Sayısal detaylar (küçük)
            st.markdown('<div class="section-title">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
            det_cols = st.columns(3)
            det_items = list(details.items())
            for i, (k, v) in enumerate(det_items):
                det_cols[i % 3].markdown(
                    f'<div class="metric-card"><div class="metric-label">{k}</div>'
                    f'<div class="metric-value" style="font-size:0.95rem">{v}</div></div>',
                    unsafe_allow_html=True
                )

            # Grafik
            st.markdown('<div class="section-title">📈 Grafik</div>', unsafe_allow_html=True)
            fig = build_chart(sel_result["df"], sel, strategy, interval)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
