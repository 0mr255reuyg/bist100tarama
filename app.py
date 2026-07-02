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
from sectors import (get_sector, get_theme_sectors, get_theme_info,
                     MACRO_THEMES, MACRO_THEMES_PRIMARY, MACRO_THEMES_SECONDARY,
                     SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
from sector_summary import build_summary, build_sector_bar_chart, _sector_returns

# ── 1. MINIMAL UI AYARLARI ───────────────────────────────────────────────────
st.set_page_config(page_title="BIST Makro Strateji", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0a0a; color: #ededed; }

/* Header & Typography */
.main-header { font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.04em; margin-bottom: 0.2rem; }
.sub-header { font-size: 0.85rem; color: #888888; margin-bottom: 2rem; font-family: 'JetBrains Mono', monospace; }
.sec-title { font-size: 1.1rem; font-weight: 600; color: #a3a3a3; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #222; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0; }

/* Minimalist Cards */
.mc { background: transparent; border: 1px solid #262626; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; transition: border-color 0.2s ease; }
.mc:hover { border-color: #404040; }
.ml { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
.mv { font-size: 1.1rem; font-weight: 600; color: #fff; font-family: 'JetBrains Mono', monospace; }

/* Status Colors */
.pass { color: #10b981 !important; } 
.fail { color: #ef4444 !important; }
.stag { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; background: #171717; border: 1px solid #262626; color: #a3a3a3; margin-left: 0.5rem; }

/* Premium Buttons */
div[data-testid="stButton"] button { background-color: #171717 !important; border: 1px solid #333 !important; border-radius: 6px !important; color: #fff !important; transition: all 0.2s ease; }
div[data-testid="stButton"] button p { font-weight: 600 !important; font-size: 0.95rem !important; }
div[data-testid="stButton"] button:hover { background-color: #262626 !important; border-color: #555 !important; }
div[data-testid="stButton"] button:focus { box-shadow: none !important; color: #fff !important; }

/* Primary Button Override */
div[data-testid="stButton"] button[data-testid="baseButton-primary"] { background-color: #ededed !important; color: #0a0a0a !important; border: none !important; }
div[data-testid="stButton"] button[data-testid="baseButton-primary"] p { color: #0a0a0a !important; font-weight: 700 !important; }
div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover { background-color: #ffffff !important; opacity: 0.9; }

/* Inputs */
.stTextInput input { background-color: #171717 !important; color: #fff !important; border: 1px solid #333 !important; border-radius: 6px !important; }
.stSelectbox div[data-baseweb="select"] { background-color: #171717 !important; border: 1px solid #333 !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

BIST100_YF = [t+".IS" for t in BIST100_OFFICIAL]

# ── 2. VERİ ÇEKME ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(tickers, period, interval):
    data = {}
    progress = st.progress(0, "Hisse verileri çekiliyor...")
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, timeout=5)
            if df is not None and len(df) >= 55:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                for col in ['Open','High','Low','Close','Volume']:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Close'])
                if len(df) >= 55: data[ticker] = df
        except: pass
        if i % 10 == 0: progress.progress(min((i+1)/len(tickers), 1.0), f"{i+1}/{len(tickers)} hisse")
    progress.empty()
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark(period, interval):
    df = yf.download("XU100.IS", period=period, interval=interval, auto_adjust=True, progress=False, timeout=5)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

# ── TLREF / FAİZ MOTORU VERİSİ (PROXY) ──
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_macro_rate():
    # TLREF proxy'si olarak şimdilik test amaçlı yfinance tahvil datası kullanılıyor. 
    # İleride EVDS API ile "TP.DK.TLREF.O.KT" çekilecek.
    df = yf.download("^TNX", period="5y", interval="1wk", auto_adjust=True, progress=False)
    if df is not None and len(df) > 55:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

# ── 3. MATEMATİK & İNDİKATÖRLER ───────────────────────────────────────────────
def _c(df): return df['Close'].squeeze().dropna()
def _h(df): return df['High'].squeeze().dropna()
def _l(df): return df['Low'].squeeze().dropna()
def _v(df): return df['Volume'].squeeze().dropna()

def sma(s, n): return s.rolling(n).mean() if len(s) >= n else pd.Series([np.nan]*len(s), index=s.index)
def ema(s, n): return s.ewm(span=n, adjust=False).mean() if len(s) >= n else pd.Series([np.nan]*len(s), index=s.index)

def rsi_calc(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def macd_calc(s, fast=12, slow=26, sig=9):
    m = ema(s,fast)-ema(s,slow); sg = ema(m.fillna(0), sig)
    return m, sg, m-sg

def atr_calc(h, l, c, n=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_ratio(v, n=20):
    if len(v) < n+1: return np.nan
    avg = float(v.iloc[-n-1:-1].mean())
    return float(v.iloc[-1])/avg if avg > 0 else np.nan

def rs_score(c, bm_c, days=20):
    common = c.index.intersection(bm_c.index)
    if len(common) < days+1: return np.nan
    c2 = c.loc[common]; bm2 = bm_c.loc[common]
    return (float(c2.iloc[-1]/c2.iloc[-days]) - 1) - (float(bm2.iloc[-1]/bm2.iloc[-days]) - 1)

def calc_adx(h, l, c, n=14):
    up = h.diff(); down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=h.index).ewm(span=n, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=h.index).ewm(span=n, adjust=False).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=n, adjust=False).mean()
    return adx, plus_di, minus_di

# ── 4. MAKRO REJİM MOTORU ─────────────────────────────────────────────────────
def get_macro_regime():
    df_rate = fetch_macro_rate()
    if df_rate.empty:
        return "Bilinmiyor", []
    
    c = _c(df_rate); h = _h(df_rate); l = _l(df_rate)
    sma8 = float(sma(c, 8).iloc[-1])
    sma54 = float(sma(c, 54).iloc[-1])
    adx, plus_di, minus_di = calc_adx(h, l, c)
    
    adx_val = float(adx.iloc[-1])
    p_di = float(plus_di.iloc[-1])
    m_di = float(minus_di.iloc[-1])

    # 3'LÜ REJİM MANTIĞI
    if sma8 > sma54 and p_di > m_di and adx_val > 25:
        return "Savunmacı (Risk Off)", ["Gıda ve Perakende", "İletişim", "Sağlık"]
    elif sma8 < sma54:
        return "Büyüme (Risk On)", ["Teknoloji ve Yazılım", "Enerji", "Otomotiv", "Sanayi ve Kimya"]
    else:
        return "Denge (Plato)", ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm"]

CURRENT_REGIME, REGIME_SECTORS = get_macro_regime()

# ── 5. STRATEJİLER (5/5 MAKSİMUM SKOR) ────────────────────────────────────────
def score_emre(df, bm_df, ticker):
    try:
        c = _c(df); v = _v(df)
        if len(c) < 55: return 0, 5, {}, {}
        bm = _c(bm_df) if not bm_df.empty else pd.Series(dtype=float)

        price = float(c.iloc[-1])
        s20 = float(sma(c,20).iloc[-1])
        s50 = float(sma(c,50).iloc[-1])
        rsi_v = float(rsi_calc(c).iloc[-1])
        rs = rs_score(c, bm, 20)
        vr = vol_ratio(v, 20)

        sector = get_sector(ticker)
        is_macro_aligned = sector in REGIME_SECTORS

        criteria = {
            "RS Pozitif (20g vs BIST)": (rs > 0, f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A"),
            "SMA20 & SMA50 Üstünde": (price > s20 and price > s50, f"{price:.2f} > {s20:.2f}/{s50:.2f}"),
            "Hacim Ort. Üstünde": (vr >= 0.9, f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
            "RSI < 80": (rsi_v < 80, f"RSI={rsi_v:.1f}"),
            "Makro Rüzgar (+1)": (is_macro_aligned, f"{sector} ({CURRENT_REGIME})")
        }
        
        details = { "Fiyat": f"{price:.2f} ₺", "SMA 20/50": f"{s20:.2f} / {s50:.2f}", "RSI (14)": f"{rsi_v:.1f}", "Hacim Oran": f"{vr:.2f}x" }
        sc = sum(1 for p,_ in criteria.values() if p)
        return sc, 5, criteria, details
    except:
        return 0, 5, {}, {}

STRATEGY_FN = { "emre": (score_emre, "Emre'nin Makro Stratejisi") }

# ── 6. UI YAPI & GRAFİK ───────────────────────────────────────────────────────
def build_chart(df, ticker, interval):
    c = _c(df); h = _h(df); l = _l(df); o = df['Open'].squeeze()
    s20 = sma(c,20); s50 = sma(c,50)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="Fiyat", increasing_fillcolor="#10b981", increasing_line_color="#10b981", decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s20, name="SMA 20", line=dict(color="#38bdf8",width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50", line=dict(color="#f59e0b",width=1.5)), row=1, col=1)

    vcols = ["#10b981" if float(cv)>=float(ov) else "#ef4444" for cv,ov in zip(c.values, o.reindex(c.index).fillna(c).values)]
    fig.add_trace(go.Bar(x=df.index, y=_v(df), name="Hacim", marker_color=vcols, opacity=0.5), row=2, col=1)

    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="JetBrains Mono", color="#a3a3a3", size=11), margin=dict(l=0,r=0,t=30,b=0), xaxis_rangeslider_visible=False, title=dict(text=f"<b>{ticker.replace('.IS','')}</b>", font=dict(color="#fff",size=14), x=0.01), hovermode="x unified")
    fig.update_xaxes(gridcolor="#1e1e1e", showgrid=True); fig.update_yaxes(gridcolor="#1e1e1e", showgrid=True)
    return fig

# ── SAYFA YÜKLEME ──
st.markdown('<div class="main-header">BIST Makro Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y")} · Minimal Sürüm</div>', unsafe_allow_html=True)

# ── SESSION STATE ──
defaults = {"strategy":"emre","selected_ticker":None,"scan_done":False,"results":[],"page":"scanner"}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── SOL MENÜ (SIDEBAR) ──
with st.sidebar:
    st.markdown("### 🔍 Hızlı İnceleme")
    search_input = st.text_input("Hisse Kodu", placeholder="Örn: GARAN").upper().strip()
    if st.button("Hisse Getir", use_container_width=True):
        st.session_state.page = "search"; st.session_state.search_ticker = search_input; st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🌐 Makro Rejim Paneli")
    rc = "#10b981" if "On" in CURRENT_REGIME else "#ef4444" if "Off" in CURRENT_REGIME else "#f59e0b"
    st.markdown(f"<div style='border:1px solid #333; padding:15px; border-radius:8px; background:#111; text-align:center;'><h4 style='margin:0; color:#888; font-size:0.8rem; font-weight:normal;'>Güncel Piyasa Rejimi</h4><h2 style='margin:5px 0 0 0; color:{rc}; font-size:1.4rem;'>{CURRENT_REGIME}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("<br><p style='font-size:0.8rem; color:#888; margin-bottom:5px;'>Öne Çıkan Sektörler:</p>", unsafe_allow_html=True)
    for s in REGIME_SECTORS:
        st.markdown(f"<div class='stag' style='margin-bottom:4px; width:100%; text-align:left;'>✦ {s}</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d","1wk"], format_func=lambda x: "Günlük" if x=="1d" else "Haftalık")
    period = "6mo" if interval == "1d" else "2y"
    
    if st.button("📊 Portföy Backtest", use_container_width=True): st.session_state.page = "perf"; st.rerun()

# ── ANA SAYFA: TARAMA ────────────────────────────────────────────────────────
if st.session_state.page == "scanner":
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Makro Tarama Başlat", type="primary"):
            st.session_state.scan_done = False
            st.session_state.results = []
            st.session_state.selected_ticker = None

    if not st.session_state.scan_done:
        with st.spinner("Piyasa verileri işleniyor..."):
            bm_df = fetch_benchmark(period, interval)
            stock_data = fetch_data(BIST100_YF, period, interval)
            results = []
            for ticker, df in stock_data.items():
                if df is None or len(df)<55: continue
                sc, mx, crit, det = score_emre(df, bm_df, ticker)
                c = _c(df); rs = rs_score(c, _c(bm_df), 20)
                results.append({ "ticker": ticker, "score": sc, "max_score": mx, "criteria": crit, "details": det, "df": df, "rs": rs if not np.isnan(rs) else -999, "sector": get_sector(ticker), "in_top5": False })
            
            # Puan ve RS sırası
            results.sort(key=lambda x: (x['score'], x['rs']), reverse=True)
            top5_count = 0; sector_counts = {}
            for r in results:
                sect = r['sector']
                if top5_count < 5 and sector_counts.get(sect, 0) < 2:
                    r['in_top5'] = True
                    sector_counts[sect] = sector_counts.get(sect, 0) + 1
                    top5_count += 1

            st.session_state.results = results
            st.session_state.scan_done = True
            st.rerun()

    results = st.session_state.results
    if results:
        top5_r = [r for r in results if r.get('in_top5')]
        
        st.markdown('<div class="sec-title">⭐ Makro Destekli Top 5</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, r in enumerate(top5_r):
            with cols[i]:
                lbl = r["ticker"].replace(".IS","")
                if st.button(f"{lbl} ({r['score']}/5)\n{r['sector'][:12]}..", key=f"t5_{lbl}", use_container_width=True):
                    st.session_state.selected_ticker = r["ticker"]
                    st.rerun()
        
        st.markdown("---")
        if st.session_state.selected_ticker:
            sel_r = next(r for r in results if r["ticker"] == st.session_state.selected_ticker)
            c1, c2 = st.columns([1, 2.5])
            with c1:
                st.markdown(f"### {sel_r['ticker'].replace('.IS','')} <span class='stag'>{sel_r['sector']}</span>", unsafe_allow_html=True)
                for k,(passed,val) in sel_r["criteria"].items():
                    icon, col = ("✅", "pass") if passed else ("❌", "fail")
                    st.markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv {col}">{icon} {val}</div></div>', unsafe_allow_html=True)
            with c2:
                st.plotly_chart(build_chart(sel_r["df"], sel_r["ticker"], interval), use_container_width=True, config={"displayModeBar":False})

# (Diğer sayfalar - Search, Perf - modüler olarak bağlanacak)
elif st.session_state.page == "search":
    st.info("Arama sayfası aktif. Buraya detay modülü gelecek.")
    if st.button("Geri Dön"): st.session_state.page="scanner"; st.rerun()

elif st.session_state.page == "perf":
    st.info("Backtest motoru güncelleniyor. Lütfen 3. aşamayı bekleyin.")
    if st.button("Geri Dön"): st.session_state.page="scanner"; st.rerun()
