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

st.set_page_config(page_title="BIST Strateji Tarayıcı", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background-color:#0d0f14;color:#e2e8f0;}
.main-header{font-size:1.8rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.03em;}
.sub-header{font-size:.82rem;color:#64748b;margin-bottom:1.2rem;font-family:'JetBrains Mono',monospace;}
.mc{background:#161b27;border:1px solid #1e2535;border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.5rem;}
.ml{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace;margin-bottom:.15rem;}
.mv{font-size:1rem;font-weight:600;color:#f1f5f9;font-family:'JetBrains Mono',monospace;}
.pass{color:#22c55e!important;} .fail{color:#ef4444!important;}
.sec{font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.1em;
  font-family:'JetBrains Mono',monospace;margin:1rem 0 .5rem 0;border-bottom:1px solid #1e2535;padding-bottom:.3rem;}
.stag{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.65rem;
  font-family:'JetBrains Mono',monospace;background:#1e2535;color:#94a3b8;margin-left:.3rem;}

/* UI DÜZELTMESİ: Beyaz zemin üstündeki buton yazılarının simsiyah ve okunaklı olması için */
div[data-testid="stButton"] button {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
}
div[data-testid="stButton"] button p {
    color: #000000 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}
div[data-testid="stButton"] button:hover {
    background-color: #e2e8f0 !important;
    border-color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

BIST100_TICKERS = BIST100_OFFICIAL
BIST100_YF = [t+".IS" for t in BIST100_TICKERS]

# ── VERİ ÇEKME ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(tickers, period, interval):
    data = {}
    progress = st.progress(0, "Veri çekiliyor...")
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, timeout=10)
            if df is not None and len(df) >= 22:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                for col in ['Open','High','Low','Close','Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Close'])
                if len(df) >= 22:
                    data[ticker] = df
        except Exception:
            pass
        if i % 10 == 0:
            progress.progress(min((i+1)/len(tickers), 1.0), f"{i+1}/{len(tickers)} hisse")
    progress.empty()
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, timeout=10)
        if df is not None and len(df) > 10:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            for col in ['Open','High','Low','Close','Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=['Close'])
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark(period, interval):
    df = yf.download("XU100.IS", period=period, interval=interval,
                     auto_adjust=True, progress=False, timeout=10)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

# ── İNDİKATÖRLER ─────────────────────────────────────────────────────────────
def _c(df): return df['Close'].squeeze().dropna()
def _h(df): return df['High'].squeeze().dropna()
def _l(df): return df['Low'].squeeze().dropna()
def _v(df): return df['Volume'].squeeze().dropna()

def sma(s, n):
    if len(s) < n: return pd.Series([np.nan]*len(s), index=s.index)
    return s.rolling(n).mean()

def ema(s, n):
    if len(s) < n: return pd.Series([np.nan]*len(s), index=s.index)
    return s.ewm(span=n, adjust=False).mean()

def rsi_calc(s, n=14):
    if len(s) < n+1: return pd.Series([50.0]*len(s), index=s.index)
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def stoch(h, l, c, k=14, smooth=3, d=3):
    if len(c) < k: return pd.Series([50.0]*len(c), index=c.index), pd.Series([50.0]*len(c), index=c.index)
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    raw = 100*(c-lo)/(hi-lo).replace(0, np.nan)
    ks = raw.rolling(smooth).mean().fillna(50)
    return ks, ks.rolling(d).mean().fillna(50)

def macd_calc(s, fast=12, slow=26, sig=9):
    m = ema(s,fast)-ema(s,slow)
    sg = ema(m.fillna(0), sig)
    return m, sg, m-sg

def atr_calc(h, l, c, n=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_ratio(v, n=20):
    if len(v) < n+1: return np.nan
    avg = float(v.iloc[-n-1:-1].mean())
    return float(v.iloc[-1])/avg if avg > 0 else np.nan

def rs_score(c, bm_c, days=20):
    if len(c) < days+1 or len(bm_c) < days+1: return np.nan
    common = c.index.intersection(bm_c.index)
    if len(common) < days+1: return np.nan
    c2 = c.loc[common]; bm2 = bm_c.loc[common]
    sr = float(c2.iloc[-1]/c2.iloc[-days]) - 1
    br = float(bm2.iloc[-1]/bm2.iloc[-days]) - 1
    return sr - br

# ── STRATEJİLER ───────────────────────────────────────────────────────────────
def score_emre(df, bm_df):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df)
        if len(c) < 22: return 0,4,{},{}
        bm = _c(bm_df) if not bm_df.empty else pd.Series(dtype=float)

        price = float(c.iloc[-1])
        s20 = float(sma(c,20).iloc[-1])
        s50 = float(sma(c,50).iloc[-1]) if len(c)>=50 else np.nan
        rsi_v = float(rsi_calc(c).iloc[-1])
        rs = rs_score(c, bm, 20) if len(bm) > 20 else np.nan

        vr = vol_ratio(v, 20)
        vol5 = float(v.iloc[-5:].mean()) if len(v) >= 5 else np.nan
        vol20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else np.nan

        criteria = {
            "RS Pozitif (20g vs BIST)":    (not np.isnan(rs) and rs > 0,
                                             f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A"),
            "SMA20 & SMA50 Üstünde":       (price > s20 and (np.isnan(s50) or price > s50),
                                             f"{price:.2f} > {s20:.2f}" + (f" / {s50:.2f}" if not np.isnan(s50) else "")),
            "Hacim Ortalamanın Üstünde":   (not np.isnan(vr) and vr >= 0.9,
                                             f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
            "RSI < 80":                    (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        details = {
            "Fiyat": f"{price:.2f} ₺",
            "SMA 20": f"{s20:.2f}",
            "SMA 50": f"{s50:.2f}" if not np.isnan(s50) else "N/A",
            "RSI (14)": f"{rsi_v:.1f}",
            "RS (20g)": f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A",
            "Hacim Oran.": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        }
        sc = sum(1 for p,_ in criteria.values() if p)
        return sc, 4, criteria, details
    except Exception as e:
        return 0, 4, {"Hata": (False, str(e))}, {}

def score_momentum(df, bm_df):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df)
        if len(c) < 26: return 0,4,{},{}

        price = float(c.iloc[-1])
        s50 = float(sma(c,50).iloc[-1]) if len(c)>=50 else np.nan
        ml, sl, hl = macd_calc(c)
        hn = float(hl.dropna().iloc[-1]) if len(hl.dropna())>1 else 0
        hp = float(hl.dropna().iloc[-2]) if len(hl.dropna())>1 else 0
        a14 = atr_calc(h,l,c)
        atr_now = float(a14.dropna().iloc[-1]) if len(a14.dropna())>0 else np.nan
        atr_avg = float(a14.rolling(60).mean().dropna().iloc[-1]) if len(a14.dropna())>=60 else np.nan
        vr = vol_ratio(v, 20)

        criteria = {
            "SMA 50 Üzerinde":             (not np.isnan(s50) and price > s50,
                                             f"{price:.2f} > {s50:.2f}" if not np.isnan(s50) else "N/A"),
            "MACD Hist Pozitif & Artıyor": (hn > 0 and hn > hp,
                                             f"{hn:.4f}"),
            "ATR Genişliyor":              (not np.isnan(atr_avg) and atr_now > atr_avg,
                                             f"{atr_now:.2f} > {atr_avg:.2f}" if not np.isnan(atr_avg) else "N/A"),
            "Hacim > 1.2x Ort.":          (not np.isnan(vr) and vr > 1.2,
                                             f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
        }
        details = {
            "Fiyat": f"{price:.2f} ₺",
            "SMA 50": f"{s50:.2f}" if not np.isnan(s50) else "N/A",
            "MACD Hist": f"{hn:.4f}",
            "ATR (14)": f"{atr_now:.2f}" if not np.isnan(atr_now) else "N/A",
            "ATR Ort": f"{atr_avg:.2f}" if not np.isnan(atr_avg) else "N/A",
            "Hacim Oran.": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        }
        sc = sum(1 for p,_ in criteria.values() if p)
        return sc, 4, criteria, details
    except Exception as e:
        return 0, 4, {"Hata": (False, str(e))}, {}

STRATEGY_FN = {
    "emre":     (score_emre,     "Emre'nin Stratejisi"),
    "momentum": (score_momentum, "Momentum Kırılımcısı"),
}

# ── GRAFİK ────────────────────────────────────────────────────────────────────
def build_chart(df, ticker, strategy, interval):
    c = _c(df); h = _h(df); l = _l(df); v = _v(df); o = df['Open'].squeeze()

    s20 = sma(c,20); s50 = sma(c,50)
    k_line, d_line = stoch(h,l,c)
    ml, sl, hl_s = macd_calc(c)
    rsi_line = rsi_calc(c)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48,0.18,0.18,0.16], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c,
        name="Fiyat", increasing_fillcolor="#22c55e", increasing_line_color="#22c55e",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
        line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s20, name="SMA 20",
        line=dict(color="#22c55e",width=1.5,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50",
        line=dict(color="#f59e0b",width=1.5,dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi_line, name="RSI",
        line=dict(color="#38bdf8",width=1.5)), row=2, col=1)
    fig.add_hrect(y0=80,y1=100,fillcolor="#ef4444",opacity=0.07,row=2,col=1)
    fig.add_hline(y=80,line=dict(color="#ef4444",width=0.7,dash="dot"),row=2,col=1)
    fig.add_hline(y=50,line=dict(color="#64748b",width=0.5,dash="dot"),row=2,col=1)

    bar_colors = ["#22c55e" if x>=0 else "#ef4444" for x in hl_s.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=hl_s, name="MACD Hist",
        marker_color=bar_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ml, name="MACD",
        line=dict(color="#38bdf8",width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sl, name="Signal",
        line=dict(color="#f97316",width=1.2,dash="dot")), row=3, col=1)

    vcols = ["#22c55e" if float(cv)>=float(ov) else "#ef4444"
             for cv,ov in zip(c.values, o.reindex(c.index).fillna(c).values)]
    fig.add_trace(go.Bar(x=df.index, y=v, name="Hacim",
        marker_color=vcols, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=v.rolling(20).mean(), name="Hacim MA20",
        line=dict(color="#f59e0b",width=1.2)), row=4, col=1)

    fig.update_layout(height=700, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        legend=dict(orientation="h",y=1.02,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker.replace('.IS','')}</b>  —  {interval.upper()}",
                   font=dict(color="#f1f5f9",size=14),x=0.01),
        hovermode="x unified")
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#1e2535",showgrid=True,zeroline=False,row=i,col=1)
        fig.update_yaxes(gridcolor="#1e2535",showgrid=True,zeroline=False,row=i,col=1)
    return fig

# ── DETAY PANELİ ──────────────────────────────────────────────────────────────
def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS","")
    sector = get_sector(result["ticker"])

    st.markdown(f"### {ticker_label} "
                f'<span class="stag">{sector}</span> '
                f'`{result["score"]}/{result["max_score"]}`',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"
        col="pass" if passed else "fail"
        html=(f'<div class="mc"><div class="ml">{k}</div>'
              f'<div class="mv {col}">{icon} {val}</div></div>')
        (c1 if i%2==0 else c2).markdown(html, unsafe_allow_html=True)

    if result["details"]:
        st.markdown('<div class="sec">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
        dc = st.columns(3)
        for i,(k,v) in enumerate(result["details"].items()):
            dc[i%3].markdown(
                f'<div class="mc"><div class="ml
