import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Oturum (Session) değişkenlerinin en başta güvene alınması (NameError önlemi)
if 'interval' not in st.session_state: st.session_state.interval = "1d"
if 'period' not in st.session_state: st.session_state.period = "6mo"
if 'page' not in st.session_state: st.session_state.page = "scanner"
if 'strategy' not in st.session_state: st.session_state.strategy = "emre"
if 'scan_done' not in st.session_state: st.session_state.scan_done = False
if 'results' not in st.session_state: st.session_state.results = []
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None
if 'bt_done' not in st.session_state: st.session_state.bt_done = False
if 'bt_results' not in st.session_state: st.session_state.bt_results = {}

from backtest_engine import run_backtest, calc_stats, build_perf_chart, STRAT_LABELS, STRAT_COLORS
from sectors import (get_sector, get_tlref_macro_regime, update_bist100_and_sectors,
                     MANUAL_MACRO_THEMES, SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
from sector_summary import build_summary, build_sector_bar_chart, _sector_returns

SECTOR_MAP, BIST100_TICKERS = update_bist100_and_sectors()
BIST100_YF = [t+".IS" for t in BIST100_TICKERS]

st.set_page_config(page_title="BIST Strateji Tarayıcı", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

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
.sec{font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.1em; font-family:'JetBrains Mono',monospace;margin:1rem 0 .5rem 0;border-bottom:1px solid #1e2535;padding-bottom:.3rem;}
.stag{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.65rem; font-family:'JetBrains Mono',monospace;background:#1e2535;color:#94a3b8;margin-left:.3rem;}
div[data-testid="stButton"] button { background-color: #f8fafc !important; border: 1px solid #cbd5e1 !important; }
div[data-testid="stButton"] button p { color: #000000 !important; font-weight: 800 !important; font-size: 1.05rem !important; }
div[data-testid="stButton"] button:hover { background-color: #e2e8f0 !important; border-color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(tickers, period, interval):
    data = {}
    progress = st.progress(0, "Veri çekiliyor...")
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
            if df is not None and len(df) >= 22:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                for col in ['Open','High','Low','Close','Volume']:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Close'])
                if len(df) >= 22: data[ticker] = df
        except Exception: pass
        if i % 10 == 0: progress.progress(min((i+1)/len(tickers), 1.0), f"{i+1}/{len(tickers)} hisse")
    progress.empty()
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
        if df is not None and len(df) > 10:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            for col in ['Open','High','Low','Close','Volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=['Close'])
    except Exception: pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark(period, interval):
    df = yf.download("XU100.IS", period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

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

def score_emre(df, bm_df, ticker):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df)
        if len(c) < 22: return 0,5,{},{}
        bm = _c(bm_df) if not bm_df.empty else pd.Series(dtype=float)

        price = float(c.iloc[-1])
        s20 = float(sma(c,20).iloc[-1])
        s50 = float(sma(c,50).iloc[-1]) if len(c)>=50 else np.nan
        rsi_v = float(rsi_calc(c).iloc[-1])
        rs = rs_score(c, bm, 20) if len(bm) > 20 else np.nan
        vr = vol_ratio(v, 20)

        criteria = {
            "RS Pozitif (20g vs BIST)":    (not np.isnan(rs) and rs > 0, f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A"),
            "SMA20 & SMA50 Üstünde":       (price > s20 and (np.isnan(s50) or price > s50), f"{price:.2f} > {s20:.2f}"),
            "Hacim Ortalamanın Üstünde":   (not np.isnan(vr) and vr >= 0.9, f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
            "RSI < 80":                    (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        
        sc = sum(1 for p,_ in criteria.values() if p)
        mx = 4
        
        # Makro Filtre Katmanı
        macro = get_tlref_macro_regime()
        is_supported = get_sector(ticker) in macro["sectors"]
        criteria["Makro Rejim Uyum Desteği"] = (is_supported, "Uyumlu Sektör" if is_supported else "Uyumsuz Sektör")
        
        if is_supported:
            sc += 1
        mx += 1
        
        details = {
            "Fiyat": f"{price:.2f} ₺", "SMA 20": f"{s20:.2f}", "SMA 50": f"{s50:.2f}" if not np.isnan(s50) else "N/A",
            "RSI (14)": f"{rsi_v:.1f}", "RS (20g)": f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A", "Hacim Oran.": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        }
        return sc, mx, criteria, details
    except Exception as e:
        return 0, 5, {"Hata": (False, str(e))}, {}

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
            "SMA 50 Üzerinde":             (not np.isnan(s50) and price > s50, f"{price:.2f} > {s50:.2f}" if not np.isnan(s50) else "N/A"),
            "MACD Hist Pozitif & Artıyor": (hn > 0 and hn > hp, f"{hn:.4f}"),
            "ATR Genişliyor":              (not np.isnan(atr_avg) and atr_now > atr_avg, f"{atr_now:.2f} > {atr_avg:.2f}" if not np.isnan(atr_avg) else "N/A"),
            "Hacim > 1.2x Ort.":          (not np.isnan(vr) and vr > 1.2, f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
        }
        details = {
            "Fiyat": f"{price:.2f} ₺", "SMA 50": f"{s50:.2f}" if not np.isnan(s50) else "N/A", "MACD Hist": f"{hn:.4f}",
            "ATR (14)": f"{atr_now:.2f}" if not np.isnan(atr_now) else "N/A", "ATR Ort": f"{atr_avg:.2f}" if not np.isnan(atr_avg) else "N/A", "Hacim Oran.": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        }
        sc = sum(1 for p,_ in criteria.values() if p)
        return sc, 4, criteria, details
    except Exception as e:
        return 0, 4, {"Hata": (False, str(e))}, {}

# SYNTAX HATASI BURADA ÇÖZÜLDÜ
STRATEGY_FN = {
    "emre": (lambda df, bm: score_emre(df, bm, st.session_state.get('active_scanning_ticker', '')), "Emre'nin Makro Stratejisi"),
    "momentum": (lambda df, bm: score_momentum(df, bm), "Momentum Kırılımcısı")
}

def build_chart(df, ticker, strategy, interval):
    c = _c(df); h = _h(df); l = _l(df); v = _v(df); o = df['Open'].squeeze()
    s20 = sma(c,20); s50 = sma(c,50)
    ml, sl, hl_s = macd_calc(c)
    rsi_line = rsi_calc(c)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48,0.18,0.18,0.16], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="Fiyat", increasing_fillcolor="#22c55e", increasing_line_color="#22c55e", decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s20, name="SMA 20", line=dict(color="#22c55e",width=1.5,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50", line=dict(color="#f59e0b",width=1.5,dash="dash")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=rsi_line, name="RSI", line=dict(color="#38bdf8",width=1.5)), row=2, col=1)
    fig.add_hrect(y0=80,y1=100,fillcolor="#ef4444",opacity=0.07,row=2,col=1)
    fig.add_hline(y=80,line=dict(color="#ef4444",width=0.7,dash="dot"),row=2,col=1)
    fig.add_hline(y=50,line=dict(color="#64748b",width=0.5,dash="dot"),row=2,col=1)

    bar_colors = ["#22c55e" if x>=0 else "#ef4444" for x in hl_s.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=hl_s, name="MACD Hist", marker_color=bar_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ml, name="MACD", line=dict(color="#38bdf8",width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sl, name="Signal", line=dict(color="#f97316",width=1.2,dash="dot")), row=3, col=1)

    vcols = ["#22c55e" if float(cv)>=float(ov) else "#ef4444" for cv,ov in zip(c.values, o.reindex(c.index).fillna(c).values)]
    fig.add_trace(go.Bar(x=df.index, y=v, name="Hacim", marker_color=vcols, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=v.rolling(20).mean(), name="Hacim MA20", line=dict(color="#f59e0b",width=1.2)), row=4, col=1)

    fig.update_layout(height=700, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", color="#94a3b8", size=11), legend=dict(orientation="h",y=1.02,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)), margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False, title=dict(text=f"<b>{ticker.replace('.IS','')}</b>  —  {interval.upper()}", font=dict(color="#f1f5f9",size=14),x=0.01), hovermode="x unified")
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#1e2535",showgrid=True,zeroline=False,row=i,col=1)
        fig.update_yaxes(gridcolor="#1e2535",showgrid=True,zeroline=False,row=i,col=1)
    return fig

def build_tlref_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df['TLREF'], name='Canlı TLREF', line=dict(color='#e2e8f0', width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA8'], name='SMA 8 (Kısa)', line=dict(color='#38bdf8', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA54'], name='SMA 54 (Uzun)', line=dict(color='#f59e0b', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX Şiddeti', line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D+'], name='D+ (Alıcılar)', line=dict(color='#22c55e', width=2.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D-'], name='D- (Satıcılar)', line=dict(color='#ef4444', width=2.2)), row=2, col=1)
    
    fig.update_layout(height=450, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", size=11), margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
    return fig

def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS","")
    sector = get_sector(result["ticker"])

    st.markdown(f"### {ticker_label} " f'<span class="stag">{sector}</span> ' f'`{result["score"]}/{result["max_score"]}`', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"
        col="pass" if passed else "fail"
        html=(f'<div class="mc"><div class="ml">{k}</div><div class="mv {col}">{icon} {val}</div></div>')
        (c1 if i%2==0 else c2).markdown(html, unsafe_allow_html=True)

    if result["details"]:
        st.markdown('<div class="sec">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
        dc = st.columns(3)
        for i,(k,v) in enumerate(result["details"].items()):
            dc[i%3].markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv" style="font-size:.9rem">{v}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">📈 Grafik</div>', unsafe_allow_html=True)
    fig = build_chart(result["df"], result["ticker"], strategy, interval)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

st.markdown('<div class="main-header">📊 BIST Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y %H:%M")} · {len(BIST100_TICKERS)} Hisse</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    sel_int = st.selectbox("Zaman Dilimi", ["1d","4h","1wk"], format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    st.session_state.interval = sel_int
    st.session_state.period = {"1d":"6mo","4h":"60d","1wk":"2y"}[sel_int]
    
    st.markdown("---")
    if st.button("🏭 Sektör Özeti Genel Pano", use_container_width=True): st.session_state.page = "sektor"; st.rerun()
    if st.button("📈 TLREF Canlı Faiz Analizi", use_container_width=True): st.session_state.page = "tlref_page"; st.rerun()
    if st.button("📊 Kar/Zarar Performans Paneli", use_container_width=True): st.session_state.page = "perf"; st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔍 Tekli Hisse Arama")
    search_input = st.text_input("Hisse Kodu (Örn: THYAO)", placeholder="GARAN...").upper().strip()
    if st.button("Hisse Detay Analiz Et", use_container_width=True):
        if search_input:
            st.session_state.page = "search"
            st.session_state.search_ticker = search_input
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚔️ Manuel Makro Tema Seçici")
    chosen_theme = st.selectbox("Senaryo Seç", ["—"] + list(MANUAL_MACRO_THEMES.keys()))
    if st.button("Seçili Temayı Tara", use_container_width=True):
        if chosen_theme != "—":
            st.session_state.page = "manual_macro_page"
            st.session_state.chosen_theme_name = chosen_theme
            st.rerun()

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Emre'nin Makro Stratejisi", use_container_width=True, type="primary"):
        st.session_state.update(strategy="emre", page="scanner", scan_done=False)
        st.rerun()
with col2:
    if st.button("🟣 Momentum Kırılımcısı", use_container_width=True):
        st.session_state.update(strategy="momentum", page="scanner", scan_done=False)
        st.rerun()

# ── SAYFALAR ──
if st.session_state.page == "tlref_page":
    st.markdown("## 📊 Canlı Makroekonomik Faiz Motoru (TLREF)")
    macro = get_tlref_macro_regime()
    
    mc1, mc2 = st.columns([1.5, 2.5])
    with mc1:
        st.markdown(f'<div class="mc" style="border-left: 5px solid #a78bfa; min-height:110px;"><div class="ml">Mevcut Faiz Rejimi</div><div class="mv" style="font-size:1.3rem; color:#a78bfa;">{macro["regime"]}</div><div style="font-size:0.75rem; color:#64748b; margin-top:5px;">Anlık TLREF: %{macro["current_rate"]:.2f}</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="mc" style="border-left: 5px solid #22c55e; min-height:110px;"><div class="ml">Desteklenen Sektörler</div><div class="mv" style="font-size:0.95rem; color:#22c55e;">{" · ".join(macro["sectors"])}</div><div style="font-size:0.72rem; color:#94a3b8; margin-top:3px;">{macro["description"]}</div></div>', unsafe_allow_html=True)
        
    st.plotly_chart(build_tlref_chart(macro["df"]), use_container_width=True, config={"displayModeBar":False})
    
    st.markdown("### 🏭 Rejimin Olumlu Etkilediği Sektörün Hisse Listesi")
    with st.spinner("Sektörel fiyatlar çekiliyor..."):
        stock_s = fetch_data(BIST100_YF,"5d","1d")
    
    list_rows = []
    for tkr, df in stock_s.items():
        sect = get_sector(tkr)
        if sect in macro["sectors"] and df is not None and len(df) >= 2:
            c = df['Close'].squeeze()
            chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
            list_rows.append({"Hisse": tkr.replace(".IS",""), "Sektör": sect, "Son Fiyat": f"{c.iloc[-1]:.2f} ₺", "Günlük Değişim %": chg})
            
    if list_rows:
        ldf = pd.DataFrame(list_rows).sort_values("Günlük Değişim %", ascending=False).reset_index(drop=True)
        st.dataframe(ldf.style.map(lambda x: 'color:#22c55e' if isinstance(x,float) and x>=0 else 'color:#ef4444' if isinstance(x,float) else '', subset=['Günlük Değişim %']), use_container_width=True, height=450)
    st.stop()

if st.session_state.page == "sektor":
    st.markdown("### 📈 BIST 100 Endeks Doğrulanmış Genel Performansı")
    with st.spinner("Endeks verileri okunuyor..."):
        xu100 = fetch_benchmark("6mo", "1d")
    if not xu100.empty:
        c_xu = xu100['Close'].squeeze()
        r1d = (c_xu.iloc[-1] / c_xu.iloc[-2] - 1) * 100
        r5d = (c_xu.iloc[-1] / c_xu.iloc[-6] - 1) * 100
        r30d = (c_xu.iloc[-1] / c_xu.iloc[-22] - 1) * 100
        
        xc1, xc2, xc3 = st.columns(3)
        xc1.markdown(f'<div class="mc"><div class="ml">BIST 100 Son Kapanış</div><div class="mv {"pass" if r1d>=0 else "fail"}">{r1d:+.2f}%</div></div>', unsafe_allow_html=True)
        xc2.markdown(f'<div class="mc"><div class="ml">BIST 100 Doğru 1 Haftalık (%1H)</div><div class="mv {"pass" if r5d>=0 else "fail"}">{r5d:+.2f}%</div></div>', unsafe_allow_html=True)
        xc3.markdown(f'<div class="mc"><div class="ml">BIST 100 Doğru 1 Aylık (%1A)</div><div class="mv {"pass" if r30d>=0 else "fail"}">{r30d:+.2f}%</div></div>', unsafe_allow_html=True)

    with st.spinner("Doğru sektörel matris hesaplanıyor..."):
        stock_s = fetch_data(BIST100_YF,"3mo","1d")
    
    st.markdown("### 📰 Günün Doğru Sektör Manşetleri")
    summary = build_summary(stock_s)
    CAT_STYLE = {
        "🚀 SON KAPANIŞ LİDERİ": ("#14532d","#22c55e"), "⚡ İVME KAZANAN": ("#14532d","#4ade80"),
        "📈 1 HAFTA ZİRVE": ("#14532d","#86efac"), "🏆 1 AY ZİRVE": ("#14532d","#bbf7d0"),
        "🔄 TOPARLAYAN": ("#1c3a1c","#a3e635"), "📉 SON KAPANIŞTA GERİDE": ("#4c0519","#ef4444"),
        "❄️ 1 HAFTA DİP": ("#4c0519","#f87171"), "🪨 1 AY DİP": ("#4c0519","#fca5a5"), "🐌 YAVAŞLAYAN": ("#2c1810","#fb923c"),
    }
    def cat_card(title, items, bg, border):
        rows=""
        for sect,val in items:
            vc="#22c55e" if val.startswith("+") else "#ef4444" if val.startswith("-") else "#94a3b8"
            rows+=(f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #1e2535;font-size:.83rem;"><span style="color:#e2e8f0">{sect}</span><span style="color:{vc};font-family:JetBrains Mono;font-weight:600">{val}</span></div>')
        return f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem"><div style="font-size:.7rem;font-weight:700;color:{border};text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;font-family:JetBrains Mono">{title}</div>{rows}</div>'

    rows_layout = [["🚀 SON KAPANIŞ LİDERİ","⚡ İVME KAZANAN"], ["📈 1 HAFTA ZİRVE","🏆 1 AY ZİRVE"], ["🔄 TOPARLAYAN","📉 SON KAPANIŞTA GERİDE"], ["❄️ 1 HAFTA DİP","🪨 1 AY DİP"], ["🐌 YAVAŞLAYAN",None]]
    for row_cats in rows_layout:
        cols = st.columns(2)
        for col,cat in zip(cols,row_cats):
            if cat and cat in summary: col.markdown(cat_card(cat, summary[cat], CAT_STYLE[cat][0], CAT_STYLE[cat][1]), unsafe_allow_html=True)
            
    st.markdown("---")
    bar_chart = build_sector_bar_chart(stock_s)
    if bar_chart: st.plotly_chart(bar_chart, use_container_width=True, config={"displayModeBar":False})
    st.stop()

if st.session_state.page == "manual_macro_page":
    tname = st.session_state.get("chosen_theme_name","")
    tinfo = MANUAL_MACRO_THEMES[tname]
    st.markdown(f"## {tname}")
    st.markdown(f"*{tinfo['açıklama']}*")
    
    theme_tickers = [t+".IS" for t,s in SECTOR_MAP.items() if s in tinfo["sektörler"]]
    with st.spinner("Veriler taranıyor..."):
        bm_df = fetch_benchmark("6mo","1d")
        td = fetch_data(theme_tickers,"6mo","1d")
        
    m_rows = []
    for ticker, df in td.items():
        if df is None or len(df) < 22: continue
        sc, mx, _, _ = score_emre(df, bm_df, ticker)
        c = _c(df); chg = (c.iloc[-1]/c.iloc[-2] - 1)*100
        m_rows.append({"Hisse": ticker.replace(".IS",""), "Sektör": get_sector(ticker), "Teknik Puan": f"{sc}/{mx}", "Günlük P&L": chg, "_sc": sc})
        
    if m_rows:
        mdf = pd.DataFrame(m_rows).sort_values("_sc", ascending=False).drop(columns=["_sc"]).reset_index(drop=True)
        st.dataframe(mdf, use_container_width=True)
    else: st.warning("Bu tema için uygun veri bulunamadı.")
    st.stop()

if st.session_state.page == "perf":
    st.markdown("## 📊 Geçmiş Dönem Performans & Kâr Alma Defteri")
    if st.button("▶️ Eşit Ağırlıklı Rebalans Simülasyonunu Çalıştır", type="primary", use_container_width=True):
        with st.spinner("Tarihsel döngüler hesaplanıyor..."):
            bm_df_bt = fetch_benchmark("2y","1d")
            stock_bt = fetch_data(BIST100_YF,"2y","1d")
        btr = {}
        for sk in ["emre", "momentum"]:
            pv, bn, trd, act, mnt = run_backtest(sk, stock_bt, bm_df_bt)
            btr[sk] = {"pv": pv, "bm": bn, "trades": trd, "active": act, "monthly": mnt, "stats": calc_stats(pv, bn, 100_000) if pv is not None else {}}
        st.session_state.bt_results = btr; st.session_state.bt_done = True

    if not st.session_state.bt_done: st.info("Simülasyonu başlatmak için butona tıklayın."); st.stop()
    
    bt_results = st.session_state.bt_results
    st.plotly_chart(build_perf_chart({sk: (bt_results[sk]["pv"], bt_results[sk]["bm"]) for sk in ["emre", "momentum"]}, 100_000), use_container_width=True)
    
    st.markdown("### 📅 Aylık Kâr Alma ve Ekleme Bazlı Defter (Tüm İşlemler)")
    t1, t2 = st.tabs([STRAT_LABELS["emre"], STRAT_LABELS["momentum"]])
    for tab, sk in zip([t1, t2], ["emre", "momentum"]):
        with tab:
            st.markdown("**Aylık Portföy Bileşen Değişimleri:**")
            st.dataframe(bt_results[sk]["monthly"], use_container_width=True)
            st.markdown("**Aylık Tüm İşlem Hareketleri (Alış, Satış, Ekleme, Kâr Al):**")
            trade_df_view = bt_results[sk]["trades"].copy()
            if not trade_df_view.empty:
                def color_pnl(val):
                    if isinstance(val, str) and '+' in val: return 'color: #22c55e'
                    elif isinstance(val, str) and '-' in val and val != '-': return 'color: #ef4444'
                    return ''
                st.dataframe(trade_df_view.style.map(color_pnl, subset=['Anlık Net K/Z']), use_container_width=True, height=450)
            else:
                st.info("Henüz loglanmış işlem bulunmuyor.")
    st.stop()

if st.session_state.page == "search":
    tk_q = st.session_state.get("search_ticker","").replace(".IS","")
    st.markdown(f"## 🔍 {tk_q} — Teknik & Makro Analiz Paneli")
    df_s = fetch_single(tk_q+".IS", "6mo", st.session_state.interval)
    bm_df = fetch_benchmark("6mo", st.session_state.interval)
    if df_s is not None:
        st.session_state['active_scanning_ticker'] = tk_q+".IS"
        sc, mx, crit, _ = STRATEGY_FN[st.session_state.strategy][0](df_s, bm_df)
        st.markdown(f"**Makro Puanı:** `{sc}/{mx}`")
        for k,(p,v) in crit.items(): st.markdown(f"- {'✅' if p else '❌'} **{k}**: {v}")
        
        fig = build_chart(df_s, tk_q+".IS", st.session_state.strategy, st.session_state.interval)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    else: st.error("Hisse kodu bulunamadı.")
    st.stop()

# ── TARAYICI ANA AKIŞ ─────────────────────────────────────────────────────────
if st.session_state.strategy is None: st.info("Yukarıdan bir strateji seçip taramayı başlatın."); st.stop()
strat = st.session_state.strategy
st.markdown(f"**Aktif Strateji Modülü:** `{STRAT_LABELS[strat]}` · Zaman Dilimi: `{st.session_state.interval.upper()}`")

if st.button("🔍 Tüm BIST Listesini Canlı Tara", type="primary", use_container_width=True): 
    st.session_state.update(scan_done=False, results=[])

if not st.session_state.scan_done:
    with st.spinner("Taranıyor..."):
        bm_df = fetch_benchmark(st.session_state.period, st.session_state.interval)
        stock_data = fetch_data(BIST100_YF, st.session_state.period, st.session_state.interval)
    res = []
    for tkr, df in stock_data.items():
        if df is None or len(df)<22: continue
        try:
            st.session_state['active_scanning_ticker'] = tkr
            sc, mx, crit, det = STRATEGY_FN[strat][0](df, bm_df)
            res.append({"ticker": tkr, "score": sc, "max_score": mx, "criteria": crit, "details": det, "df": df, "sector": get_sector(tkr), "in_top5": False})
        except Exception: pass
    res.sort(key=lambda x: x['score'], reverse=True)
    tc = 0; scnt = {}
    for r in res:
        s = r['sector']
        if tc < 5 and (strat != "emre" or scnt.get(s, 0) < 2): r['in_top5'] = True; scnt[s] = scnt.get(s,0)+1; tc+=1
    st.session_state.update(results=res, scan_done=True); st.rerun()

top5 = [r for r in st.session_state.results if r.get('in_top5')]
near = [r for r in st.session_state.results if not r.get('in_top5')]
l, r = st.columns([1, 2.4], gap="large")
with l:
    st.markdown("### ⭐ Top 5 Portföy")
    for r_item in top5:
        if st.button(f"⭐ {r_item['ticker'].replace('.IS','')} ({r_item['score']}/{r_item['max_score']})", key=f"t5_{r_item['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r_item["ticker"]; st.rerun()
            
    st.markdown(f'<div class="sec">⚠️ Filtreye Takılan Diğerleri</div>', unsafe_allow_html=True)
    for r_item in near[:15]:
        if st.button(f"⚫ {r_item['ticker'].replace('.IS','')} ({r_item['score']}/{r_item['max_score']})", key=f"nr_{r_item['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r_item["ticker"]; st.rerun()
with r:
    sel = st.session_state.get("selected_ticker")
    if sel:
        res_obj = next((x for x in st.session_state.results if x["ticker"]==sel), None)
        if res_obj: render_detail(res_obj, strat, st.session_state.interval)
    else: st.info("🔍 Analizi detaylı incelemek için sol listeden bir hisseye tıklayın.")
