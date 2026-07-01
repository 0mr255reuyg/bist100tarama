import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import google.generativeai as genai

warnings.filterwarnings('ignore')

from backtest_engine import run_backtest, calc_stats, build_perf_chart, STRAT_LABELS, STRAT_COLORS
from sectors import (get_sector, get_theme_sectors, get_theme_info, update_bist100_and_sectors,
                     MACRO_THEMES, MACRO_THEMES_PRIMARY, MACRO_THEMES_SECONDARY,
                     SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
from sector_summary import build_summary, build_sector_bar_chart, _sector_returns

# Başlangıçta listeyi otomatik güncelle/canlı çek
SECTOR_MAP, BIST100_TICKERS = update_bist100_and_sectors()
BIST100_YF = [t+".IS" for t in BIST100_TICKERS]

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

# ── VERI ÇEKME FONKSİYONLARI (Aynı Kaldı) ──────────────────────────────────────
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

# ── İNDİKATÖRLER & STRATEJİ SKORLAMALARI (Aynı Kaldı) ───────────────────────────
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
        criteria = {
            "RS Pozitif (20g vs BIST)":    (not np.isnan(rs) and rs > 0, f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A"),
            "SMA20 & SMA50 Üstünde":       (price > s20 and (np.isnan(s50) or price > s50), f"{price:.2f} > {s20:.2f}" + (f" / {s50:.2f}" if not np.isnan(s50) else "")),
            "Hacim Ortalamanın Üstünde":   (not np.isnan(vr) and vr >= 0.9, f"{vr:.2f}x" if not np.isnan(vr) else "N/A"),
            "RSI < 80":                    (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        details = {
            "Fiyat": f"{price:.2f} ₺", "SMA 20": f"{s20:.2f}", "SMA 50": f"{s50:.2f}" if not np.isnan(s50) else "N/A",
            "RSI (14)": f"{rsi_v:.1f}", "RS (20g)": f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A", "Hacim Oran.": f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
        }
        sc = sum(1 for p,_ in criteria.values() if p)
        return sc, 4, criteria, details
    except Exception as e: return 0, 4, {"Hata": (False, str(e))}, {}

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
    except Exception as e: return 0, 4, {"Hata": (False, str(e))}, {}

STRATEGY_FN = {"emre": (score_emre, "Emre'nin Stratejisi"), "momentum": (score_momentum, "Momentum Kırılımcısı")}

# ── GRAFİK & DETAY PANELİ FONKSİYONLARI (Aynı Kaldı) ───────────────────────────
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

def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS","")
    sector = get_sector(result["ticker"])
    st.markdown(f"### {ticker_label} " f'<span class="stag">{sector}</span> ' f'`{result["score"]}/{result["max_score"]}`', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"
        col="pass" if passed else "fail"
        html=(f'<div class="mc"><div class="ml">{k}</div>' f'<div class="mv {col}">{icon} {val}</div></div>')
        (c1 if i%2==0 else c2).markdown(html, unsafe_allow_html=True)
    if result["details"]:
        st.markdown('<div class="sec">📐 İndikatör Değerleri</div>', unsafe_allow_html=True)
        dc = st.columns(3)
        for i,(k,v) in enumerate(result["details"].items()):
            dc[i%3].markdown(f'<div class="mc"><div class="ml">{k}</div>' f'<div class="mv" style="font-size:.9rem">{v}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">📈 Grafik</div>', unsafe_allow_html=True)
    fig = build_chart(result["df"], result["ticker"], strategy, interval)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 BIST Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y %H:%M")} · {len(BIST100_TICKERS)} Hisse (Canlı Liste) · yfinance</div>', unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {"strategy":None,"selected_ticker":None,"scan_done":False,"results":[],"page":"scanner",
            "bt_done":False,"bt_results":{},"bm_df":None,"search_result":None, "ai_messages":[]}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d","4h","1wk"], format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    period_map = {"1d":"6mo","4h":"60d","1wk":"2y"}
    period = period_map[interval]

    st.markdown("---")
    st.markdown("### 🤖 Sekreter API Anahtarı")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AI Sekreter için anahtar girin...")
    if gemini_key:
        genai.configure(api_key=gemini_key)

    st.markdown("---")
    st.markdown("### 🔍 Hisse Ara")
    search_input = st.text_input("Ticker", placeholder="GARAN...").upper().strip()
    if st.button("Ara", use_container_width=True, key="search_btn"):
        if search_input:
            st.session_state.page = "search"
            st.session_state.search_ticker = search_input
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 İşlemler Panel")
    if st.button("🤖 Finansal Sekreter Chat", use_container_width=True, type="primary"):
        st.session_state.page = "ai_secretary"
        st.rerun()

    st.markdown("---")
    st.markdown("### 🌍 Makro Tema")
    all_themes = ["—"] + list(MACRO_THEMES_PRIMARY.keys()) + ["──────"] + list(MACRO_THEMES_SECONDARY.keys())
    macro_theme = st.selectbox("Tema", all_themes)
    if st.button("Tema'ya Göre Tara", use_container_width=True, key="macro_btn"):
        if macro_theme not in ["—","──────"]:
            st.session_state.page = "macro"
            st.session_state.macro_theme = macro_theme
            st.rerun()

    st.markdown("---")
    if st.button("📊 Performans", use_container_width=True, key="perf_btn"):
        st.session_state.page = "perf"
        st.rerun()
    if st.button("🏭 Sektör Özeti", use_container_width=True, key="sektor_btn"):
        st.session_state.page = "sektor"
        st.rerun()

# ── STRATEJI BUTONLARI ────────────────────────────────────────────────────────
col1,col2,_ = st.columns([2,2,4])
with col1:
    if st.button("🟡 Emre'nin Stratejisi", use_container_width=True):
        st.session_state.update(strategy="emre", selected_ticker=None, scan_done=False, page="scanner", results=[])
        st.rerun()
with col2:
    if st.button("🟣 Momentum Kırılımcısı", use_container_width=True):
        st.session_state.update(strategy="momentum", selected_ticker=None, scan_done=False, page="scanner", results=[])
        st.rerun()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: 🤖 FINANSAL SEKRETER (YENİ EKLENDİ)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "ai_secretary":
    st.markdown("## 🤖 Finansal Sekreter")
    st.markdown("Stratejileriniz, tarama sonuçlarınız veya genel piyasa koşulları hakkında soru sorun.")
    
    if not gemini_key:
        st.warning("⚠️ Lütfen sol menüden geçerli bir Gemini API Key giriniz. (Ücretsiz alabilirsiniz)")
    else:
        # Sohbet geçmişini listele
        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_query = st.chat_input("Sekretere bir soru sorun...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.ai_messages.append({"role": "user", "content": user_query})
            
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                system_prompt = "Sen gelişmiş bir algo-trading ve finans asistanısın. Kullanıcıya teknik indikatörler, BIST stratejileri ve portföy optimizasyonu konularında net cevaplar ver."
                full_prompt = f"{system_prompt}\n\nKullanıcı Sorusu: {user_query}"
                
                with st.spinner("Düşünüyor..."):
                    response = model.generate_content(full_prompt)
                    
                with st.chat_message("assistant"):
                    st.write(response.text)
                st.session_state.ai_messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"AI Yanıt Hatası: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: ARAMA (Aynı Kaldı)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "search":
    ticker_q = st.session_state.get("search_ticker","")
    ticker_yf = ticker_q+".IS"
    st.markdown(f"## 🔍 {ticker_q} — Analiz")
    sector = get_sector(ticker_yf)
    st.markdown(f"**Sektör:** `{sector}`")
    with st.spinner("Veri çekiliyor..."):
        df_s  = fetch_single(ticker_yf, period, interval)
        bm_df = fetch_benchmark(period, interval)
    if df_s is None or len(df_s) < 22:
        st.error(f"❌ {ticker_q} için yeterli veri çekilemedi. Ticker doğru mu?")
        st.stop()
    col1, col2 = st.columns(2)
    for col, sk in [(col1,"emre"),(col2,"momentum")]:
        with col:
            fn = STRATEGY_FN[sk][0]; label = STRATEGY_FN[sk][1]
            st.markdown(f"### {label}")
            sc, mx, crit, det = fn(df_s, bm_df)
            color = "#22c55e" if sc==mx else ("#f59e0b" if sc>=mx-1 else "#ef4444")
            st.markdown(f"**Skor:** <span style='color:{color};font-size:1.3rem;font-weight:700'>{sc}/{mx}</span>", unsafe_allow_html=True)
            for k,(passed,val) in crit.items():
                icon="✅" if passed else "❌"; cc="pass" if passed else "fail"
                st.markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv {cc}">{icon} {val}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    fig = build_chart(df_s, ticker_yf, "emre", interval)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: SEKTÖR ÖZETİ (Aynı Kaldı)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "sektor":
    st.markdown(f"## 🏭 BİST Sektörel Özet · {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    st.markdown("---")
    with st.spinner("Sektör verileri hesaplanıyor..."):
        bm_df_s = fetch_benchmark("3mo","1d")
        stock_s = fetch_data(BIST100_YF,"3mo","1d")
    summary = build_summary(stock_s)
    CAT_STYLE = {
        "🚀 SON KAPANIŞ LİDERİ":  ("#14532d","#22c55e"), "⚡ İVME KAZANAN":        ("#14532d","#4ade80"),
        "📈 1 HAFTA ZİRVE":        ("#14532d","#86efac"), "🏆 1 AY ZİRVE":          ("#14532d","#bbf7d0"),
        "🔄 TOPARLAYAN":          ("#1c3a1c","#a3e635"), "📉 SON KAPANIŞTA GERİDE":("#4c0519","#ef4444"),
        "❄️ 1 HAFTA DİP":        ("#4c0519","#f87171"), "🪨 1 AY DİP":            ("#4c0519","#fca5a5"), "🐌 YAVAŞLAYAN":          ("#2c1810","#fb923c"),
    }
    def cat_card(title, items, bg, border):
        rows=""
        for sect,val in items:
            vc="#22c55e" if val.startswith("+") else "#ef4444" if val.startswith("-") else "#94a3b8"
            rows+=(f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #1e2535;font-size:.83rem;"><span style="color:#e2e8f0">{sect}</span><span style="color:{vc};font-family:JetBrains Mono;font-weight:600">{val}</span></div>')
        return (f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem"><div style="font-size:.7rem;font-weight:700;color:{border};text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;font-family:JetBrains Mono">{title}</div>{rows}</div>')

    st.markdown("### 📰 Günün Manşetleri")
    rows_layout = [["🚀 SON KAPANIŞ LİDERİ","⚡ İVME KAZANAN"], ["📈 1 HAFTA ZİRVE","🏆 1 AY ZİRVE"], ["🔄 TOPARLAYAN","📉 SON KAPANIŞTA GERİDE"], ["❄️ 1 HAFTA DİP","🪨 1 AY DİP"], ["🐌 YAVAŞLAYAN",None]]
    for row_cats in rows_layout:
        cols = st.columns(2)
        for col,cat in zip(cols,row_cats):
            if cat and cat in summary:
                bg,border = CAT_STYLE.get(cat,("#161b27","#475569"))
                col.markdown(cat_card(cat, summary[cat], bg, border), unsafe_allow_html=True)
    st.markdown("---")
    bar_chart = build_sector_bar_chart(stock_s)
    if bar_chart: st.plotly_chart(bar_chart, use_container_width=True, config={"displayModeBar":False})
    st.markdown("---")
    st.markdown("### 📋 Sektör Detay Tablosu")
    full_df = _sector_returns(stock_s)
    if not full_df.empty:
        full_df = full_df.sort_values('ret_1d',ascending=False).reset_index(drop=True)
        full_df.columns=['Sektör','Son Kapanış %','1 Hafta %','1 Ay %','İvme (5g)','İvme (21g)','Hisse Sayısı']
        for col_name in ['Son Kapanış %','1 Hafta %','1 Ay %']:
            full_df[col_name] = full_df[col_name].apply(lambda x: f"{x:+.2f}%" if not np.isnan(x) else "N/A")
        def color_ret(val):
            if isinstance(val,str) and val.startswith('+'): return 'color:#22c55e'
            if isinstance(val,str) and val.startswith('-'): return 'color:#ef4444'
            return ''
        st.dataframe(full_df.style.map(color_ret,subset=['Son Kapanış %','1 Hafta %','1 Ay %']), use_container_width=True, height=500)
    st.stop()

# ── SAYFA: MAKRO TEMA (Aynı Kaldı) ───────────────────────────────────────────
if st.session_state.page == "macro":
    macro_theme = st.session_state.get("macro_theme","")
    if not macro_theme or macro_theme in ["—","──────"]:
        st.session_state.page = "scanner"; st.rerun()
    info = get_theme_info(macro_theme); theme_sectors = info.get("sektörler",[]); aciklama = info.get("açıklama",""); one_cikanlar = info.get("öne_çıkan",[])
    st.markdown(f"## 🌍 {macro_theme}\n*{aciklama}*")
    if one_cikanlar:
        st.markdown("**⭐ Öne Çıkan Hisseler:**")
        oc_cols = st.columns(min(len(one_cikanlar),5))
        for col,tkr in zip(oc_cols,one_cikanlar):
            col.markdown(f'<div class="mc" style="text-align:center"><div style="font-weight:700;color:#f1f5f9">{tkr}</div><div style="font-size:.65rem;color:#64748b">{get_sector(tkr)}</div></div>', unsafe_allow_html=True)
    st.markdown(f"**Sektörler:** {' · '.join(theme_sectors)}")
    st.markdown("---")
    theme_tickers=[t+".IS" for t,s in SECTOR_MAP.items() if s in theme_sectors]
    with st.spinner("Veri çekiliyor..."):
        bm_df = fetch_benchmark(period,interval); td = fetch_data(theme_tickers,period,interval)
    rows=[]
    for ticker,df in td.items():
        if df is None or len(df)<22: continue
        try:
            sc_e,mx_e,_,_ = score_emre(df,bm_df); sc_m,mx_m,_,_ = score_momentum(df,bm_df); c=_c(df); bm=_c(bm_df); rs=rs_score(c,bm,20)
            rows.append({'Hisse':ticker.replace('.IS',''),'Sektör':get_sector(ticker),'Emre':f"{sc_e}/{mx_e}",'Momentum':f"{sc_m}/{mx_m}",'RS (20g)':f"{rs*100:.1f}%" if not np.isnan(rs) else "N/A",'Fiyat':f"₺{float(c.iloc[-1]):.2f}",'_es':sc_e,'_rs':rs if not np.isnan(rs) else -999})
        except Exception: pass
    if rows:
        df_rows=pd.DataFrame(rows).sort_values(['_es','_rs'],ascending=False)
        st.dataframe(df_rows.drop(columns=['_es','_rs']),use_container_width=True,height=350)
        sel_t=st.selectbox("Hisse seç",df_rows['Hisse'].tolist())
        if sel_t and sel_t+".IS" in td:
            tkr_yf=sel_t+".IS"; sc,mx,crit,det=score_emre(td[tkr_yf],bm_df)
            render_detail({"ticker":tkr_yf,"score":sc,"max_score":mx,"criteria":crit,"details":det,"df":td[tkr_yf]},"emre",interval)
    else: st.warning("Bu sektörler için veri alınamadı.")
    st.stop()

# ── SAYFA: PERFORMANS / BACKTEST (Aynı Kaldı) ─────────────────────────────────
if st.session_state.page == "perf":
    st.markdown("## 📊 Strateji Performansı — Haziran 2024'ten İtibaren\nHer ayın ilk borsa günü top 5 hisse · 100.000 ₺ başlangıç")
    st.markdown("---")
    if st.button("▶️ Backtest Çalıştır (Haz 2024 →)", type="primary"):
        with st.spinner("Veriler çekiliyor ve backtest hesaplanıyor..."):
            bm_df_bt = fetch_benchmark("2y","1d"); stock_bt = fetch_data(BIST100_YF,"2y","1d")
        bt_results={}
        prog=st.progress(0)
        for i,sk in enumerate(["emre","momentum"]):
            prog.progress((i+1)/2, text=f"{STRATEGY_FN[sk][1]} hesaplanıyor...")
            pv,bm_n,trades,active,monthly=run_backtest(sk,stock_bt,bm_df_bt)
            bt_results[sk]={"pv":pv,"bm":bm_n,"trades":trades,"active":active,"monthly":monthly,"stats":calc_stats(pv,bm_n,100_000) if pv is not None else {}}
        prog.empty(); st.session_state.bt_results=bt_results; st.session_state.bt_done=True
    bt_results=st.session_state.bt_results
    if not bt_results: st.info("▶️ Backtest çalıştır butonuna bas."); st.stop()
    st.markdown("### 📌 Bu Ay Aktif Portföyler")
    c1,c2=st.columns(2)
    for col,sk in zip([c1,c2],["emre","momentum"]):
        with col:
            st.markdown(f"**{STRATEGY_FN[sk][1]}**")
            for pos in sorted(bt_results[sk].get("active",[]),key=lambda x:x['pnl_pct'],reverse=True):
                pnl=pos['pnl_pct']; pc="#22c55e" if pnl>=0 else "#ef4444"; sect=get_sector(pos['ticker'])
                st.markdown(f'<div class="mc"><div style="display:flex;justify-content:space-between"><span style="font-weight:700;color:#f1f5f9">{pos["ticker"]}</span><span class="stag">{sect}</span><span style="color:{pc};font-family:JetBrains Mono;font-weight:600">{pnl:+.1f}%</span></div><div style="font-size:.7rem;color:#64748b;margin-top:.2rem">Alış: {pos["buy_date"]} · ₺{pos["buy_price"]:.2f} → ₺{pos["current_price"]:.2f}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📈 Portföy Performansı vs BIST 100")
    perf_map={sk:(bt_results[sk]["pv"],bt_results[sk]["bm"]) for sk in ["emre","momentum"] if bt_results[sk].get("pv") is not None}
    if perf_map: st.plotly_chart(build_perf_chart(perf_map,100_000), use_container_width=True, config={"displayModeBar":False})
    st.markdown("### 📐 Özet İstatistikler")
    s1,s2=st.columns(2)
    for col,sk in zip([s1,s2],["emre","momentum"]):
        with col:
            st.markdown(f"**{STRATEGY_FN[sk][1]}**")
            for k,v in bt_results[sk].get("stats",{}).items():
                is_pos="+" in str(v) and "Drawdown" not in k; is_neg=str(v).startswith("-") or "Drawdown" in k; vc="#22c55e" if is_pos else "#ef4444" if is_neg else "#f1f5f9"
                st.markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv" style="color:{vc}">{v}</div></div>', unsafe_allow_html=True)
    st.stop()

# ── SAYFA: TARAMA (Aynı Kaldı) ────────────────────────────────────────────────
if st.session_state.strategy is None: st.info("👆 Yukarıdan bir strateji seç."); st.stop()
strategy = st.session_state.strategy; strategy_label = STRATEGY_FN[strategy][1]
st.markdown(f"**Aktif:** `{strategy_label}` · `{interval.upper()}` · `{len(BIST100_TICKERS)} hisse`")
st.markdown("---")
if st.button("🔍 Tara — Tüm BIST", type="primary"):
    st.session_state.update(scan_done=False, results=[], selected_ticker=None)

if not st.session_state.scan_done:
    with st.spinner("Benchmark çekiliyor..."): bm_df = fetch_benchmark(period, interval); st.session_state.bm_df = bm_df
    with st.spinner(f"{len(BIST100_TICKERS)} hisse taranıyor..."): stock_data = fetch_data(BIST100_YF, period, interval)
    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df)<22: continue
        try:
            sc, mx, crit, det = score_emre(df, bm_df) if strategy == "emre" else score_momentum(df, bm_df)
            c = _c(df); bm = _c(bm_df); rs = rs_score(c, bm, 20); rs_val = rs if not np.isnan(rs) else -999
            results.append({"ticker": ticker, "score": sc, "max_score": mx, "criteria": crit, "details": det, "df": df, "rs": rs_val, "sector": get_sector(ticker), "in_top5": False})
        except Exception: pass
    results.sort(key=lambda x: (x['score'], x['rs']) if strategy=="emre" else x['score'], reverse=True)
    top5_count = 0; sector_counts = {}
    for r in results:
        sect = r['sector']
        if top5_count < 5 and (strategy != "emre" or sector_counts.get(sect, 0) < 2):
            r['in_top5'] = True; sector_counts[sect] = sector_counts.get(sect, 0) + 1; top5_count += 1
    st.session_state.update(results=results, scan_done=True); st.rerun()

results = st.session_state.results
if not results: st.warning("Veri yok — Tara butonuna bas."); st.stop()
top5_r = [r for r in results if r.get('in_top5')]; near_r = [r for r in results if not r.get('in_top5')]
left_col, right_col = st.columns([1, 2.4], gap="large")
with left_col:
    st.markdown(f'<div class="sec">⭐ Top 5 — Bu Ay ({len(top5_r)})</div>', unsafe_allow_html=True)
    for r in top5_r:
        lbl = r["ticker"].replace(".IS",""); sect = r.get("sector",""); rs_s = f" RS:{r['rs'] * 100:.1f}%" if r.get('rs') and r['rs'] != -999 else ""
        if st.button(f"⭐ {lbl} {r['score']}/{r['max_score']}{rs_s} [{sect}]", key=f"t5_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]; st.rerun()
    st.markdown(f'<div class="sec">⚠️ Diğer Hisseler ({len(near_r)})</div>', unsafe_allow_html=True)
    for r in near_r[:20]: # Ekranı tıkabasa doldurmamak adına ilk 20 diğer hisse
        lbl = r["ticker"].replace(".IS",""); sect = r.get("sector",""); miss = r["max_score"]-r["score"]; emoji = "🟡" if miss==1 else "🟠" if miss==2 else "🔴"
        if st.button(f"{emoji} {lbl} {r['score']}/{r['max_score']} [{sect}]", key=f"nr_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]; st.rerun()
with right_col:
    sel = st.session_state.get("selected_ticker")
    if not sel:
        if top5_r:
            st.markdown(f"**Bu ay {strategy_label} top 5:**")
            for r in top5_r: st.markdown(f"⭐ **{r['ticker'].replace('.IS','')}** `{r['score']}/{r['max_score']}` · {r.get('sector','')}")
    else:
        sel_result = next((r for r in results if r["ticker"]==sel), None)
        if sel_result: render_detail(sel_result, strategy, interval)
