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
from sectors import (get_sector, get_tlref_macro_regime, update_bist100_and_sectors,
                     SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
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

# ── DATA RETRIEVAL FONKSİYONLARI ──────────────────────────────────────────────
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
def fetch_benchmark(period, interval):
    df = yf.download("XU100.IS", period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

# ── İNDİKATÖRLER VE STRATEJİ MATEMATİKLERİ ───────────────────────────────────
def _c(df): return df['Close'].squeeze().dropna()
def _h(df): return df['High'].squeeze().dropna()
def _l(df): return df['Low'].squeeze().dropna()
def _v(df): return df['Volume'].squeeze().dropna()

def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi_calc(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100/(1+g/l.replace(0, np.nan))).fillna(50)

def macd_calc(s, fast=12, slow=26, sig=9):
    m = ema(s,fast)-ema(s,slow); sg = ema(m.fillna(0), sig)
    return m, sg, m-sg

def atr_calc(h, l, c, n=14): return pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(n).mean()
def vol_ratio(v, n=20):
    avg = float(v.iloc[-n-1:-1].mean())
    return float(v.iloc[-1])/avg if avg > 0 else np.nan

def rs_score(c, bm_c, days=20):
    common = c.index.intersection(bm_c.index)
    return (float(c.loc[common].iloc[-1]/c.loc[common].iloc[-days]) - 1) - (float(bm_c.loc[common].iloc[-1]/bm_c.loc[common].iloc[-days]) - 1)

def score_emre(df, bm_df, ticker):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df); bm = _c(bm_df)
        price = float(c.iloc[-1]); s20 = float(sma(c,20).iloc[-1]); s50 = float(sma(c,50).iloc[-1]); rsi_v = float(rsi_calc(c).iloc[-1]); rs = rs_score(c, bm, 20); vr = vol_ratio(v, 20)
        
        criteria = {
            "RS Pozitif (20g vs BIST)": (not np.isnan(rs) and rs > 0, f"{rs*100:.1f}%"),
            "SMA20 & SMA50 Üstünde": (price > s20 and price > s50, f"{price:.2f} > {s20:.2f}"),
            "Hacim Ortalamanın Üstünde": (not np.isnan(vr) and vr >= 0.9, f"{vr:.2f}x"),
            "RSI < 80": (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        
        sc = sum(1 for p,_ in criteria.values() if p)
        mx = 4
        
        # Makroekonomik Faiz Rejimi Filtresi
        macro = get_tlref_macro_regime()
        is_supported = get_sector(ticker) in macro["sectors"]
        criteria["Makro Rejim Uyum Desteği"] = (is_supported, "Uyumlu Sektör" if is_supported else "Uyumsuz Sektör")
        
        sc = sc + 1 if is_supported else sc
        mx += 1
        
        details = {"Fiyat": f"{price:.2f} ₺", "SMA 20": f"{s20:.2f}", "SMA 50": f"{s50:.2f}", "RSI (14)": f"{rsi_v:.1f}", "RS (20g)": f"{rs*100:.1f}%", "Hacim Oran.": f"{vr:.2f}x"}
        return sc, mx, criteria, details
    except Exception as e: return 0, 5, {"Hata": (False, str(e))}, {}

def score_momentum(df, bm_df):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df)
        price = float(c.iloc[-1]); s50 = float(sma(c,50).iloc[-1]); ml, sl, hl = macd_calc(c)
        hn = float(hl.dropna().iloc[-1]); hp = float(hl.dropna().iloc[-2]); a14 = atr_calc(h,l,c)
        atr_now = float(a14.dropna().iloc[-1]); atr_avg = float(a14.rolling(60).mean().dropna().iloc[-1]); vr = vol_ratio(v, 20)
        criteria = {
            "SMA 50 Üzerinde": (price > s50, f"{price:.2f} > {s50:.2f}"),
            "MACD Hist Pozitif & Artıyor": (hn > 0 and hn > hp, f"{hn:.4f}"),
            "ATR Genişliyor": (atr_now > atr_avg, f"{atr_now:.2f} > {atr_avg:.2f}"),
            "Hacim > 1.2x Ort.": (not np.isnan(vr) and vr > 1.2, f"{vr:.2f}x"),
        }
        details = {"Fiyat": f"{price:.2f} ₺", "SMA 50": f"{s50:.2f}", "MACD Hist": f"{hn:.4f}", "ATR (14)": f"{atr_now:.2f}", "ATR Ort": f"{atr_avg:.2f}", "Hacim Oran.": f"{vr:.2f}x"}
        return sum(1 for p,_ in criteria.values() if p), 4, criteria, details
    except Exception as e: return 0, 4, {"Hata": (False, str(e))}, {}

STRATEGY_FN = {
    "emre": (lambda df, bm: score_emre(df, bm, st.session_state.get('active_scanning_ticker', '')), "Emre'nin Makro Stratejisi"),
    "momentum": (lambda df, bm: score_momentum(df, bm), "Momentum Kırılımcısı")
}

def build_chart(df, ticker, strategy, interval):
    c = _c(df); h = _h(df); l = _l(df); v = _v(df); o = df['Open'].squeeze()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48,0.18,0.18,0.16], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="Fiyat", increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444"), row=1, col=1)
    fig.update_layout(height=600, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", color="#94a3b8"))
    return fig

# D+ ve D- DÜZELTİLDİ: Kalın ve net çizgiler
def build_tlref_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df['TLREF'], name='Canlı TLREF', line=dict(color='#e2e8f0', width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA8'], name='SMA 8', line=dict(color='#22c55e', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA54'], name='SMA 54', line=dict(color='#f59e0b', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX Trend Şiddeti', line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D+'], name='D+ (Alıcılar)', line=dict(color='#22c55e', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D-'], name='D- (Satıcılar)', line=dict(color='#ef4444', width=2)), row=2, col=1)
    
    fig.update_layout(height=380, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", size=10), margin=dict(l=10,r=10,t=10,b=10))
    return fig

def render_detail(result, strategy, interval):
    st.markdown(f"### {result['ticker'].replace('.IS','')} <span class=\"stag\">{get_sector(result['ticker'])}</span> `{result['score']}/{result['max_score']}`", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"; col="pass" if passed else "fail"
        (c1 if i%2==0 else c2).markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv {col}">{icon} {val}</div></div>', unsafe_allow_html=True)

# Session States
defaults = {"strategy":None,"selected_ticker":None,"scan_done":False,"results":[],"page":"scanner","bt_done":False,"bt_results":{},"bm_df":None}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# Sidebar Layout
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d","4h","1wk"], format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    period = {"1d":"6mo","4h":"60d","1wk":"2y"}[interval]
    st.markdown("---")
    if st.button("🏭 Sektör Özeti & TLREF", use_container_width=True): st.session_state.page = "sektor"; st.rerun()
    if st.button("📊 Performans & Backtest", use_container_width=True): st.session_state.page = "perf"; st.rerun()

# Üst Seçim Butonları (Sadece 2 Tane Kaldı)
col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Emre'nin Makro Stratejisi", use_container_width=True, type="primary"): st.session_state.update(strategy="emre", page="scanner", scan_done=False); st.rerun()
with col2:
    if st.button("🟣 Momentum Kırılımcısı", use_container_width=True): st.session_state.update(strategy="momentum", page="scanner", scan_done=False); st.rerun()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: SEKTÖR ÖZETİ & TLREF PANOSU
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "sektor":
    st.markdown("### 📈 BIST 100 Endeks Genel Performansı")
    with st.spinner("Endeks verileri okunuyor..."):
        xu100 = fetch_benchmark("6mo", "1d")
    if not xu100.empty:
        c_xu = xu100['Close'].squeeze()
        r1d = (c_xu.iloc[-1] / c_xu.iloc[-2] - 1) * 100
        r5d = (c_xu.iloc[-1] / c_xu.iloc[-6] - 1) * 100
        r30d = (c_xu.iloc[-1] / c_xu.iloc[-22] - 1) * 100
        
        xc1, xc2, xc3 = st.columns(3)
        xc1.markdown(f'<div class="mc"><div class="ml">BIST 100 Son Kapanış</div><div class="mv {"pass" if r1d>=0 else "fail"}">{r1d:+.2f}%</div></div>', unsafe_allow_html=True)
        xc2.markdown(f'<div class="mc"><div class="ml">BIST 100 1 Haftalık</div><div class="mv {"pass" if r5d>=0 else "fail"}">{r5d:+.2f}%</div></div>', unsafe_allow_html=True)
        xc3.markdown(f'<div class="mc"><div class="ml">BIST 100 1 Aylık</div><div class="mv {"pass" if r30d>=0 else "fail"}">{r30d:+.2f}%</div></div>', unsafe_allow_html=True)

    macro = get_tlref_macro_regime()
    with st.spinner("Sektör canlı momentumları hesaplanıyor..."):
        stock_s = fetch_data(BIST100_YF,"3mo","1d")
    full_df = _sector_returns(stock_s)
    
    st.markdown("---")
    st.markdown("## 📊 Canlı Makroekonomik Faiz Motoru (TLREF)")
    mc1, mc2 = st.columns([1.5, 2.5])
    with mc1:
        st.markdown(f'<div class="mc" style="border-left: 5px solid #a78bfa; min-height:120px;"><div class="ml">Mevcut Faiz Rejimi</div><div class="mv" style="font-size:1.3rem; color:#a78bfa;">{macro["regime"]}</div><div style="font-size:0.75rem; color:#64748b; margin-top:5px;">Canlı Faiz Oranı: %{macro["current_rate"]:.2f}</div></div>', unsafe_allow_html=True)
    with mc2:
        sect_perf_str = []
        for s in macro["sectors"]:
            if not full_df.empty and s in full_df['sector'].values:
                row = full_df[full_df['sector'] == s].iloc[0]
                sect_perf_str.append(f"<b>{s}</b> (<span class='pass'>{row['ret_5d']:+.1f}% 1H</span> / <span class='pass'>{row['ret_21d']:+.1f}% 1A</span>)")
            else:
                sect_perf_str.append(f"<b>{s}</b>")
        st.markdown(f'<div class="mc" style="border-left: 5px solid #22c55e; min-height:120px;"><div class="ml">Desteklenen Sektör Performansları</div><div class="mv" style="font-size:0.85rem; font-weight:normal; line-height:1.4rem;">{" · ".join(sect_perf_str)}</div><div style="font-size:0.72rem; color:#94a3b8; margin-top:3px;">{macro["description"]}</div></div>', unsafe_allow_html=True)
        
    st.plotly_chart(build_tlref_chart(macro["df"]), use_container_width=True, config={"displayModeBar":False})
    
    st.markdown("### 📰 Günün Sektör Manşetleri")
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
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: PERFORMANCE / BACKTEST PANELİ
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "perf":
    st.markdown("## 📊 Strateji Performans ve Karşılaştırmalı Backtest Modülü")
    st.markdown("Haziran 2024'ten günümüze portföy dağılımları · **Her ay başı Kümülatif Eşit Ağırlık Rebalansı** · 100.000 ₺ başlangıç sermayesi")
    st.markdown("---")

    if st.button("▶️ Tüm Karşılaştırmalı Backtestleri Çalıştır", type="primary", use_container_width=True):
        with st.spinner("2 yıllık BIST geçmiş verileri analiz ediliyor (1 dakika sürebilir)..."):
            bm_df_bt = fetch_benchmark("2y","1d")
            stock_bt = fetch_data(BIST100_YF,"2y","1d")
            
        bt_results = {}
        prog = st.progress(0)
        strats_to_run = ["emre", "momentum"]
        for i, sk in enumerate(strats_to_run):
            prog.progress((i+1)/2, text=f"{STRAT_LABELS[sk]} simüle ediliyor...")
            pv, bm_n, trades, active, monthly = run_backtest(sk, stock_bt, bm_df_bt)
            bt_results[sk] = {"pv": pv, "bm": bm_n, "trades": trades, "active": active, "monthly": monthly, "stats": calc_stats(pv, bm_n, 100_000) if pv is not None else {}}
        prog.empty()
        st.session_state.bt_results = bt_results
        st.session_state.bt_done = True

    if not st.session_state.bt_done:
        st.info("📊 Geçmişten günümüze performans çizelgelerini listelemek için lütfen yukarıdaki butona tıklayarak simülasyonu başlatın.")
        st.stop()

    bt_results = st.session_state.bt_results
    st.markdown("### 📈 Strateji Performans Karşılaştırma Grafiği")
    perf_map = {sk: (bt_results[sk]["pv"], bt_results[sk]["bm"]) for sk in ["emre", "momentum"] if bt_results[sk].get("pv") is not None}
    if perf_map:
        st.plotly_chart(build_perf_chart(perf_map, 100_000), use_container_width=True, config={"displayModeBar":False})

    st.markdown("### 📐 Temel Strateji İstatistikleri")
    s1, s2 = st.columns(2)
    for col, sk in zip([s1, s2], ["emre", "momentum"]):
        with col:
            st.markdown(f"**{STRAT_LABELS[sk]}**")
            for k, v in bt_results[sk].get("stats",{}).items():
                st.markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv">{v}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📅 Aylık Portföy Değişim Detayları")
    t1, t2 = st.tabs([STRAT_LABELS["emre"], STRAT_LABELS["momentum"]])
    for tab, sk in zip([t1, t2], ["emre", "momentum"]):
        with tab:
            mdf = bt_results[sk].get("monthly")
            if mdf is not None and len(mdf) > 0: st.dataframe(mdf, use_container_width=True, height=400)

    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: SCANNER / TARAYICI ANA MODÜL
# ═══════════════════════════════════════════════════════════════════
if st.session_state.strategy is None:
    st.info("👆 Lütfen üst paneldeki butonları kullanarak çalıştırılacak strateji mimarisini seçin.")
    st.stop()

strat = st.session_state.strategy
st.markdown(f"**Aktif Strateji Modülü:** `{STRAT_LABELS.get(strat, strat)}` · Zaman Dilimi: `{interval.upper()}`")
if st.button("🔍 Tüm BIST Listesini Canlı Tara", type="primary", use_container_width=True):
    st.session_state.update(scan_done=False, results=[], selected_ticker=None)

if not st.session_state.scan_done:
    with st.spinner("Piyasa verileri indiriliyor ve taranıyor..."):
        bm_df = fetch_benchmark(period, interval)
        stock_data = fetch_data(BIST100_YF, period, interval)
    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df)<22: continue
        try:
            st.session_state['active_scanning_ticker'] = ticker
            sc, mx, crit, det = STRATEGY_FN[strat][0](df, bm_df)
            c = _c(df); bm = _c(bm_df); rs = rs_score(c, bm, 20)
            results.append({"ticker": ticker, "score": sc, "max_score": mx, "criteria": crit, "details": det, "df": df, "rs": rs if not np.isnan(rs) else -999, "sector": get_sector(ticker), "in_top5": False})
        except Exception: pass
    results.sort(key=lambda x: (x['score'], x['rs']), reverse=True)
    
    t_count = 0; s_counts = {}
    for r in results:
        sect = r['sector']
        if t_count < 5 and (strat != "emre" or s_counts.get(sect, 0) < 2):
            r['in_top5'] = True; s_counts[sect] = s_counts.get(sect, 0) + 1; t_count += 1
    st.session_state.update(results=results, scan_done=True); st.rerun()

top5_r = [r for r in st.session_state.results if r.get('in_top5')]
near_r = [r for r in st.session_state.results if not r.get('in_top5')]
left, right = st.columns([1, 2.4], gap="large")

with left:
    st.markdown(f'<div class="sec">⭐ Seçilen Top 5 Portföy</div>', unsafe_allow_html=True)
    for r in top5_r:
        if st.button(f"⭐ {r['ticker'].replace('.IS','')} ({r['score']}/{r['max_score']}) [{r['sector']}]", key=f"t5_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]; st.rerun()
    st.markdown(f'<div class="sec">⚠️ Filtreye Takılan Diğerleri</div>', unsafe_allow_html=True)
    for r in near_r[:15]:
        if st.button(f"⚫ {r['ticker'].replace('.IS','')} ({r['score']}/{r['max_score']})", key=f"nr_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]; st.rerun()

with right:
    sel = st.session_state.get("selected_ticker")
    if sel:
        res = next((r for r in st.session_state.results if r["ticker"]==sel), None)
        if res: render_detail(res, strat, interval)
    else: st.info("🔍 Grafik ve indikatör analiz kırılımlarını detaylı incelemek için sol listeden bir hisseye tıklayın.")
