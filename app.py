import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# NameError risklerini sıfırlayan global state tanımlama blokları
if 'interval' not in st.session_state: st.session_state.interval = "1d"
if 'period' not in st.session_state: st.session_state.period = "6mo"
if 'page' not in st.session_state: st.session_state.page = "scanner"

from backtest_engine import run_backtest, calc_stats, build_perf_chart, STRAT_LABELS, STRAT_COLORS
from sectors import (get_sector, get_tlref_macro_regime, update_bist100_and_sectors,
                     MANUAL_MACRO_THEMES, SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
from sector_summary import build_summary, build_sector_bar_chart, _sector_returns

SECTOR_MAP, BIST100_TICKERS = update_bist100_and_sectors()
BIST100_YF = [t+".IS" for t in BIST100_TICKERS]

st.set_page_config(page_title="BIST Strateji Tarayıcı", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# CSS Yapılandırma Bloğu (Aynı şekilde korundu)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700&family=JetBrains+Mono:wght=400;500&display=swap');
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
def fetch_data(tickers, _period, _interval):
    data = {}
    progress = st.progress(0, "Veri çekiliyor...")
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=_period, interval=_interval, auto_adjust=True, progress=False, timeout=10)
            if df is not None and len(df) >= 5:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df.dropna(subset=['Close'])
        except Exception: pass
        if i % 15 == 0: progress.progress(min((i+1)/len(tickers), 1.0), f"{i+1}/{len(tickers)} hisse")
    progress.empty()
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single(ticker, _period, _interval):
    try:
        df = yf.download(ticker, period=_period, interval=_interval, auto_adjust=True, progress=False, timeout=10)
        if df is not None and len(df) > 2:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df.dropna(subset=['Close'])
    except Exception: pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark(_period, _interval):
    df = yf.download("XU100.IS", period=_period, interval=_interval, auto_adjust=True, progress=False, timeout=10)
    if df is not None and len(df) > 0:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=['Close'])
    return pd.DataFrame()

# İndikatör Kırılımları
def _c(df): return df['Close'].squeeze().dropna()
def _v(df): return df['Volume'].squeeze().dropna()
def sma(s, n): return s.rolling(n).mean()
def rsi_calc(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100/(1+g/l.replace(0, np.nan))).fillna(50)
def rs_score(c, bm_c, days=20):
    common = c.index.intersection(bm_c.index)
    if len(common) < days: return 0.0
    return (float(c.loc[common].iloc[-1]/c.loc[common].iloc[-days]) - 1) - (float(bm_c.loc[common].iloc[-1]/bm_c.loc[common].iloc[-days]) - 1)

def score_emre(df, bm_df, ticker):
    try:
        c = _c(df); bm = _c(bm_df); price = float(c.iloc[-1]); s20 = float(sma(c,20).iloc[-1]); s50 = float(sma(c,50).iloc[-1]); rsi_v = float(rsi_calc(c).iloc[-1]); rs = rs_score(c, bm, 20); vr = float(_v(df).iloc[-1]) / max(float(_v(df).iloc[-21:-1].mean()), 1.0)
        criteria = {
            "RS Pozitif (20g vs BIST)": (not np.isnan(rs) and rs > 0, f"{rs*100:.1f}%"),
            "SMA20 & SMA50 Üstünde": (price > s20 and price > s50, f"{price:.2f} > {s20:.2f}"),
            "Hacim Ortalamanın Üstünde": (not np.isnan(vr) and vr >= 0.9, f"{vr:.2f}x"),
            "RSI < 80": (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        sc = sum(1 for p,_ in criteria.values() if p); mx = 4
        macro = get_tlref_macro_regime()
        is_supported = get_sector(ticker) in macro["sectors"]
        criteria["Makro Rejim Uyum Desteği"] = (is_supported, "Uyumlu Sektör" if is_supported else "Uyumsuz Sektör")
        if is_supported: sc += 1
        mx += 1
        return sc, mx, criteria, {"Fiyat": f"{price:.2f} ₺", "SMA 20": f"{s20:.2f}", "SMA 50": f"{s50:.2f}"}
    except Exception: return 0, 5, {}, {}

STRATEGY_FN = {
    "emre": (lambda df, bm: score_emre(df, bm, st.session_state.get('active_scanning_ticker', '')), "Emre'nin Makro Stratejisi"),
    "momentum": (lambda df, bm: (1, 4, {"Momentum Kriteri": (True, "Aktif")}, {})), "Momentum Kırılımcısı"
}

def build_chart(df, ticker, strategy, interval):
    fig = make_subplots(rows=1, cols=1); fig.add_trace(go.Scatter(x=df.index, y=df['Close'].squeeze(), name="Kapanış"))
    fig.update_layout(height=400, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14")
    return fig

def build_tlref_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df['TLREF'], name='Canlı TLREF', line=dict(color='#e2e8f0', width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA8'], name='SMA 8', line=dict(color='#38bdf8', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA54'], name='SMA 54', line=dict(color='#f59e0b', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX Şiddet', line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D+'], name='D+ (Alıcı İvme)', line=dict(color='#22c55e', width=2.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D-'], name='D- (Satıcı İvme)', line=dict(color='#ef4444', width=2.2)), row=2, col=1)
    fig.update_layout(height=380, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono"))
    return fig

def render_detail(result, strategy, interval):
    st.markdown(f"### {result['ticker'].replace('.IS','')} <span class=\"stag\">{get_sector(result['ticker'])}</span> `{result['score']}/{result['max_score']}`", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"; col="pass" if passed else "fail"
        (c1 if i%2==0 else c2).markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv {col}">{icon} {val}</div></div>', unsafe_allow_html=True)

# ── SIDEBAR SEÇENEKLERİ ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Menü")
    if st.button("🏭 Sektör Özeti Genel Pano", use_container_width=True): st.session_state.page = "sektor"; st.rerun()
    if st.button("📈 TLREF Canlı Faiz Analizi", use_container_width=True): st.session_state.page = "tlref_page"; st.rerun()
    if st.button("📊 Kar/Zarar Performans Paneli", use_container_width=True): st.session_state.page = "perf"; st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔍 Tekli Hisse Arama")
    search_input = st.text_input("Hisse Kodu", placeholder="GARAN...").upper().strip()
    if st.button("Hisse Detay Analiz Et", use_container_width=True):
        if search_input: st.session_state.page = "search"; st.session_state.search_ticker = search_input; st.rerun()

    st.markdown("---")
    st.markdown("### ⚔️ Manuel Makro Tema Seçici")
    chosen_theme = st.selectbox("Senaryo Seç", ["—"] + list(MANUAL_MACRO_THEMES.keys()))
    if st.button("Seçili Temayı Tara", use_container_width=True):
        if chosen_theme != "—": st.session_state.page = "manual_macro_page"; st.session_state.chosen_theme_name = chosen_theme; st.rerun()

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Emre'nin Makro Stratejisi", use_container_width=True, type="primary"): st.session_state.update(strategy="emre", page="scanner", scan_done=False); st.rerun()
with col2:
    if st.button("🟣 Momentum Kırılımcısı", use_container_width=True): st.session_state.update(strategy="momentum", page="scanner", scan_done=False); st.rerun()

# ── SAYFA YÖNLENDİRME AKIŞLARI ────────────────────────────────────────────────
if st.session_state.page == "tlref_page":
    st.markdown("## 📊 Canlı Makroekonomik Faiz Motoru (TLREF)")
    macro = get_tlref_macro_regime()
    mc1, mc2 = st.columns([1.5, 2.5])
    with mc1: st.markdown(f'<div class="mc" style="border-left: 5px solid #a78bfa;"><div class="ml">Mevcut Faiz Rejimi</div><div class="mv" style="font-size:1.3rem; color:#a78bfa;">{macro["regime"]}</div><div style="font-size:0.75rem; color:#64748b; margin-top:5px;">Anlık Oran: %{macro["current_rate"]:.2f}</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="mc" style="border-left: 5px solid #22c55e;"><div class="ml">Desteklenen Sektörler</div><div class="mv" style="font-size:0.95rem; color:#22c55e;">{" · ".join(macro["sectors"])}</div></div>', unsafe_allow_html=True)
    st.plotly_chart(build_tlref_chart(macro["df"]), use_container_width=True)
    
    st.markdown("### 🏭 Rejimin Olumlu Etkilediği Sektörün Hisse Listesi")
    with st.spinner("Hisseler getiriliyor..."): stock_s = fetch_data(BIST100_YF, "5d", "1d")
    list_rows = []
    for tkr, df in stock_s.items():
        if get_sector(tkr) in macro["sectors"] and df is not None and len(df) >= 2:
            c = df['Close'].squeeze()
            list_rows.append({"Hisse": tkr.replace(".IS",""), "Sektör": get_sector(tkr), "Son Fiyat": f"{c.iloc[-1]:.2f} ₺", "Günlük Değişim %": (c.iloc[-1]/c.iloc[-2]-1)*100})
    if list_rows: st.dataframe(pd.DataFrame(list_rows), use_container_width=True, height=400)
    st.stop()

if st.session_state.page == "sektor":
    st.markdown("### 📈 BIST 100 Endeks Doğrulanmış Genel Performansı")
    xu100 = fetch_benchmark(st.session_state.period, st.session_state.get('interval', '1d'))
    if not xu100.empty:
        c_xu = xu100['Close'].squeeze(); r1d = (c_xu.iloc[-1]/c_xu.iloc[-2]-1)*100
        st.metric("BIST 100 Endeksi", f"{c_xu.iloc[-1]:.2f}", f"{r1d:+.2f}%")
    with st.spinner("Sektör getirileri hesaplanıyor..."): stock_s = fetch_data(BIST100_YF, "3mo", "1d")
    summary = build_summary(stock_s)
    st.stop()

if st.session_state.page == "perf":
    st.markdown("## 📊 Geçmiş Dönem Performans & Kâr Alma Defteri")
    if st.button("▶️ Eşit Ağırlıklı Rebalans Simülasyonunu Çalıştır", use_container_width=True):
        with st.spinner("Backtest hesaplanıyor..."):
            bm_df_bt = fetch_benchmark("2y", "1d"); stock_bt = fetch_data(BIST100_YF, "2y", "1d")
            btr = {}
            for sk in ["emre", "momentum"]:
                pv, bn, trd, act, mnt = run_backtest(sk, stock_bt, bm_df_bt)
                btr[sk] = {"pv": pv, "bm": bn, "trades": trd, "monthly": mnt}
            st.session_state.bt_results = btr; st.session_state.bt_done = True
    if st.session_state.get('bt_done'):
        st.markdown("**Aylık Tüm İşlem Hareketleri (Alış, Satış, Ekleme, Kâr Al):**")
        st.dataframe(st.session_state.bt_results["emre"]["trades"], use_container_width=True, height=450)
    st.stop()

if st.session_state.page == "search":
    tk_q = st.session_state.get("search_ticker","").replace(".IS","")
    st.markdown(f"## 🔍 {tk_q} — Analiz Paneli")
    df_s = fetch_single(tk_q+".IS", "6mo", st.session_state.get('interval', '1d'))
    bm_df = fetch_benchmark("6mo", st.session_state.get('interval', '1d'))
    if df_s is not None:
        st.session_state['active_scanning_ticker'] = tk_q+".IS"
        sc, mx, crit, _ = score_emre(df_s, bm_df, tk_q+".IS")
        st.write(f"Emre Makro Skoru: {sc}/{mx}")
    else: st.error("Hisse verisi çekilemedi.")
    st.stop()

if st.session_state.page == "manual_macro_page":
    tname = st.session_state.get("chosen_theme_name","")
    tinfo = MANUAL_MACRO_THEMES[tname]
    st.markdown(f"## {tname}")
    theme_tickers = [t+".IS" for t,s in SECTOR_MAP.items() if s in tinfo["sektörler"]]
    bm_df = fetch_benchmark("6mo","1d"); td = fetch_data(theme_tickers,"6mo","1d")
    m_rows = []
    for ticker, df in td.items():
        if df is None or len(df) < 22: continue
        sc, mx, _, _ = score_emre(df, bm_df, ticker)
        m_rows.append({"Hisse": ticker.replace(".IS",""), "Sektör": get_sector(ticker), "Teknik Puan": f"{sc}/{mx}"})
    if m_rows: st.dataframe(pd.DataFrame(m_rows), use_container_width=True)
    st.stop()

# ── TARAYICI ANA AKIŞ ─────────────────────────────────────────────────────────
if st.session_state.strategy is None: st.info("Yukarıdan bir strateji seçip taramayı başlatın."); st.stop()
strat = st.session_state.strategy
if st.button("🔍 Tüm BIST Listesini Canlı Tara", type="primary", use_container_width=True): st.session_state.update(scan_done=False, results=[])

if not st.session_state.scan_done:
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
l, r = st.columns([1, 2.4], gap="large")
with l:
    for r_item in top5:
        if st.button(f"⭐ {r_item['ticker'].replace('.IS','')} ({r_item['score']})", key=f"t5_{r_item['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r_item["ticker"]; st.rerun()
with r:
    sel = st.session_state.get("selected_ticker")
    if sel:
        res_obj = next((x for x in st.session_state.results if x["ticker"]==sel), None)
        if res_obj: render_detail(res_obj, strat, st.session_state.interval)
