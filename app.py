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
from sectors import (get_sector, get_tlref_macro_regime, update_bist100_and_sectors,
                     SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS)
from sector_summary import build_summary, build_sector_bar_chart, _sector_returns

SECTOR_MAP, BIST100_TICKERS = update_bist100_and_sectors()
BIST100_YF = [t+".IS" for t in BIST100_TICKERS]

st.set_page_config(page_title="BIST Strateji Tarayıcı", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# CSS stilleri (Aynı şekilde korundu)
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

# Veri Çekme Fonksiyonları (Aynı)
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

# İndikatörler (Aynı)
def _c(df): return df['Close'].squeeze().dropna()
def _h(df): return df['High'].squeeze().dropna()
def _l(df): return df['Low'].squeeze().dropna()
def _v(df): return df['Volume'].squeeze().dropna()

def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi_calc(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100/(1+g/l.replace(0, np.nan))).fillna(50)

def stoch(h, l, c, k=14, smooth=3, d=3):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    raw = 100*(c-lo)/(hi-lo).replace(0, np.nan)
    ks = raw.rolling(smooth).mean().fillna(50)
    return ks, ks.rolling(d).mean().fillna(50)

def macd_calc(s, fast=12, slow=26, sig=9):
    m = ema(s,fast)-ema(s,slow); sg = ema(m.fillna(0), sig)
    return m, sg, m-sg

def atr_calc(h, l, c, n=14): return pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(n).mean()
def vol_ratio(v, n=20):
    avg = float(v.iloc[-n-1:-1].mean())
    return float(v.iloc[-1])/avg if avg > 0 else np.nan

def rs_score(c, bm_c, days=20):
    common = c.index.intersection(bm_c.index)
    c2 = c.loc[common]; bm2 = bm_c.loc[common]
    return (float(c2.iloc[-1]/c2.iloc[-days]) - 1) - (float(bm2.iloc[-1]/bm2.iloc[-days]) - 1)

# ── STRATEJİ TANIMLAMALARI ────────────────────────────────────────────────────
def score_emre(df, bm_df):
    try:
        c = _c(df); h = _h(df); l = _l(df); v = _v(df); bm = _c(bm_df)
        price = float(c.iloc[-1]); s20 = float(sma(c,20).iloc[-1]); s50 = float(sma(c,50).iloc[-1]); rsi_v = float(rsi_calc(c).iloc[-1]); rs = rs_score(c, bm, 20); vr = vol_ratio(v, 20)
        criteria = {
            "RS Pozitif (20g vs BIST)": (not np.isnan(rs) and rs > 0, f"{rs*100:.1f}%"),
            "SMA20 & SMA50 Üstünde": (price > s20 and price > s50, f"{price:.2f} > {s20:.2f}"),
            "Hacim Ortalamanın Üstünde": (not np.isnan(vr) and vr >= 0.9, f"{vr:.2f}x"),
            "RSI < 80": (rsi_v < 80, f"RSI={rsi_v:.1f}"),
        }
        details = {"Fiyat": f"{price:.2f} ₺", "SMA 20": f"{s20:.2f}", "SMA 50": f"{s50:.2f}", "RSI (14)": f"{rsi_v:.1f}", "RS (20g)": f"{rs*100:.1f}%", "Hacim Oran.": f"{vr:.2f}x"}
        return sum(1 for p,_ in criteria.values() if p), 4, criteria, details
    except Exception as e: return 0, 4, {"Hata": (False, str(e))}, {}

def score_emre_advanced(df, bm_df, ticker):
    """
    EMRE GELİŞMİŞ STRATEJİSİ: Temel teknik puanı hesaplar, 
    ancak hisse o anki makro faiz rejiminin desteklediği sektörde değilse eler veya cezalandırır.
    """
    sc, mx, criteria, details = score_emre(df, bm_df)
    macro = get_tlref_macro_regime()
    hisse_sektor = get_sector(ticker)
    
    # Makro filtre çarpanı veya katı kuralı
    is_supported = hisse_sektor in macro["sectors"]
    criteria["Makro Rüzgar Desteği"] = (is_supported, "Uyumlu Sektör" if is_supported else "Uyumsuz Sektör")
    
    new_score = sc + 1 if is_supported else sc
    return new_score, mx + 1, criteria, details

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
    "emre": (score_emre, "Emre'nin Stratejisi"),
    "emre_adv": (lambda df, bm: score_emre_advanced(df, bm, st.session_state.get('active_scanning_ticker', '')), "Emre'nin Stratejisi (Gelişmiş)"),
    "momentum": (score_momentum, "Momentum Kırılımcısı")
}

# Grafik Fonksiyonu (Aynı)
def build_chart(df, ticker, strategy, interval):
    c = _c(df); h = _h(df); l = _l(df); v = _v(df); o = df['Open'].squeeze()
    s20 = sma(c,20); s50 = sma(c,50); ml, sl, hl_s = macd_calc(c); rsi_line = rsi_calc(c)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48,0.18,0.18,0.16], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name="Fiyat", increasing_fillcolor="#22c55e", increasing_line_color="#22c55e", decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s20, name="SMA 20", line=dict(color="#22c55e",width=1.5,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50", line=dict(color="#f59e0b",width=1.5,dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_line, name="RSI", line=dict(color="#38bdf8",width=1.5)), row=2, col=1)
    fig.add_hrect(y0=80,y1=100,fillcolor="#ef4444",opacity=0.07,row=2,col=1)
    fig.add_hline(y=80,line=dict(color="#ef4444",width=0.7,dash="dot"),row=2,col=1)
    fig.update_layout(height=700, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", color="#94a3b8", size=11), margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False, title=dict(text=f"<b>{ticker.replace('.IS','')}</b>", font=dict(color="#f1f5f9",size=14),x=0.01), hovermode="x unified")
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#1e2535",row=i,col=1)
        fig.update_yaxes(gridcolor="#1e2535",row=i,col=1)
    return fig

# ── TLREF GRAFİK OLUŞTURUCU (YENİ) ────────────────────────────────────────────
def build_tlref_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df['TLREF'], name='Canlı TLREF', line=dict(color='#e2e8f0', width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA8'], name='SMA 8 (2 Ay)', line=dict(color='#22c55e', width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA54'], name='SMA 54 (1 Yıl)', line=dict(color='#f59e0b', width=1.5, dash='dash')), row=1, col=1)
    
    # ADX / DMI Bölümü
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX (Trend Şiddeti)', line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D+'], name='D+ (Alıcılar)', line=dict(color='#38bdf8', width=1.2, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D-'], name='D- (Satıcılar)', line=dict(color='#ef4444', width=1.2, dash='dot')), row=2, col=1)
    
    fig.update_layout(height=400, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(family="JetBrains Mono", color="#94a3b8", size=10), margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
    return fig

def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS","")
    st.markdown(f"### {ticker_label} " f'<span class="stag">{get_sector(result["ticker"])}</span> ' f'`{result["score"]}/{result["max_score"]}`', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i,(k,(passed,val)) in enumerate(result["criteria"].items()):
        icon="✅" if passed else "❌"; col="pass" if passed else "fail"
        st.markdown(f'<div class="mc"><div class="ml">{k}</div><div class="mv {col}">{icon} {val}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">📈 Grafik</div>', unsafe_allow_html=True)
    st.plotly_chart(build_chart(result["df"], result["ticker"], strategy, interval), use_container_width=True, config={"displayModeBar":False})

# Header & States
st.markdown('<div class="main-header">📊 BIST Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y %H:%M")} · Canlı Entegrasyon Altyapısı</div>', unsafe_allow_html=True)

defaults = {"strategy":None,"selected_ticker":None,"scan_done":False,"results":[],"page":"scanner","bt_done":False,"bt_results":{},"bm_df":None,"ai_messages":[]}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d","4h","1wk"], format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    period = {"1d":"6mo","4h":"60d","1wk":"2y"}[interval]
    st.markdown("---")
    st.markdown("### 🤖 Sekreter API Anahtarı")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AI Sekreter için anahtar girin...")
    if gemini_key: genai.configure(api_key=gemini_key)
    st.markdown("---")
    if st.button("🤖 Finansal Sekreter Chat", use_container_width=True, type="primary"):
        st.session_state.page = "ai_secretary"; st.rerun()
    if st.button("🏭 Sektör Özeti & TLREF", use_container_width=True):
        st.session_state.page = "sektor"; st.rerun()
    if st.button("📊 Performans", use_container_width=True):
        st.session_state.page = "perf"; st.rerun()

# Üst Butonlar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟡 Emre'nin Stratejisi", use_container_width=True):
        st.session_state.update(strategy="emre", page="scanner", scan_done=False); st.rerun()
with col2:
    if st.button("🔥 Emre'nin Stratejisi (Gelişmiş)", use_container_width=True, type="primary"):
        st.session_state.update(strategy="emre_adv", page="scanner", scan_done=False); st.rerun()
with col3:
    if st.button("🟣 Momentum Kırılımcısı", use_container_width=True):
        st.session_state.update(strategy="momentum", page="scanner", scan_done=False); st.rerun()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: SEKTÖR ÖZETİ & TLREF MOTORU (GÜNCELLENDİ)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "sektor":
    macro = get_tlref_macro_regime()
    st.markdown(f"## 📊 Canlı Makroekonomik Faiz Motoru (TLREF)")
    
    # 3'lü Gösterge Panosu
    mc1, mc2 = st.columns([1.5, 2.5])
    with mc1:
        st.markdown(f'<div class="mc" style="border-left: 5px solid #a78bfa;"><div class="ml">Mevcut Faiz Rejimi</div><div class="mv" style="font-size:1.3rem; color:#a78bfa;">{macro["regime"]}</div><div style="font-size:0.75rem; color:#64748b; margin-top:5px;">Canlı TLREF Oranı: %{macro["current_rate"]:.2f}</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="mc" style="border-left: 5px solid #22c55e;"><div class="ml">Desteklenen Sektörler</div><div class="mv" style="font-size:0.95rem; color:#22c55e;">{" · ".join(macro["sectors"])}</div><div style="font-size:0.75rem; color:#94a3b8; margin-top:5px;">{macro["description"]}</div></div>', unsafe_allow_html=True)
        
    st.plotly_chart(build_tlref_chart(macro["df"]), use_container_width=True, config={"displayModeBar":False})
    
    st.markdown("---")
    st.markdown("### 🏭 BİST Sektörel Dağılım ve Momentum Oranları")
    with st.spinner("Sektör momentumları hesaplanıyor..."):
        stock_s = fetch_data(BIST100_YF,"3mo","1d")
    summary = build_summary(stock_s)
    bar_chart = build_sector_bar_chart(stock_s)
    if bar_chart: st.plotly_chart(bar_chart, use_container_width=True, config={"displayModeBar":False})
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: GÖREN AI FINANSAL SEKRETER (YENİ)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "ai_secretary":
    st.markdown("## 🤖 Finansal Sekreter (Sistem Bilgisine Hakim)")
    if not gemini_key:
        st.warning("⚠️ Lütfen sol menüden bir Gemini API Key giriniz.")
    else:
        macro_info = get_tlref_macro_regime()
        top_stocks_context = [r['ticker'].replace(".IS","") for r in st.session_state.results if r.get('in_top5')]
        
        # AI'ın körlüğünü bitiren gizli sistem promptu injection katmanı
        system_prompt = f"""Sen Emre'nin gelişmiş algo-trading finans asistanısın. Uygulamanın hesapladığı canlı veriler aşağıdadır:
        - Mevcut Makro Faiz Rejimi: {macro_info['regime']}
        - Rejimin Olumlu Etkilediği Sektörler: {', '.join(macro_info['sectors'])}
        - Tarayıcıda Çıkan En Son Top Hisse Önerileri: {', '.join(top_stocks_context) if top_stocks_context else 'Henüz tarama yapılmadı.'}
        Bu bilgilere tam hakim olarak konuş. Cevaplarını kısa, veri odaklı ve profesyonel bir quat analisti gibi ver. Asla varant (warrant) araçlarından bahsetme, kapsam dışıdır."""
        
        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
                
        user_query = st.chat_input("Mevcut faiz rejimi ve hisseler hakkında soru sorun...")
        if user_query:
            with st.chat_message("user"): st.write(user_query)
            st.session_state.ai_messages.append({"role": "user", "content": user_query})
            
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                with st.spinner("Analiz ediliyor..."):
                    response = model.generate_content(f"{system_prompt}\n\nKullanıcı: {user_query}")
                with st.chat_message("assistant"): st.write(response.text)
                st.session_state.ai_messages.append({"role": "assistant", "content": response.text})
            except Exception as e: st.error(f"Hata: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SAYFA: SCANNER / TARAYICI
# ═══════════════════════════════════════════════════════════════════
if st.session_state.strategy is None:
    st.info("👆 Yukarıdaki butonlardan çalıştırmak istediğin stratejiyi seç.")
    st.stop()

strat = st.session_state.strategy
st.markdown(f"**Aktif Strateji Modülü:** `{STRATEGY_FN[strat][1]}` · Zaman Dilimi: `{interval.upper()}`")

if st.button("🔍 Tüm BIST Listesini Canlı Tara", type="primary", use_container_width=True):
    st.session_state.update(scan_done=False, results=[], selected_ticker=None)

if not st.session_state.scan_done:
    with st.spinner("Veriler işleniyor..."):
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
    
    # Sektör Limit Atama Kontrolleri
    t_count = 0; s_counts = {}
    for r in results:
        sect = r['sector']
        if t_count < 5 and (strat != "emre" and strat != "emre_adv" or s_counts.get(sect, 0) < 2):
            r['in_top5'] = True; s_counts[sect] = s_counts.get(sect, 0) + 1; t_count += 1
            
    st.session_state.update(results=results, scan_done=True); st.rerun()

# Listeleme Arayüzü (Aynı şekilde render_detail fonksiyonuna bağlanır)
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
