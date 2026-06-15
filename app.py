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
from sectors import get_sector, get_theme_sectors, MACRO_THEMES, SECTOR_MAP
from sector_summary import build_summary, build_sector_bar_chart

st.set_page_config(page_title="BIST 100 Strateji Tarayıcı", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background-color:#0d0f14;color:#e2e8f0;}
.main-header{font-size:1.8rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.03em;margin-bottom:.2rem;}
.sub-header{font-size:.85rem;color:#64748b;margin-bottom:1.5rem;font-family:'JetBrains Mono',monospace;}
.metric-card{background:#161b27;border:1px solid #1e2535;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.6rem;}
.metric-label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace;margin-bottom:.2rem;}
.metric-value{font-size:1.1rem;font-weight:600;color:#f1f5f9;font-family:'JetBrains Mono',monospace;}
.metric-pass{color:#22c55e!important;} .metric-fail{color:#ef4444!important;}
.section-title{font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.1em;
  font-family:'JetBrains Mono',monospace;margin:1.2rem 0 .6rem 0;border-bottom:1px solid #1e2535;padding-bottom:.4rem;}
div[data-testid="stButton"] button{border-radius:8px;font-weight:500;font-size:.85rem;}
.sector-tag{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.68rem;
  font-family:'JetBrains Mono',monospace;background:#1e2535;color:#94a3b8;margin-left:.4rem;}
.perf-box{background:#161b27;border:1px solid #1e2535;border-radius:12px;padding:1.2rem;text-align:center;}
.perf-val{font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.perf-label{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem;}
</style>
""", unsafe_allow_html=True)

# ── BIST 100 ──────────────────────────────────────────────────────────────────
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
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
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
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_ratio(v, n=20):
    avg = v.rolling(n).mean().iloc[-1]
    return v.iloc[-1] / avg if avg and not np.isnan(avg) and avg > 0 else np.nan

def rs_score(c, bm_c, days=20):
    if len(c) < days or len(bm_c) < days: return np.nan
    sr = (float(c.iloc[-1]) / float(c.iloc[-days])) - 1
    br = (float(bm_c.iloc[-1]) / float(bm_c.iloc[-days])) - 1
    return sr - br

# ── TEMEL ANALİZ (yfinance) ───────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "F/K":      info.get("trailingPE"),
            "PD/DD":    info.get("priceToBook"),
            "Kar Büyümesi": info.get("earningsGrowth"),
            "ROE":      info.get("returnOnEquity"),
            "Borç/Özsermaye": info.get("debtToEquity"),
        }
    except Exception:
        return {}

# ── STRATEJİ 1: EMRENİN STRATEJİSİ ──────────────────────────────────────────
def score_emre(df, bm_df):
    c  = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();   v = df['Volume'].squeeze()
    bm = bm_df['Close'].squeeze()

    if len(c) < 55: return 0, 4, {}, {}

    price  = float(c.iloc[-1])
    s20    = float(sma(c, 20).iloc[-1])
    s50    = float(sma(c, 50).iloc[-1])
    rsi_v  = float(rsi_calc(c).iloc[-1])
    rs     = rs_score(c, bm, 20)

    # Hacim konfirmasyonu
    vol5   = float(v.iloc[-5:].mean())
    vol20  = float(v.rolling(20).mean().iloc[-1])
    last10 = pd.DataFrame({'close': c.iloc[-10:], 'vol': v.iloc[-10:]})
    up_vol = float(last10[last10['close'] > last10['close'].shift()]['vol'].sum())
    dn_vol = float(last10[last10['close'] < last10['close'].shift()]['vol'].sum())

    criteria = {
        "RS Pozitif (20g vs BIST)":     (rs is not None and not np.isnan(rs) and rs > 0,
                                          f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A"),
        "SMA20 & SMA50 Üstünde":        (price > s20 and price > s50,
                                          f"{price:.2f} > {s20:.2f} / {s50:.2f}"),
        "Hacim Konfirmasyonu":           (vol5 > vol20 and up_vol > dn_vol,
                                          f"5g={vol5/vol20:.2f}x · ↑{up_vol/1e6:.1f}M > ↓{dn_vol/1e6:.1f}M"),
        "RSI < 80":                      (rsi_v < 80,
                                          f"RSI={rsi_v:.1f}"),
    }
    details = {
        "Fiyat":      f"{price:.2f} ₺",
        "SMA 20":     f"{s20:.2f}",
        "SMA 50":     f"{s50:.2f}",
        "RS (20g)":   f"{rs*100:.1f}%" if rs is not None and not np.isnan(rs) else "N/A",
        "RSI (14)":   f"{rsi_v:.1f}",
        "Hacim 5g/20g": f"{vol5/vol20:.2f}x" if vol20 > 0 else "N/A",
        "↑Hacim/↓Hacim": f"{up_vol/dn_vol:.2f}x" if dn_vol > 0 else "N/A",
    }
    sc = sum(1 for passed, _ in criteria.values() if passed)
    return sc, len(criteria), criteria, details

# RS ham skoru (sıralama için)
def raw_rs(df, bm_df):
    try:
        c  = df['Close'].squeeze()
        bm = bm_df['Close'].squeeze()
        return rs_score(c, bm, 20)
    except Exception:
        return np.nan

# ── STRATEJİ 2: MOMENTUM KIRILIMCISI ─────────────────────────────────────────
def score_momentum(df, bm_df):
    c  = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();   v = df['Volume'].squeeze()

    s50 = sma(c, 50).iloc[-1]
    ml, sl, hl = macd_calc(c)
    hn = hl.iloc[-1]; hp = hl.iloc[-2]
    a14 = atr_calc(h, lo, c)
    atr_now = a14.iloc[-1]; atr_avg = a14.rolling(60).mean().iloc[-1]
    vr  = vol_ratio(v)
    price = float(c.iloc[-1])
    rsi_v = float(rsi_calc(c).iloc[-1])

    criteria = {
        "SMA 50 Üzerinde":             (price > float(s50),
                                         f"{price:.2f} > {float(s50):.2f}"),
        "MACD Hist Pozitif & Artıyor": (float(hn) > 0 and float(hn) > float(hp),
                                         f"{float(hn):.4f}"),
        "ATR Genişliyor":              (not np.isnan(float(atr_avg)) and float(atr_now) > float(atr_avg),
                                         f"{float(atr_now):.2f} > {float(atr_avg):.2f}"),
        "Hacim > 1.2x Ort.":          (not np.isnan(vr) and vr > 1.2,
                                         f"{vr:.2f}x"),
    }
    details = {
        "Fiyat":      f"{price:.2f} ₺",
        "SMA 50":     f"{float(s50):.2f}",
        "RSI (14)":   f"{rsi_v:.1f}",
        "MACD Hist":  f"{float(hn):.4f}",
        "ATR (14)":   f"{float(atr_now):.2f}",
        "ATR Ort(60)":f"{float(atr_avg):.2f}" if not np.isnan(float(atr_avg)) else "N/A",
        "Hacim Oranı":f"{vr:.2f}x" if not np.isnan(vr) else "N/A",
    }
    sc = sum(1 for passed, _ in criteria.values() if passed)
    return sc, len(criteria), criteria, details

STRATEGY_FN = {
    "emre":     (score_emre,     "Emre'nin Stratejisi"),
    "momentum": (score_momentum, "Momentum Kırılımcısı"),
}

# ── EMRE STRATEJİSİ TAM TARAMA (sektör filtreli top5) ───────────────────────
def run_emre_scan(stock_data, bm_df):
    """
    RS sıralama → Filtre 2,3,4 → Sektör çeşitlendirme (max 2) → Top 5
    Returns: list of dicts
    """
    bm = bm_df['Close'].squeeze()

    # Adım 1: RS hesapla, sıralanmış liste yap
    candidates = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 55: continue
        try:
            c   = df['Close'].squeeze()
            rs  = rs_score(c, bm, 20)
            if rs is None or np.isnan(rs) or rs <= 0: continue
            candidates.append({'ticker': ticker, 'rs': rs, 'df': df})
        except Exception:
            pass

    candidates.sort(key=lambda x: x['rs'], reverse=True)
    top15 = candidates[:15]

    # Adım 2,3,4: Filtrele
    filtered = []
    for cand in top15:
        df  = cand['df']
        c   = df['Close'].squeeze(); h = df['High'].squeeze()
        lo  = df['Low'].squeeze();   v = df['Volume'].squeeze()
        price = float(c.iloc[-1])

        # Filtre 2: SMA20 & SMA50
        s20 = float(sma(c, 20).iloc[-1]); s50 = float(sma(c, 50).iloc[-1])
        if price <= s20 or price <= s50: continue

        # Filtre 3: Hacim konfirmasyonu
        vol5  = float(v.iloc[-5:].mean())
        vol20 = float(v.rolling(20).mean().iloc[-1])
        last10 = pd.DataFrame({'close': c.iloc[-10:], 'vol': v.iloc[-10:]})
        up_vol = float(last10[last10['close'] > last10['close'].shift()]['vol'].sum())
        dn_vol = float(last10[last10['close'] < last10['close'].shift()]['vol'].sum())
        if vol5 <= vol20 or up_vol <= dn_vol: continue

        # Filtre 4: RSI < 80
        rsi_v = float(rsi_calc(c).iloc[-1])
        if rsi_v >= 80: continue

        sector = get_sector(cand['ticker'])
        cand['sector'] = sector
        cand['price']  = price
        cand['rsi']    = rsi_v
        filtered.append(cand)

    # Filtre 5: Sektör çeşitlendirme (max 2 aynı sektör)
    sector_count = {}
    final = []
    for cand in filtered:
        sec = cand['sector']
        if sector_count.get(sec, 0) >= 2: continue
        sector_count[sec] = sector_count.get(sec, 0) + 1
        final.append(cand)
        if len(final) >= 5: break

    return final, filtered   # final=top5, filtered=tüm geçenler (sektör öncesi)

# ── GRAFİK ────────────────────────────────────────────────────────────────────
def build_chart(df, ticker, strategy, interval):
    c  = df['Close'].squeeze(); h = df['High'].squeeze()
    lo = df['Low'].squeeze();   v = df['Volume'].squeeze()
    o  = df['Open'].squeeze()

    s20 = sma(c, 20); s50 = sma(c, 50)
    e20 = ema(c, 20)
    k_line, d_line = stoch(h, lo, c)
    ml, sl, hl_s = macd_calc(c)
    rsi_line = rsi_calc(c)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48,0.18,0.18,0.16],
                        vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=lo, close=c,
        name="Fiyat",
        increasing_fillcolor="#22c55e", increasing_line_color="#22c55e",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
        line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=s20, name="SMA 20",
        line=dict(color="#22c55e", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=s50, name="SMA 50",
        line=dict(color="#f59e0b", width=1.5, dash="dash")), row=1, col=1)

    # RSI paneli
    fig.add_trace(go.Scatter(x=df.index, y=rsi_line, name="RSI",
        line=dict(color="#38bdf8", width=1.5)), row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.07, row=2, col=1)
    fig.add_hline(y=80, line=dict(color="#ef4444", width=0.7, dash="dot"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#64748b", width=0.5, dash="dot"), row=2, col=1)

    bar_colors = ["#22c55e" if x >= 0 else "#ef4444" for x in hl_s]
    fig.add_trace(go.Bar(x=df.index, y=hl_s, name="MACD Hist",
        marker_color=bar_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ml, name="MACD",
        line=dict(color="#38bdf8", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sl, name="Signal",
        line=dict(color="#f97316", width=1.2, dash="dot")), row=3, col=1)

    vcols = ["#22c55e" if cv >= ov else "#ef4444" for cv, ov in zip(c, o)]
    fig.add_trace(go.Bar(x=df.index, y=v, name="Hacim",
        marker_color=vcols, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=v.rolling(20).mean(), name="Hacim MA20",
        line=dict(color="#f59e0b", width=1.2)), row=4, col=1)

    strat_label = STRATEGY_FN[strategy][1]
    fig.update_layout(
        height=700, paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
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

# ── AKTİF PORTFÖY PERFORMANS KUTUCUKLARI ─────────────────────────────────────
def render_perf_boxes(active_positions, bm_df):
    """Aktif pozisyonların alım gününden itibaren % ve BIST100 karşılaştırması"""
    if not active_positions: return
    bm = bm_df['Close'].squeeze()
    st.markdown('<div class="section-title">📊 Bu Ay Portföy Performansı (Alım Gününden İtibaren)</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(active_positions))
    for col, pos in zip(cols, active_positions):
        pnl  = pos.get('pnl_pct', 0)
        tkr  = pos.get('ticker', '')
        sect = get_sector(tkr)
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        col.markdown(
            f'<div class="perf-box">'
            f'<div style="font-size:.85rem;font-weight:700;color:#f1f5f9">{tkr}</div>'
            f'<div style="font-size:.65rem;color:#64748b;margin-bottom:.5rem">{sect}</div>'
            f'<div class="perf-val" style="color:{pnl_color}">{pnl:+.1f}%</div>'
            f'<div class="perf-label">Alım: {pos.get("buy_date","")}</div>'
            f'<div style="font-size:.7rem;color:#64748b;margin-top:.3rem">'
            f'₺{pos.get("buy_price",0):.2f} → ₺{pos.get("current_price",0):.2f}</div>'
            f'</div>', unsafe_allow_html=True)

# ── DETAY PANELİ ──────────────────────────────────────────────────────────────
def render_detail(result, strategy, interval):
    ticker_label = result["ticker"].replace(".IS", "")
    sector       = get_sector(result["ticker"])
    st.markdown(f"### {ticker_label} "
                f'<span class="sector-tag">{sector}</span>'
                f"  `{result['score']}/{result['max_score']}`",
                unsafe_allow_html=True)

    # Temel analiz
    with st.expander("📋 Temel Analiz", expanded=False):
        fund = fetch_fundamentals(result["ticker"])
        if fund:
            fc = st.columns(len(fund))
            for i, (k, v2) in enumerate(fund.items()):
                val_str = f"{v2:.2f}" if isinstance(v2, float) and not np.isnan(v2) else (str(v2) if v2 else "N/A")
                fc[i].markdown(
                    f'<div class="metric-card"><div class="metric-label">{k}</div>'
                    f'<div class="metric-value" style="font-size:.95rem">{val_str}</div></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown("*Temel veri alınamadı.*")

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
    for i, (k, v2) in enumerate(result["details"].items()):
        dc[i % 3].markdown(
            f'<div class="metric-card"><div class="metric-label">{k}</div>'
            f'<div class="metric-value" style="font-size:.95rem">{v2}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Grafik</div>', unsafe_allow_html=True)
    fig = build_chart(result["df"], result["ticker"], strategy, interval)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">📊 BIST 100 Strateji Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{datetime.now().strftime("%d.%m.%Y %H:%M")} · {len(BIST100_TICKERS)} hisse · yfinance</div>',
            unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    interval = st.selectbox("Zaman Dilimi", ["1d", "4h", "1wk"],
        format_func=lambda x: {"1d":"Günlük (1D)","4h":"4 Saatlik (4H)","1wk":"Haftalık (1W)"}[x])
    period_map = {"1d":"6mo","4h":"60d","1wk":"2y"}
    period = period_map[interval]

    st.markdown("---")
    st.markdown("### 🔍 Hisse Ara")
    search_query = st.text_input("Ticker", placeholder="GARAN...").upper().strip()
    search_btn   = st.button("Ara", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌍 Makro Tema Filtresi")
    macro_theme  = st.selectbox("Tema seç", ["—"] + list(MACRO_THEMES.keys()))
    macro_btn    = st.button("Tema'ya Göre Tara", use_container_width=True)

    st.markdown("---")
    perf_btn   = st.button("📊 Performans",      use_container_width=True, type="primary")
    sektor_btn = st.button("🏭 Sektör Özeti",    use_container_width=True)

    st.markdown("---")
    st.markdown("**🟡 Emre:** RS>0 · SMA20&50 · Hacim ↑ · RSI<80 · Max 2/sektör")
    st.markdown("**🟣 Momentum:** SMA50 · MACD artan · ATR genişleme · Hacim")

# ── BUTONLAR ──────────────────────────────────────────────────────────────────
col_b1, col_b2, col_b3 = st.columns([2, 2, 5])
with col_b1:
    btn_emre = st.button("🟡 Emre'nin Stratejisi", use_container_width=True, type="primary")
with col_b2:
    btn_mo   = st.button("🟣 Momentum Kırılımcısı", use_container_width=True)

for key, default in [("strategy", None), ("selected_ticker", None),
                      ("scan_done", False), ("results", []),
                      ("page", "scanner"), ("bt_done", False), ("bt_results", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

if btn_emre:   st.session_state.update(strategy="emre",     selected_ticker=None, scan_done=False, page="scanner")
if btn_mo:     st.session_state.update(strategy="momentum", selected_ticker=None, scan_done=False, page="scanner")
if perf_btn:   st.session_state.update(page="perf")
if sektor_btn: st.session_state.update(page="sektor")

# ── SEKTÖR ÖZET SAYFASI ──────────────────────────────────────────────────────
if st.session_state.page == "sektor":
    st.markdown("## 🏭 BİST 100 Günün ve Haftanın Sektörel Özeti")
    st.markdown(f"*{datetime.now().strftime('%d.%m.%Y %H:%M')}*")
    st.markdown("---")

    with st.spinner("Sektör verileri hesaplanıyor..."):
        bm_df_s    = fetch_benchmark("3mo", "1d")
        stock_s    = fetch_data(BIST100_YF, "3mo", "1d")
        summary    = build_summary(stock_s)
        bar_chart  = build_sector_bar_chart(stock_s)

    # Kategori renk & border tanımları
    CAT_STYLE = {
        "🚀 BUGÜN LİDER":    ("#14532d", "#22c55e"),
        "⚡ İVME KAZANAN":   ("#14532d", "#4ade80"),
        "📈 1 HAFTA ZİRVE":  ("#14532d", "#86efac"),
        "🏆 1 AY ZİRVE":     ("#14532d", "#bbf7d0"),
        "🔄 TOPARLAYAN":     ("#1c3a1c", "#a3e635"),
        "📉 BUGÜN GERİDE":   ("#4c0519", "#ef4444"),
        "❄️ 1 HAFTA DİP":   ("#4c0519", "#f87171"),
        "🪨 1 AY DİP":       ("#4c0519", "#fca5a5"),
        "🐌 YAVAŞLAYAN":     ("#2c1810", "#fb923c"),
    }

    def render_cat_card(title, items, bg, border):
        rows_html = ""
        for sect, val in items:
            val_color = "#22c55e" if val.startswith("+") else "#ef4444" if val.startswith("-") else "#94a3b8"
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:.35rem 0;border-bottom:1px solid #1e2535;font-size:.85rem;">'
                f'<span style="color:#e2e8f0">{sect}</span>'
                f'<span style="color:{val_color};font-family:JetBrains Mono;font-weight:600">{val}</span>'
                f'</div>'
            )
        return (
            f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
            f'padding:1rem 1.2rem;margin-bottom:.8rem;height:100%">'
            f'<div style="font-size:.72rem;font-weight:700;color:{border};text-transform:uppercase;'
            f'letter-spacing:.1em;margin-bottom:.6rem;font-family:JetBrains Mono">{title}</div>'
            f'{rows_html}</div>'
        )

    # Manşet bölümü — 3 sütun × 3 satır
    CAT_ORDER = [
        ["🚀 BUGÜN LİDER",   "⚡ İVME KAZANAN"],
        ["📈 1 HAFTA ZİRVE", "🏆 1 AY ZİRVE"],
        ["🔄 TOPARLAYAN",    "📉 BUGÜN GERİDE"],
        ["❄️ 1 HAFTA DİP",  "🪨 1 AY DİP"],
        ["🐌 YAVAŞLAYAN",   None],
    ]

    st.markdown("### 📰 Günün Manşetleri")
    for row_cats in CAT_ORDER:
        cols = st.columns(2)
        for col, cat in zip(cols, row_cats):
            if cat is None: continue
            if cat not in summary: continue
            bg, border = CAT_STYLE.get(cat, ("#161b27","#475569"))
            col.markdown(render_cat_card(cat, summary[cat], bg, border),
                         unsafe_allow_html=True)

    # Bar chart
    st.markdown("---")
    st.markdown("### 📊 Sektör Getiri Karşılaştırması")
    st.markdown("*Bugün varsayılan · Haftalık ve Aylık legend'dan açılabilir*")
    if bar_chart:
        st.plotly_chart(bar_chart, use_container_width=True, config={"displayModeBar": False})

    # Tam sektör tablosu
    st.markdown("---")
    st.markdown("### 📋 Tüm Sektörler Detay Tablosu")
    from sector_summary import _sector_returns
    full_df = _sector_returns(stock_s)
    if not full_df.empty:
        full_df = full_df.sort_values('ret_1d', ascending=False).reset_index(drop=True)
        full_df.columns = ['Sektör','Bugün %','1 Hafta %','1 Ay %','İvme (5g)','İvme (21g)','Hisse Sayısı']
        for col_name in ['Bugün %','1 Hafta %','1 Ay %']:
            full_df[col_name] = full_df[col_name].apply(lambda x: f"{x:+.2f}%" if not np.isnan(x) else "N/A")

        def color_ret(val):
            if isinstance(val, str) and val.startswith('+'): return 'color: #22c55e'
            elif isinstance(val, str) and val.startswith('-'): return 'color: #ef4444'
            return ''

        styled = full_df.style.map(color_ret, subset=['Bugün %','1 Hafta %','1 Ay %'])
        st.dataframe(styled, use_container_width=True, height=500)

    st.stop()

# ── PERFORMANS SAYFASI ────────────────────────────────────────────────────────
if st.session_state.page == "perf":
    st.markdown("## 📊 Strateji Performans Karşılaştırması")
    st.markdown("Her ayın ilk borsa günü top 5 hisse · Günlük · 2 Yıl · 100.000 ₺")
    st.markdown("---")

    run_bt_btn = st.button("▶️ Backtest'i Çalıştır", type="primary")
    if run_bt_btn or not st.session_state.bt_done:
        with st.spinner("2 yıllık veri çekiliyor..."):
            bm_df_bt = fetch_benchmark("2y", "1d")
            stock_bt = fetch_data(BIST100_YF, "2y", "1d")
        bt_results = {}
        strat_keys = ["emre", "momentum"]
        prog = st.progress(0)
        for i, sk in enumerate(strat_keys):
            prog.progress((i+1)/len(strat_keys), text=f"{STRATEGY_FN[sk][1]} hesaplanıyor...")
            pv, bm_n, trades, active, monthly = run_backtest(sk, stock_bt, bm_df_bt)
            bt_results[sk] = {
                "pv": pv, "bm": bm_n, "trades": trades, "active": active,
                "monthly": monthly,
                "stats": calc_stats(pv, bm_n, 100_000) if pv is not None else {}
            }
        prog.empty()
        st.session_state.bt_results  = bt_results
        st.session_state.bt_bm       = bm_df_bt
        st.session_state.bt_done     = True

    bt_results = st.session_state.bt_results
    if not bt_results:
        st.info("▶️ Backtest'i çalıştır butonuna bas.")
        st.stop()

    # Aktif portföyler
    st.markdown("### 📌 Bu Ay Aktif Portföyler")
    col1, col2 = st.columns(2)
    for col, sk in zip([col1, col2], ["emre","momentum"]):
        with col:
            label  = STRATEGY_FN[sk][1]
            active = bt_results[sk].get("active", [])
            st.markdown(f"**{label}** — {len(active)} hisse")
            for pos in sorted(active, key=lambda x: x['pnl_pct'], reverse=True):
                pnl   = pos['pnl_pct']
                sect  = get_sector(pos['ticker'])
                color = "#22c55e" if pnl >= 0 else "#ef4444"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<span style="font-weight:700;color:#f1f5f9">{pos["ticker"]}</span>'
                    f'<span class="sector-tag">{sect}</span>'
                    f'<span style="color:{color};font-family:JetBrains Mono;font-weight:600">{pnl:+.1f}%</span>'
                    f'</div>'
                    f'<div style="font-size:.72rem;color:#64748b;margin-top:.3rem">'
                    f'Alış: {pos["buy_date"]} · ₺{pos["buy_price"]:.2f} → ₺{pos["current_price"]:.2f}</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Portföy Performansı vs BIST 100")
    perf_map = {sk: (bt_results[sk]["pv"], bt_results[sk]["bm"])
                for sk in ["emre","momentum"] if bt_results[sk].get("pv") is not None}
    if perf_map:
        st.plotly_chart(build_perf_chart(perf_map, 100_000),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("### 📐 Özet İstatistikler")
    stat_cols = st.columns(2)
    for col, sk in zip(stat_cols, ["emre","momentum"]):
        with col:
            st.markdown(f"**{STRATEGY_FN[sk][1]}**")
            for k, v2 in bt_results[sk].get("stats", {}).items():
                is_pos = "+" in str(v2) and "Drawdown" not in k
                is_neg = str(v2).startswith("-") or "Drawdown" in k
                vc = "#22c55e" if is_pos else "#ef4444" if is_neg else "#f1f5f9"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{k}</div>'
                    f'<div class="metric-value" style="color:{vc};font-size:1rem">{v2}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📅 Aylık Portföy Detayı")
    monthly_tabs = st.tabs([STRATEGY_FN["emre"][1], STRATEGY_FN["momentum"][1]])
    for tab, sk in zip(monthly_tabs, ["emre","momentum"]):
        with tab:
            mdf = bt_results[sk].get("monthly")
            if mdf is not None and len(mdf) > 0:
                def color_pnl(val):
                    if isinstance(val, str) and val.startswith('+'): return 'color:#22c55e'
                    elif isinstance(val, str) and val.startswith('-'): return 'color:#ef4444'
                    return ''
                cols_to_style = ['Aylık P&L'] if 'Aylık P&L' in mdf.columns else []
                styled = mdf.style.map(color_pnl, subset=cols_to_style)
                st.dataframe(styled, use_container_width=True, height=600)

    st.markdown("---")
    st.markdown("### 📋 İşlem Geçmişi")
    log_tabs = st.tabs([STRATEGY_FN["emre"][1], STRATEGY_FN["momentum"][1]])
    for tab, sk in zip(log_tabs, ["emre","momentum"]):
        with tab:
            trades = bt_results[sk].get("trades")
            if trades is not None and len(trades) > 0:
                sales = trades[trades["İşlem"] == "SATIŞ"] if "İşlem" in trades.columns else trades
                if len(sales): st.dataframe(sales.reset_index(drop=True), use_container_width=True, height=300)
    st.stop()

# ── MAKRO TEMA SAYFASI ────────────────────────────────────────────────────────
if macro_btn and macro_theme != "—":
    st.markdown(f"## 🌍 {macro_theme} — Sektör Taraması")
    theme_sectors = get_theme_sectors(macro_theme)
    st.markdown(f"**Hedef sektörler:** {' · '.join(theme_sectors)}")
    st.markdown("---")

    theme_tickers = [t + ".IS" for t, s in SECTOR_MAP.items()
                     if s in theme_sectors and t in BIST100_TICKERS]

    with st.spinner("Veri çekiliyor..."):
        bm_df   = fetch_benchmark(period, interval)
        td      = fetch_data(theme_tickers, period, interval)

    rows = []
    for ticker, df in td.items():
        if df is None or len(df) < 55: continue
        try:
            sc_e, mx_e, _, _ = score_emre(df, bm_df)
            sc_m, mx_m, _, _ = score_momentum(df, bm_df)
            c  = df['Close'].squeeze()
            bm = bm_df['Close'].squeeze()
            rs = rs_score(c, bm, 20)
            rows.append({
                'Hisse':    ticker.replace('.IS',''),
                'Sektör':   get_sector(ticker),
                'Emre':     f"{sc_e}/{mx_e}",
                'Momentum': f"{sc_m}/{mx_m}",
                'RS (20g)': f"{rs*100:.1f}%" if rs and not np.isnan(rs) else "N/A",
                'Fiyat':    f"₺{float(c.iloc[-1]):.2f}",
                '_emre_sc': sc_e, '_mom_sc': sc_m,
            })
        except Exception:
            pass

    if rows:
        df_rows = pd.DataFrame(rows).sort_values('_emre_sc', ascending=False)
        display = df_rows.drop(columns=['_emre_sc','_mom_sc'])
        st.dataframe(display, use_container_width=True, height=400)

        st.markdown("**Hisse detayı için aşağıdan seç:**")
        sel_theme = st.selectbox("Hisse", [r['Hisse'] for r in rows])
        if sel_theme:
            tkr_yf = sel_theme + ".IS"
            if tkr_yf in td:
                sc, mx, crit, det = score_emre(td[tkr_yf], bm_df)
                res = {"ticker": tkr_yf, "score": sc, "max_score": mx,
                       "criteria": crit, "details": det, "df": td[tkr_yf]}
                render_detail(res, "emre", interval)
    else:
        st.warning("Bu sektörler için yeterli hisse verisi alınamadı.")
    st.stop()

# ── ARAMA MODU ────────────────────────────────────────────────────────────────
if search_btn and search_query:
    ticker_yf = search_query + ".IS"
    st.markdown(f"## 🔍 {search_query} — İki Strateji Analizi")
    with st.spinner("Veri çekiliyor..."):
        single = fetch_data([ticker_yf], period=period, interval=interval)
        bm_df  = fetch_benchmark(period=period, interval=interval)

    if ticker_yf not in single:
        st.error(f"❌ {search_query} için veri çekilemedi.")
    else:
        df  = single[ticker_yf]
        sector = get_sector(ticker_yf)
        st.markdown(f"**Sektör:** `{sector}`")
        col1, col2 = st.columns(2)
        for col, sk, fn in [(col1,"emre",score_emre),(col2,"momentum",score_momentum)]:
            with col:
                st.markdown(f"### {STRATEGY_FN[sk][1]}")
                try:
                    sc, mx, crit, det = fn(df, bm_df)
                    color = "#22c55e" if sc==mx else ("#f59e0b" if sc>=mx-1 else "#ef4444")
                    st.markdown(f"**Skor:** <span style='color:{color};font-size:1.3rem;font-weight:700'>{sc}/{mx}</span>",
                                unsafe_allow_html=True)
                    for k, (passed, val) in crit.items():
                        icon = "✅" if passed else "❌"
                        cc   = "metric-pass" if passed else "metric-fail"
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">{k}</div>'
                            f'<div class="metric-value {cc}">{icon} {val}</div></div>',
                            unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Hata: {e}")
        st.markdown("---")
        fig = build_chart(df, ticker_yf, "emre", interval)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.stop()

# ── ANA TARAMA ────────────────────────────────────────────────────────────────
if st.session_state.strategy is None:
    st.info("👆 Bir strateji seç ve taramayı başlat.")
    st.stop()

strategy       = st.session_state.strategy
strategy_label = STRATEGY_FN[strategy][1]

st.markdown(f"**Aktif:** `{strategy_label}` · `{interval.upper()}` · `{len(BIST100_TICKERS)} hisse`")
st.markdown("---")

tara_btn = st.button("🔍 Tara — Tüm BIST 100", type="primary")

if tara_btn or not st.session_state.scan_done:
    with st.spinner(f"Taranıyor ({len(BIST100_TICKERS)} hisse)..."):
        bm_df      = fetch_benchmark(period=period, interval=interval)
        stock_data = fetch_data(BIST100_YF, period=period, interval=interval)
        st.session_state.bm_df = bm_df

    if strategy == "emre":
        final5, filtered_all = run_emre_scan(stock_data, bm_df)
        # final5'i result formatına çevir
        results = []
        for cand in filtered_all:
            df   = cand['df']
            sc, mx, crit, det = score_emre(df, bm_df)
            in_top5 = cand['ticker'] in {f['ticker'] for f in final5}
            results.append({
                "ticker": cand['ticker'], "score": sc, "max_score": mx,
                "criteria": crit, "details": det, "df": df,
                "rs": cand['rs'], "sector": cand['sector'],
                "in_top5": in_top5,
            })
        results.sort(key=lambda x: x['rs'], reverse=True)
    else:
        score_fn = STRATEGY_FN[strategy][0]
        results  = []
        for ticker, df in stock_data.items():
            if df is None or len(df) < 30: continue
            try:
                sc, mx, crit, det = score_fn(df, bm_df)
                results.append({"ticker": ticker, "score": sc, "max_score": mx,
                                 "criteria": crit, "details": det, "df": df,
                                 "sector": get_sector(ticker), "in_top5": False})
            except Exception:
                pass
        results.sort(key=lambda x: x['score'], reverse=True)
        # top5 işaretle
        for r in results[:5]: r['in_top5'] = True

    st.session_state.results   = results
    st.session_state.scan_done = True

results = st.session_state.get("results", [])
if not results:
    st.warning("Veri yok.")
    st.stop()

top5_r = [r for r in results if r.get('in_top5')]
near_r = [r for r in results if not r.get('in_top5')]

# ── AKTIF PORTFÖY PERFORMANS KUTUCUKLARI ─────────────────────────────────────
bm_df_cached = st.session_state.get("bm_df")
if top5_r and bm_df_cached is not None:
    # Sahte aktif pozisyon (bu ayki tarama sonucu)
    fake_active = [{
        'ticker':        r['ticker'].replace('.IS',''),
        'buy_date':      datetime.now().strftime('%d.%m.%Y'),
        'buy_price':     float(r['df']['Close'].squeeze().iloc[-1]),
        'current_price': float(r['df']['Close'].squeeze().iloc[-1]),
        'pnl_pct':       0.0,
    } for r in top5_r]
    render_perf_boxes(fake_active, bm_df_cached)

# ── LISTE + DETAY ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="large")

with left_col:
    st.markdown(f'<div class="section-title">🟡 Top 5 — Bu Ay ({len(top5_r)})</div>',
                unsafe_allow_html=True)
    for r in top5_r:
        lbl  = r["ticker"].replace(".IS","")
        sect = r.get("sector","")
        rs_s = f" · RS {r['rs']*100:.1f}%" if 'rs' in r and not np.isnan(r['rs']) else ""
        if st.button(f"⭐ {lbl}  {r['score']}/{r['max_score']}{rs_s}  [{sect}]",
                     key=f"t5_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

    st.markdown(f'<div class="section-title">⚠️ Diğer Geçenler ({len(near_r)})</div>',
                unsafe_allow_html=True)
    for r in near_r:
        lbl  = r["ticker"].replace(".IS","")
        sect = r.get("sector","")
        miss = r["max_score"] - r["score"]
        emoji = "🟡" if miss==1 else "🟠" if miss==2 else "🔴"
        if st.button(f"{emoji} {lbl}  {r['score']}/{r['max_score']}  [{sect}]",
                     key=f"nr_{r['ticker']}", use_container_width=True):
            st.session_state.selected_ticker = r["ticker"]

with right_col:
    sel = st.session_state.get("selected_ticker")
    if sel is None:
        st.markdown("### ← Soldan bir hisse seç")
        if top5_r:
            st.markdown(f"**Bu ay {strategy_label} top 5:**")
            for r in top5_r:
                st.markdown(f"⭐ **{r['ticker'].replace('.IS','')}** — "
                            f"`{r['score']}/{r['max_score']}` · {r.get('sector','')}")
    else:
        sel_result = next((r for r in results if r["ticker"] == sel), None)
        if sel_result:
            render_detail(sel_result, strategy, interval)
        else:
            st.warning("Hisse bulunamadı.")
