import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sectors import SECTOR_MAP

def _sector_returns(stock_data):
    valid_dfs = [df for df in stock_data.values() if df is not None and not df.empty]
    if not valid_dfs: return pd.DataFrame()

    rows = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 25: continue
        try:
            df_sorted = df.sort_index()
            tkr = ticker.replace(".IS","")
            sect = SECTOR_MAP.get(tkr, "Diğer")

            # Kaymasız net yüzde getiri hesaplamaları revize edildi
            ret_1d = float(df_sorted['Close'].pct_change(1).iloc[-1] * 100)
            ret_5d = float(df_sorted['Close'].pct_change(5).iloc[-1] * 100) if len(df_sorted) >= 6 else np.nan
            ret_21d = float(df_sorted['Close'].pct_change(21).iloc[-1] * 100) if len(df_sorted) >= 22 else np.nan
            mom_5d = float(df_sorted['Close'].pct_change(5).iloc[-1] - df_sorted['Close'].pct_change(5).iloc[-6]) if len(df_sorted) >= 11 else 0

            rows.append({
                'ticker': tkr, 'sector': sect,
                'ret_1d': ret_1d, 'ret_5d': ret_5d, 'ret_21d': ret_21d, 'mom_5d': mom_5d
            })
        except Exception: pass

    if not rows: return pd.DataFrame()
    df_all = pd.DataFrame(rows)
    return df_all.groupby('sector').agg(
        ret_1d=('ret_1d','mean'), ret_5d=('ret_5d','mean'), ret_21d=('ret_21d','mean'),
        mom_5d=('mom_5d','mean'), hisse_sayisi=('ticker','count')
    ).reset_index()

def build_summary(stock_data):
    df = _sector_returns(stock_data)
    if df.empty: return {}
    df_clean = df.dropna(subset=['ret_1d', 'ret_5d', 'ret_21d'])
    def fmt(val): return f"{val:+.2f}%"

    return {
        "🚀 SON KAPANIŞ LİDERİ": [(s, fmt(v)) for s,v in df_clean.nlargest(3, 'ret_1d')[['sector','ret_1d']].values.tolist()],
        "⚡ İVME KAZANAN": [(s, fmt(v)) for s,v in df_clean.nlargest(3, 'mom_5d')[['sector','mom_5d']].values.tolist()],
        "📈 1 HAFTA ZİRVE": [(s, fmt(v)) for s,v in df_clean.nlargest(3, 'ret_5d')[['sector','ret_5d']].values.tolist()],
        "🏆 1 AY ZİRVE": [(s, fmt(v)) for s,v in df_clean.nlargest(3, 'ret_21d')[['sector','ret_21d']].values.tolist()],
        "🔄 TOPARLAYAN": [(s, fmt(v)) for s,v in df_clean[(df_clean['ret_5d'] > 0) & (df_clean['ret_21d'] < 0)].nlargest(3,'ret_5d')[['sector','ret_5d']].values.tolist()] or [("—","")],
        "📉 SON KAPANIŞTA GERİDE": [(s, fmt(v)) for s,v in df_clean.nsmallest(3, 'ret_1d')[['sector','ret_1d']].values.tolist()],
        "❄️ 1 HAFTA DİP": [(s, fmt(v)) for s,v in df_clean.nsmallest(3, 'ret_5d')[['sector','ret_5d']].values.tolist()],
        "🪨 1 AY DİP": [(s, fmt(v)) for s,v in df_clean.nsmallest(3, 'ret_21d')[['sector','ret_21d']].values.tolist()],
        "🐌 YAVAŞLAYAN": [(s, fmt(v)) for s,v in df_clean.nsmallest(3, 'mom_5d')[['sector','mom_5d']].values.tolist()],
    }

def build_sector_bar_chart(stock_data):
    df = _sector_returns(stock_data)
    if df.empty: return None
    df = df.dropna(subset=['ret_1d']).sort_values('ret_1d', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df['sector'], x=df['ret_1d'], orientation='h', marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in df['ret_1d']]))
    fig.update_layout(height=500, paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14', font=dict(family='JetBrains Mono', color='#94a3b8'))
    return fig
