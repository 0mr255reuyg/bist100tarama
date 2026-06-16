"""
Sektörel Özet Modülü
Fotoğraftaki gibi: Bugün Lider, İvme Kazanan, 1H Zirve, 1A Zirve,
Toparlayan, Bugün Geride, 1H Dip, 1A Dip, Yavaşlayan
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sectors import SECTOR_MAP

def _sector_returns(stock_data):
    """
    Her hisse için 1g, 5g, 21g getiri hesapla.
    Sektör bazında ağırlıksız ortalama al.
    Returns: DataFrame(sektör, ret_1d, ret_5d, ret_21d, mom_5d, mom_21d)
    """
    rows = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 25: continue
        try:
            tkr  = ticker.replace(".IS","")
            sect = SECTOR_MAP.get(tkr, "Diğer")
            c    = df['Close'].squeeze()

            # Sıfıra bölünme hatasını engellemek için güvenlik kontrolleri
            if float(c.iloc[-2]) == 0 or float(c.iloc[-6]) == 0 or float(c.iloc[-22]) == 0:
                continue

            ret_1d  = float(c.iloc[-1] / c.iloc[-2]  - 1) * 100 if len(c) >= 2  else np.nan
            ret_5d  = float(c.iloc[-1] / c.iloc[-6]  - 1) * 100 if len(c) >= 6  else np.nan
            ret_21d = float(c.iloc[-1] / c.iloc[-22] - 1) * 100 if len(c) >= 22 else np.nan

            # İvme: son 5g getiri - önceki 5g getiri (momentum kırılımı)
            mom_5d_prev = (float(c.iloc[-6] / c.iloc[-11] - 1)*100) if len(c) >= 11 and float(c.iloc[-11]) != 0 else 0
            mom_5d  = ret_5d - mom_5d_prev

            # Yavaşlama: son 21g getiri - önceki 21g getiri
            mom_21d_prev = (float(c.iloc[-22] / c.iloc[-43] - 1)*100) if len(c) >= 43 and float(c.iloc[-43]) != 0 else 0
            mom_21d = ret_21d - mom_21d_prev

            rows.append({
                'ticker': tkr, 'sector': sect,
                'ret_1d': ret_1d, 'ret_5d': ret_5d, 'ret_21d': ret_21d,
                'mom_5d': mom_5d, 'mom_21d': mom_21d,
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
    
    # Sektörlere göre grupla ve ortalamaları al
    sect_df = df_all.groupby('sector').agg(
        ret_1d=('ret_1d','mean'),
        ret_5d=('ret_5d','mean'),
        ret_21d=('ret_21d','mean'),
        mom_5d=('mom_5d','mean'),
        mom_21d=('mom_21d','mean'),
        hisse_sayisi=('ticker','count'),
    ).reset_index()

    return sect_df


def build_summary(stock_data):
    """
    Returns dict of category → list of (sektor, deger_str)
    """
    df = _sector_returns(stock_data)
    if df.empty:
        return {}

    # NaN değerlerin sıralamayı bozmasını engellemek için dropna kullanıyoruz
    df_clean = df.dropna(subset=['ret_1d', 'ret_5d', 'ret_21d', 'mom_5d'])

    # Sırala ve En İyi / En Kötü 3'leri Çek
    top_1d   = df_clean.nlargest(3, 'ret_1d')[['sector','ret_1d']].values.tolist()
    bot_1d   = df_clean.nsmallest(3, 'ret_1d')[['sector','ret_1d']].values.tolist()
    
    top_5d   = df_clean.nlargest(3, 'ret_5d')[['sector','ret_5d']].values.tolist()
    bot_5d   = df_clean.nsmallest(3, 'ret_5d')[['sector','ret_5d']].values.tolist()
    
    top_21d  = df_clean.nlargest(3, 'ret_21d')[['sector','ret_21d']].values.tolist()
    bot_21d  = df_clean.nsmallest(3, 'ret_21d')[['sector','ret_21d']].values.tolist()
    
    ivme     = df_clean.nlargest(3, 'mom_5d')[['sector','mom_5d']].values.tolist()
    yavas    = df_clean.nsmallest(3, 'mom_5d')[['sector','mom_5d']].values.tolist()
    
    # Toparlayan: 1 Haftalık pozitif, 1 Aylık negatif olanlar arasından en iyi sıçramayı yapanlar
    toparlayan = df_clean[(df_clean['ret_5d'] > 0) & (df_clean['ret_21d'] < 0)].nlargest(3,'ret_5d')[['sector','ret_5d']].values.tolist()

    def fmt(val): return f"{val:+.2f}%"

    return {
        "🚀 BUGÜN LİDER":      [(s, fmt(v)) for s,v in top_1d],
        "⚡ İVME KAZANAN":     [(s, fmt(v)) for s,v in ivme],
        "📈 1 HAFTA ZİRVE":    [(s, fmt(v)) for s,v in top_5d],
        "🏆 1 AY ZİRVE":       [(s, fmt(v)) for s,v in top_21d],
        "🔄 TOPARLAYAN":       [(s, fmt(v)) for s,v in toparlayan] or [("—","")],
        "📉 BUGÜN GERİDE":     [(s, fmt(v)) for s,v in bot_1d],
        "❄️ 1 HAFTA DİP":     [(s, fmt(v)) for s,v in bot_5d],
        "🪨 1 AY DİP":         [(s, fmt(v)) for s,v in bot_21d],
        "🐌 YAVAŞLAYAN":       [(s, fmt(v)) for s,v in yavas],
    }


def build_sector_bar_chart(stock_data):
    """Sektörlerin 1g/5g/21g getirilerini bar chart olarak göster"""
    df = _sector_returns(stock_data)
    if df.empty: return None

    # Grafik görünümünü güzelleştirmek için bugünkü getiriye göre küçükten büyüğe diziyoruz
    df = df.dropna(subset=['ret_1d']).sort_values('ret_1d', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['sector'], x=df['ret_1d'],
        name='Bugün', orientation='h',
        marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in df['ret_1d']],
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        y=df['sector'], x=df['ret_5d'],
        name='1 Hafta', orientation='h',
        marker_color='#38bdf8', opacity=0.5, visible='legendonly',
    ))
    fig.add_trace(go.Bar(
        y=df['sector'], x=df['ret_21d'],
        name='1 Ay', orientation='h',
        marker_color='#a78bfa', opacity=0.5, visible='legendonly',
    ))

    fig.update_layout(
        height=max(450, len(df) * 30), # Uzun sektör isimlerinin sıkışmaması için yüksekliği artırdık
        paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14',
        font=dict(family='JetBrains Mono', color='#94a3b8', size=11),
        legend=dict(orientation='h', y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=20, b=10),
        barmode='overlay',
        xaxis=dict(ticksuffix='%', gridcolor='#1e2535', zeroline=True,
                   zerolinecolor='#475569', zerolinewidth=1),
        yaxis=dict(gridcolor='#1e2535'),
        title=dict(text='Sektör Getirileri', font=dict(color='#f1f5f9', size=13), x=0.01),
    )
    return fig
