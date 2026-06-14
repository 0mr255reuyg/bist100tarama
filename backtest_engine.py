"""
Backtest Engine — BIST 100 Strateji Tarayıcı
Her ayın ilk borsa günü en yüksek puanlı TOP 5 hisseyi al.
Öbür ay top 5'te yoksa sat. 2 yıl · günlük · 100k₺.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── İNDİKATÖRLER ─────────────────────────────────────────────────────────────
def _sma(s, n):  return s.rolling(n).mean()
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _stoch(h, l, c, k=14, smooth=3, d=3):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    raw = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    ks = raw.rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def _macd(s, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow); sg = _ema(m, sig)
    return m, sg, m - sg

def _atr(h, l, c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _vol_ratio_at(v, n=20):
    if len(v) < n + 1: return np.nan
    avg = v.iloc[-n-1:-1].mean()
    return float(v.iloc[-1]) / avg if avg > 0 else np.nan

def _rs_at(c_s, bm, date, days=63):
    bm_valid = bm.index[bm.index <= date]
    if len(c_s) < days or len(bm_valid) < days: return np.nan
    bm_s = bm.loc[bm_valid]
    sr = (float(c_s.iloc[-1]) / float(c_s.iloc[-days])) - 1
    br = (float(bm_s.iloc[-1]) / float(bm_s.iloc[-days])) - 1
    return ((1+sr)/(1+br) - 1) if br != 0 else np.nan


# ── KRİTER SKORU (geçmiş tarih için, int döner) ──────────────────────────────
def _score_at(strategy, df, bm_df, date):
    """
    Belirtilen tarihe kadar olan veriyle kaç kriter geçildi döner.
    (score, max_score, max_criteria_count)
    """
    try:
        c  = df['Close'].squeeze()
        h  = df['High'].squeeze()
        lo = df['Low'].squeeze()
        v  = df['Volume'].squeeze()
        bm = bm_df['Close'].squeeze()

        valid = c.index[c.index <= date]
        if len(valid) < 55: return 0, 1
        c_s  = c.loc[valid]; h_s = h.loc[valid]
        lo_s = lo.loc[valid]; v_s = v.loc[valid]
        price = float(c_s.iloc[-1])
        vr    = _vol_ratio_at(v_s)

        if strategy == "rs":
            if len(c_s) < 63: return 0, 5
            s50   = float(_sma(c_s, 50).iloc[-1])
            rsi_v = float(_rsi(c_s).iloc[-1])
            k_l, _ = _stoch(h_s, lo_s, c_s)
            k_now  = float(k_l.iloc[-1]); k_prev = float(k_l.iloc[-2])
            rs     = _rs_at(c_s, bm, date)
            checks = [
                price > s50,
                50 <= rsi_v <= 70,
                k_now > k_prev and k_prev < 40,
                (not np.isnan(vr)) and vr >= 0.99,
                (rs is not None) and (not np.isnan(rs)) and rs > 0,
            ]
            return sum(checks), 5

        elif strategy == "momentum":
            if len(c_s) < 60: return 0, 4
            s50 = float(_sma(c_s, 50).iloc[-1])
            _, _, hl = _macd(c_s)
            hn = float(hl.iloc[-1]); hp = float(hl.iloc[-2])
            a14 = _atr(h_s, lo_s, c_s)
            atr_now = float(a14.iloc[-1])
            atr_avg = float(a14.rolling(60).mean().iloc[-1])
            checks = [
                price > s50,
                hn > 0 and hn > hp,
                (not np.isnan(atr_avg)) and atr_now > atr_avg,
                (not np.isnan(vr)) and vr > 1.2,
            ]
            return sum(checks), 4

        elif strategy == "trend":
            if len(c_s) < 200: return 0, 4
            e20  = float(_ema(c_s, 20).iloc[-1])
            e50  = float(_ema(c_s, 50).iloc[-1])
            e200 = float(_ema(c_s, 200).iloc[-1])
            rsi_v = float(_rsi(c_s).iloc[-1])
            checks = [
                e20 > e50 > e200,
                price > e20,
                50 <= rsi_v <= 65,
                (not np.isnan(vr)) and vr >= 1.2,
            ]
            return sum(checks), 4

    except Exception:
        pass
    return 0, 1


# ── AY BAŞLARI ────────────────────────────────────────────────────────────────
def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())


# ── ANA BACKTEST ──────────────────────────────────────────────────────────────
def run_backtest(strategy, stock_data, benchmark_df,
                 start_capital=100_000, top_n=5):
    """
    Her ay başında en yüksek puanlı top_n hisseyi al.
    Döner: pv_df, bm_norm, trades_df, active_now, monthly_table
    """
    bm = benchmark_df['Close'].squeeze()
    end_date   = pd.Timestamp(bm.index[-1])
    start_date = end_date - pd.DateOffset(years=2)

    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    if len(bm_idx) < 10:
        return None, None, None, [], pd.DataFrame()

    rebal_dates = _month_starts(bm_idx, start_date, end_date)
    if len(rebal_dates) < 2:
        return None, None, None, [], pd.DataFrame()

    cash     = float(start_capital)
    holdings = {}   # ticker → {shares, buy_price, buy_date, score}
    pv_log   = []
    trades   = []
    monthly_rows = []   # aylık tablo için

    for i, rdate in enumerate(rebal_dates):
        # Portföy değeri (rebalans öncesi)
        port_val = cash
        for tkr, pos in holdings.items():
            if tkr in stock_data:
                c = stock_data[tkr]['Close'].squeeze()
                vd = c.index[c.index <= rdate]
                if len(vd):
                    port_val += pos['shares'] * float(c.loc[vd[-1]])
        pv_log.append({'date': rdate, 'value': port_val})

        # Tarama: tüm hisseleri skorla
        candidates = []
        for tkr, df in stock_data.items():
            if df is None or len(df) < 55: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate)
            if sc == 0: continue
            c = df['Close'].squeeze()
            vd = c.index[c.index <= rdate]
            if len(vd):
                price = float(c.loc[vd[-1]])
                candidates.append({
                    'ticker': tkr, 'score': sc,
                    'max': mx, 'price': price
                })

        # En yüksek puanlı top_n
        candidates.sort(key=lambda x: (x['score'], x['ticker']), reverse=True)
        top5 = candidates[:top_n]
        target_tickers = {q['ticker'] for q in top5}

        # Aylık tablo satırı — bu ayın top 5'i
        row = {'Ay': rdate.strftime('%b %Y'), 'Port. Değer': port_val}
        for rank, q in enumerate(top5, 1):
            row[f'#{rank}'] = f"{q['ticker'].replace('.IS','')} ({q['score']}/{q['max']})"
        monthly_rows.append(row)

        # Sat: top5'te olmayan holdings
        for tkr in list(holdings.keys()):
            if tkr not in target_tickers:
                pos = holdings[tkr]
                c = stock_data[tkr]['Close'].squeeze()
                vd = c.index[c.index <= rdate]
                sell_p = float(c.loc[vd[-1]]) if len(vd) else pos['buy_price']
                pnl = (sell_p / pos['buy_price'] - 1) * 100
                cash += pos['shares'] * sell_p
                trades.append({
                    'Ay':    rdate.strftime('%b %Y'),
                    'Hisse': tkr.replace('.IS',''),
                    'İşlem': 'SATIŞ',
                    'Alış ₺': f"{pos['buy_price']:.2f}",
                    'Satış ₺': f"{sell_p:.2f}",
                    'P&L':   f"{pnl:+.1f}%",
                    'Süre':  f"{(rdate - pos['buy_date']).days} gün",
                    'Skor':  f"{pos['score']}/{pos['max']}",
                })
                del holdings[tkr]

        # Al: top5'te olup holdings'de olmayanlar
        new_buys = [q for q in top5 if q['ticker'] not in holdings]
        n_total  = len(target_tickers)
        if new_buys and n_total > 0:
            alloc = port_val / n_total
            for q in new_buys:
                amt = min(alloc, cash)
                if amt < 50 or q['price'] <= 0: continue
                shares = amt / q['price']
                holdings[q['ticker']] = {
                    'shares':    shares,
                    'buy_price': q['price'],
                    'buy_date':  rdate,
                    'score':     q['score'],
                    'max':       q['max'],
                }
                cash -= amt
                trades.append({
                    'Ay':     rdate.strftime('%b %Y'),
                    'Hisse':  q['ticker'].replace('.IS',''),
                    'İşlem':  'ALIŞ',
                    'Alış ₺': f"{q['price']:.2f}",
                    'Satış ₺': '-',
                    'P&L':    '-',
                    'Süre':   '-',
                    'Skor':   f"{q['score']}/{q['max']}",
                })

    # Son değer + aktif pozisyonlar
    last_date = bm_idx[-1]
    final_val = cash
    active_now = []
    for tkr, pos in holdings.items():
        if tkr in stock_data:
            c = stock_data[tkr]['Close'].squeeze()
            vd = c.index[c.index <= last_date]
            cur_p = float(c.loc[vd[-1]]) if len(vd) else pos['buy_price']
            final_val += pos['shares'] * cur_p
            pnl = (cur_p / pos['buy_price'] - 1) * 100
            active_now.append({
                'ticker':        tkr.replace('.IS',''),
                'buy_date':      pos['buy_date'].strftime('%d.%m.%Y'),
                'buy_price':     pos['buy_price'],
                'current_price': cur_p,
                'pnl_pct':       pnl,
                'score':         pos['score'],
                'max':           pos['max'],
            })
    pv_log.append({'date': last_date, 'value': final_val})

    pv_df     = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Aylık tabloya P&L sütunu ekle (ay içi portföy değişimi)
    monthly_df = pd.DataFrame(monthly_rows)
    if len(monthly_df) > 1:
        monthly_df['Aylık P&L'] = monthly_df['Port. Değer'].pct_change() * 100
        monthly_df['Aylık P&L'] = monthly_df['Aylık P&L'].apply(
            lambda x: f"{x:+.1f}%" if not pd.isna(x) else '-'
        )
        monthly_df['Port. Değer'] = monthly_df['Port. Değer'].apply(
            lambda x: f"₺{x:,.0f}"
        )

    # Benchmark normalize
    bm_s    = bm.loc[bm_idx[0]:last_date]
    bm_norm = (bm_s / float(bm_s.iloc[0])) * start_capital

    return pv_df, bm_norm, trades_df, active_now, monthly_df


# ── İSTATİSTİKLER ─────────────────────────────────────────────────────────────
def calc_stats(pv_df, bm_norm, start_capital):
    pv     = pv_df['value']
    total  = (pv.iloc[-1] / start_capital - 1) * 100
    n_yrs  = max((pv_df.index[-1] - pv_df.index[0]).days / 365, 0.01)
    cagr   = ((pv.iloc[-1] / start_capital) ** (1/n_yrs) - 1) * 100
    dd     = (pv / pv.cummax() - 1) * 100
    max_dd = dd.min()
    bm_ret = (bm_norm.iloc[-1] / start_capital - 1) * 100 if bm_norm is not None else None
    alpha  = total - bm_ret if bm_ret is not None else None
    return {
        'Toplam Getiri':    f"{total:+.1f}%",
        'CAGR':             f"{cagr:+.1f}%",
        'Max Drawdown':     f"{max_dd:.1f}%",
        'BIST100 Getirisi': f"{bm_ret:+.1f}%" if bm_ret is not None else 'N/A',
        'Alpha':            f"{alpha:+.1f}%" if alpha is not None else 'N/A',
        'Son Değer':        f"₺{pv.iloc[-1]:,.0f}",
    }


# ── RENK & LABEL SABİTLERİ ────────────────────────────────────────────────────
STRAT_COLORS = {
    'rs':       '#38bdf8',
    'momentum': '#a78bfa',
    'trend':    '#4ade80',
}
STRAT_LABELS = {
    'rs':       '🔵 Rölatif Güç',
    'momentum': '🟣 Momentum',
    'trend':    '🟢 Trend Sürücüsü',
}


# ── PERFORMANS GRAFİĞİ ────────────────────────────────────────────────────────
def build_perf_chart(results_map, start_capital):
    """
    results_map: { strategy_key: (pv_df, bm_norm) }
    Üst: kümülatif % · Alt: drawdown %
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.04)

    bm_drawn = False
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        color  = STRAT_COLORS.get(strat, '#94a3b8')
        label  = STRAT_LABELS.get(strat, strat)
        pv     = pv_df['value']
        cumret = (pv / start_capital - 1) * 100

        fig.add_trace(go.Scatter(
            x=pv_df.index, y=cumret,
            name=label,
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>{label}</b><br>%{{y:+.1f}}%<extra></extra>",
        ), row=1, col=1)

        dd = (pv / pv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=pv_df.index, y=dd,
            name=f"DD {label}",
            line=dict(color=color, width=1.2),
            fill='tozeroy',
            opacity=0.4,
            showlegend=False,
            hovertemplate="DD: %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        if not bm_drawn and bm_norm is not None:
            bm_ret = (bm_norm / start_capital - 1) * 100
            fig.add_trace(go.Scatter(
                x=bm_norm.index, y=bm_ret,
                name='BIST 100',
                line=dict(color='#f59e0b', width=1.5, dash='dot'),
                hovertemplate="BIST100: %{y:+.1f}%<extra></extra>",
            ), row=1, col=1)
            bm_drawn = True

    fig.add_hline(y=0, line=dict(color='#475569', width=0.8, dash='dash'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#475569', width=0.6), row=2, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14',
        font=dict(family='JetBrains Mono', color='#94a3b8', size=11),
        legend=dict(orientation='h', y=1.03, x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode='x unified',
        yaxis=dict(ticksuffix='%', gridcolor='#1e2535',
                   title=dict(text='Kümülatif Getiri %', font=dict(size=10))),
        yaxis2=dict(ticksuffix='%', gridcolor='#1e2535',
                    title=dict(text='Drawdown %', font=dict(size=10))),
        xaxis=dict(gridcolor='#1e2535'),
        xaxis2=dict(gridcolor='#1e2535'),
    )
    return fig
