"""
Backtest Engine — BIST 100 Strateji Tarayıcı
Her ayın ilk borsa günü tam puan alan hisseleri eşit ağırlıkla al,
öbür ay uymuyorsa sat. 3 strateji için ayrı 100k₺ portföy.
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

def _vol_ratio_at(v, pos, n=20):
    if pos < n: return np.nan
    avg = v.iloc[pos-n:pos].mean()
    return v.iloc[pos] / avg if avg > 0 else np.nan

def _rs_at(c, bm, pos, bm_pos, days=63):
    if pos < days or bm_pos < days: return np.nan
    sr = (c.iloc[pos] / c.iloc[pos-days]) - 1
    br = (bm.iloc[bm_pos] / bm.iloc[bm_pos-days]) - 1
    return ((1+sr)/(1+br) - 1) if br != 0 else np.nan


# ── KRİTER KONTROL (geçmiş tarih için) ───────────────────────────────────────
def _check_at(strategy, df, bm_df, date):
    """
    Belirtilen tarihe kadar olan veriyle kriterleri kontrol et.
    Tam puan → True, değil → False
    """
    try:
        c = df['Close'].squeeze(); h = df['High'].squeeze()
        lo = df['Low'].squeeze(); v = df['Volume'].squeeze()
        bm = bm_df['Close'].squeeze()

        valid = c.index[c.index <= date]
        if len(valid) < 60: return False, []
        pos = len(valid) - 1  # son geçerli pozisyon
        c_s = c.iloc[:pos+1]; h_s = h.iloc[:pos+1]
        lo_s = lo.iloc[:pos+1]; v_s = v.iloc[:pos+1]

        price = c_s.iloc[-1]
        vr    = _vol_ratio_at(v_s, pos)

        if strategy == "rs":
            if len(c_s) < 63: return False, []
            s50     = _sma(c_s, 50).iloc[-1]
            rsi_v   = _rsi(c_s).iloc[-1]
            k_l, _  = _stoch(h_s, lo_s, c_s)
            k_now   = k_l.iloc[-1]; k_prev = k_l.iloc[-2]
            # RS
            bm_valid = bm.index[bm.index <= date]
            bm_pos   = len(bm_valid) - 1
            rs       = _rs_at(c, bm, pos, bm_pos)
            checks = [
                price > s50,
                50 <= rsi_v <= 70,
                k_now > k_prev and k_prev < 40,
                (not np.isnan(vr)) and vr >= 0.99,
                (rs is not None) and (not np.isnan(rs)) and rs > 0,
            ]
            return all(checks), checks

        elif strategy == "momentum":
            if len(c_s) < 60: return False, []
            s50   = _sma(c_s, 50).iloc[-1]
            _, _, hl = _macd(c_s)
            hn = hl.iloc[-1]; hp = hl.iloc[-2]
            a14 = _atr(h_s, lo_s, c_s)
            atr_now = a14.iloc[-1]
            atr_avg = a14.rolling(60).mean().iloc[-1]
            checks = [
                price > s50,
                hn > 0 and hn > hp,
                (not np.isnan(atr_avg)) and atr_now > atr_avg,
                (not np.isnan(vr)) and vr > 1.2,
            ]
            return all(checks), checks

        elif strategy == "trend":
            if len(c_s) < 200: return False, []
            e20  = _ema(c_s, 20).iloc[-1]
            e50  = _ema(c_s, 50).iloc[-1]
            e200 = _ema(c_s, 200).iloc[-1]
            rsi_v = _rsi(c_s).iloc[-1]
            checks = [
                e20 > e50 > e200,
                price > e20,
                50 <= rsi_v <= 65,
                (not np.isnan(vr)) and vr >= 1.2,
            ]
            return all(checks), checks

    except Exception:
        return False, []
    return False, []


# ── AY BAŞLARI ────────────────────────────────────────────────────────────────
def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())


# ── ANA BACKTEST ──────────────────────────────────────────────────────────────
def run_backtest(strategy, stock_data, benchmark_df, start_capital=100_000):
    """
    Returns:
        pv_df      — DataFrame(date, value) portföy değeri
        bm_norm    — Series benchmark normalize
        trades_df  — DataFrame işlem logu
        active_now — list of {ticker, buy_date, buy_price, current_price, pnl_pct}
    """
    bm = benchmark_df['Close'].squeeze()

    end_date   = pd.Timestamp(bm.index[-1])
    start_date = end_date - pd.DateOffset(years=2)

    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    if len(bm_idx) < 10:
        return None, None, None, []

    rebal_dates = _month_starts(bm_idx, start_date, end_date)
    if len(rebal_dates) < 2:
        return None, None, None, []

    cash     = float(start_capital)
    holdings = {}   # ticker → {shares, buy_price, buy_date}
    pv_log   = []
    trades   = []

    for i, rdate in enumerate(rebal_dates):
        # Anlık portföy değeri
        port_val = cash
        for tkr, pos in holdings.items():
            if tkr in stock_data:
                c = stock_data[tkr]['Close'].squeeze()
                v = c.index[c.index <= rdate]
                if len(v): port_val += pos['shares'] * float(c.loc[v[-1]])
        pv_log.append({'date': rdate, 'value': port_val})

        # Tarama: tam puan alanlar
        qualified = []
        for tkr, df in stock_data.items():
            if df is None or len(df) < 60: continue
            passed, _ = _check_at(strategy, df, benchmark_df, rdate)
            if passed:
                c = df['Close'].squeeze()
                v = c.index[c.index <= rdate]
                if len(v):
                    qualified.append({'ticker': tkr, 'price': float(c.loc[v[-1]])})

        target = {q['ticker'] for q in qualified}

        # Sat: portföyde olup kritere uymayanlar
        for tkr in list(holdings.keys()):
            if tkr not in target:
                pos = holdings[tkr]
                c = stock_data[tkr]['Close'].squeeze()
                v = c.index[c.index <= rdate]
                sell_p = float(c.loc[v[-1]]) if len(v) else pos['buy_price']
                proceeds = pos['shares'] * sell_p
                pnl = (sell_p / pos['buy_price'] - 1) * 100
                cash += proceeds
                trades.append({
                    'Tarih':  rdate.strftime('%d.%m.%Y'),
                    'Hisse':  tkr.replace('.IS',''),
                    'İşlem':  'SATIŞ',
                    'Fiyat':  f"{sell_p:.2f} ₺",
                    'P&L':    f"{pnl:+.1f}%",
                    'Süre':   f"{(rdate - pos['buy_date']).days} gün",
                })
                del holdings[tkr]

        # Al: hedefte olup portföyde olmayanlar
        new_buys = [q for q in qualified if q['ticker'] not in holdings]
        n_total  = len(target)
        if new_buys and n_total > 0:
            alloc = port_val / n_total
            for q in new_buys:
                amt = min(alloc, cash)
                if amt < 50: continue
                shares = amt / q['price']
                holdings[q['ticker']] = {
                    'shares':    shares,
                    'buy_price': q['price'],
                    'buy_date':  rdate,
                }
                cash -= amt
                trades.append({
                    'Tarih':  rdate.strftime('%d.%m.%Y'),
                    'Hisse':  q['ticker'].replace('.IS',''),
                    'İşlem':  'ALIŞ',
                    'Fiyat':  f"{q['price']:.2f} ₺",
                    'P&L':    '-',
                    'Süre':   '-',
                })

    # Son değer
    final_val = cash
    last_date = bm_idx[-1]
    active_now = []
    for tkr, pos in holdings.items():
        if tkr in stock_data:
            c = stock_data[tkr]['Close'].squeeze()
            v = c.index[c.index <= last_date]
            cur_p = float(c.loc[v[-1]]) if len(v) else pos['buy_price']
            final_val += pos['shares'] * cur_p
            pnl = (cur_p / pos['buy_price'] - 1) * 100
            active_now.append({
                'ticker':      tkr.replace('.IS',''),
                'buy_date':    pos['buy_date'].strftime('%d.%m.%Y'),
                'buy_price':   pos['buy_price'],
                'current_price': cur_p,
                'pnl_pct':     pnl,
                'shares':      pos['shares'],
            })
    pv_log.append({'date': last_date, 'value': final_val})

    pv_df = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Benchmark normalize
    bm_s = bm.loc[bm_idx[0]:last_date]
    bm_norm = (bm_s / float(bm_s.iloc[0])) * start_capital

    return pv_df, bm_norm, trades_df, active_now


# ── İSTATİSTİKLER ─────────────────────────────────────────────────────────────
def calc_stats(pv_df, bm_norm, start_capital):
    pv      = pv_df['value']
    total   = (pv.iloc[-1] / start_capital - 1) * 100
    n_yrs   = max((pv_df.index[-1] - pv_df.index[0]).days / 365, 0.01)
    cagr    = ((pv.iloc[-1] / start_capital) ** (1/n_yrs) - 1) * 100
    dd      = (pv / pv.cummax() - 1) * 100
    max_dd  = dd.min()
    bm_ret  = (bm_norm.iloc[-1] / start_capital - 1) * 100 if bm_norm is not None else None
    alpha   = total - bm_ret if bm_ret is not None else None
    return {
        'Toplam Getiri':   f"{total:+.1f}%",
        'CAGR':            f"{cagr:+.1f}%",
        'Max Drawdown':    f"{max_dd:.1f}%",
        'BIST100 Getirisi':f"{bm_ret:+.1f}%" if bm_ret is not None else 'N/A',
        'Alpha':           f"{alpha:+.1f}%" if alpha is not None else 'N/A',
        'Son Değer':       f"₺{pv.iloc[-1]:,.0f}",
    }


# ── GRAFİK ───────────────────────────────────────────────────────────────────
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

def build_perf_chart(results_map, start_capital):
    """
    results_map: { strategy_key: (pv_df, bm_norm) }
    Tüm stratejileri tek grafikte, bm bir kez çiz.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.04)

    bm_drawn = False
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        color = STRAT_COLORS.get(strat, '#94a3b8')
        label = STRAT_LABELS.get(strat, strat)
        pv    = pv_df['value']

        fig.add_trace(go.Scatter(
            x=pv_df.index, y=pv,
            name=label,
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>{label}</b><br>₺%{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)

        # Drawdown
        dd = (pv / pv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=pv_df.index, y=dd,
            name=f"DD {label}",
            line=dict(color=color, width=1.2),
            fill='tozeroy',
            fillcolor=color.replace('#', 'rgba(').replace(')', ',0.08)') if color.startswith('#') else color,
            showlegend=False,
            hovertemplate=f"DD: %{{y:.1f}}%<extra></extra>",
        ), row=2, col=1)

        if not bm_drawn and bm_norm is not None:
            fig.add_trace(go.Scatter(
                x=bm_norm.index, y=bm_norm,
                name='BIST 100',
                line=dict(color='#f59e0b', width=1.5, dash='dot'),
                hovertemplate="BIST100: ₺%{y:,.0f}<extra></extra>",
            ), row=1, col=1)
            bm_drawn = True

    # Başlangıç çizgisi
    fig.add_hline(y=start_capital, line=dict(color='#475569', width=0.8, dash='dash'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#475569', width=0.6), row=2, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14',
        font=dict(family='JetBrains Mono', color='#94a3b8', size=11),
        legend=dict(orientation='h', y=1.03, x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode='x unified',
        yaxis=dict(tickprefix='₺', tickformat=',.0f', gridcolor='#1e2535'),
        yaxis2=dict(ticksuffix='%', gridcolor='#1e2535'),
        xaxis2=dict(gridcolor='#1e2535'),
        xaxis=dict(gridcolor='#1e2535'),
    )
    return fig
