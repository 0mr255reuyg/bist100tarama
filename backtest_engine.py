import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sectors import get_sector

def _sma(s, n):  return s.rolling(n).mean()
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _macd(s, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow); sg = _ema(m, sig)
    return m, sg, m - sg

def _vol_ratio_at(v, n=20):
    if len(v) < n + 1: return 1.0
    return float(v.iloc[-1]) / max(float(v.iloc[-n-1:-1].mean()), 1.0)

def _rs_at(c_s, bm, date, days=20):
    bm_valid = bm.index[bm.index <= date]
    if len(c_s) < days or len(bm_valid) < days: return 0.0
    return ((float(c_s.iloc[-1]) / float(c_s.iloc[-days])) - 1) - ((float(bm.loc[bm_valid].iloc[-1]) / float(bm.loc[bm_valid].iloc[-days])) - 1)

def _get_historical_macro_sectors(date):
    np.random.seed(int(date.strftime('%Y%m%d')) % 100)
    regimes = [
        ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"],
        ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"],
        ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"]
    ]
    return regimes[np.random.choice([0, 1, 2])]

def _score_at(strategy, df, bm_df, date, ticker):
    try:
        c = df['Close'].squeeze(); v = df['Volume'].squeeze(); bm = bm_df['Close'].squeeze()
        valid = c.index[c.index <= date]
        if len(valid) < 30: return 0, 5
        c_s = c.loc[valid]; v_s = v.loc[valid]; price = float(c_s.iloc[-1])
        s20 = float(_sma(c_s, 20).iloc[-1]); s50 = float(_sma(c_s, 50).iloc[-1])
        rsi_v = float(_rsi(c_s).iloc[-1]); rs = _rs_at(c_s, bm, date, days=20); vr = _vol_ratio_at(v_s, 20)

        if strategy == "emre":
            checks = [rs > 0, price > s20 and price > s50, vr >= 0.9, rsi_v < 80]
            score = sum(checks); max_score = 4
            if get_sector(ticker) in _get_historical_macro_sectors(date): score += 1
            return score, max_score + 1
        elif strategy == "momentum":
            return sum([price > s50, rsi_v > 50]), 4
    except Exception: pass
    return 0, 4

def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())

def run_backtest(strategy, stock_data, benchmark_df, start_capital=100_000, top_n=5):
    bm = benchmark_df['Close'].squeeze()
    start_date = pd.Timestamp('2024-06-01'); end_date = pd.Timestamp(bm.index[-1])
    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    if len(bm_idx) < 5: return None, None, None, [], pd.DataFrame()
    rebal_dates = _month_starts(bm_idx, start_date, end_date)

    cash = float(start_capital); holdings = {}; pv_log = []; trades = []; monthly_rows = []   

    for i, rdate in enumerate(rebal_dates):
        port_val = cash; current_prices = {}
        for tkr, pos in holdings.items():
            if tkr in stock_data:
                c = stock_data[tkr]['Close'].squeeze(); vd = c.index[c.index <= rdate]
                p = float(c.loc[vd[-1]]) if len(vd) else pos['buy_price']
                current_prices[tkr] = p; port_val += pos['shares'] * p
        pv_log.append({'date': rdate, 'value': port_val})

        candidates = []
        for tkr, df in stock_data.items():
            if df is None or len(df) < 20: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate, tkr)
            c = df['Close'].squeeze(); vd = c.index[c.index <= rdate]
            if len(vd): candidates.append({'ticker': tkr, 'score': sc, 'max': mx, 'price': float(c.loc[vd[-1]]), 'rs': _rs_at(c.loc[vd], bm, rdate, days=20)})

        candidates.sort(key=lambda x: (x['score'], x['rs']), reverse=True)
        top5 = []; sector_counts = {}
        for cand in candidates:
            sect = get_sector(cand['ticker'])
            if strategy == "emre":
                if sector_counts.get(sect, 0) < 2: top5.append(cand); sector_counts[sect] = sector_counts.get(sect, 0) + 1
            else: top5.append(cand)
            if len(top5) == top_n: break

        target_tickers = {q['ticker']: q for q in top5}
        alloc_per_stock = port_val / len(top5) if top5 else 0
        row = {'Ay': rdate.strftime('%b %Y'), 'Portföy Büyüklüğü': port_val}
        for rank, q in enumerate(top5, 1): row[f'#{rank}'] = f"{q['ticker'].replace('.IS','')} ({q['score']}/{q['max']})"
        monthly_rows.append(row)

        for tkr in list(holdings.keys()):
            if tkr not in target_tickers:
                pos = holdings[tkr]; sell_p = current_prices.get(tkr, pos['buy_price'])
                cash += pos['shares'] * sell_p
                trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🔴 SATIŞ (Çıkış)', 'İşlem Hacmi ₺': f"{(pos['shares']*sell_p):.2f}", 'Fiyat ₺': f"{sell_p:.2f}", 'Anlık Net K/Z': f"{((sell_p/pos['buy_price']-1)*100):+.1f}%"})
                del holdings[tkr]

        for tkr, q in target_tickers.items():
            if tkr in holdings:
                pos = holdings[tkr]; curr_p = current_prices[tkr]; curr_val = pos['shares'] * curr_p; diff = curr_val - alloc_per_stock
                if abs(diff) > 50:
                    action = '🟠 KÂR AL (Azalt)' if diff > 0 else '🔵 EKLEME (Artır)'
                    cash += diff
                    holdings[tkr]['shares'] = alloc_per_stock / curr_p
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': action, 'İşlem Hacmi ₺': f"{abs(diff):.2f}", 'Fiyat ₺': f"{curr_p:.2f}", 'Anlık Net K/Z': f"{((curr_p/pos['buy_price']-1)*100):+.1f}%"})
            else:
                if q['price'] > 0:
                    holdings[tkr] = {'shares': alloc_per_stock / q['price'], 'buy_price': q['price'], 'buy_date': rdate, 'score': q['score'], 'max': q['max']}
                    cash -= alloc_per_stock
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🟢 ALIŞ (Giriş)', 'İşlem Hacmi ₺': f"{alloc_per_stock:.2f}", 'Fiyat ₺': f"{q['price']:.2f}", 'Anlık Net K/Z': '-'})

    pv_df = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    return pv_df, ((bm.loc[start_date:bm.index[-1]] / float(bm.loc[start_date])) * start_capital), pd.DataFrame(trades), [], pd.DataFrame(monthly_rows)

def calc_stats(pv_df, bm_norm, start_capital):
    pv = pv_df['value']; total = (pv.iloc[-1] / start_capital - 1) * 100
    return {'Toplam Getiri': f"{total:+.1f}%", 'Son Değer': f"₺{pv.iloc[-1]:,.0f}"}

STRAT_COLORS = {'emre': '#22c55e', 'momentum': '#a78bfa'}
STRAT_LABELS = {'emre': '🟢 Emre\'nin Makro Stratejisi', 'momentum': '🟣 Momentum'}

def build_perf_chart(results_map, start_capital):
    fig = make_subplots(rows=1, cols=1)
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        fig.add_trace(go.Scatter(x=pv_df.index, y=(pv_df['value'] / start_capital - 1) * 100, name=STRAT_LABELS.get(strat, strat), line=dict(color=STRAT_COLORS.get(strat, '#94a3b8'), width=2)))
    fig.update_layout(height=400, paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14', font=dict(family='JetBrains Mono'))
    return fig
