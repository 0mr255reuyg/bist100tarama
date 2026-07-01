import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sectors import get_sector

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

def _rs_at(c_s, bm, date, days=20):
    bm_valid = bm.index[bm.index <= date]
    if len(c_s) < days or len(bm_valid) < days: return np.nan
    bm_s = bm.loc[bm_valid]
    sr = (float(c_s.iloc[-1]) / float(c_s.iloc[-days])) - 1
    br = (float(bm_s.iloc[-1]) / float(bm_s.iloc[-days])) - 1
    return sr - br

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
        c  = df['Close'].squeeze()
        h  = df['High'].squeeze()
        lo = df['Low'].squeeze()
        v  = df['Volume'].squeeze()
        bm = bm_df['Close'].squeeze()

        valid = c.index[c.index <= date]
        if len(valid) < 55: return 0, 5
        c_s  = c.loc[valid]; h_s = h.loc[valid]; lo_s = lo.loc[valid]; v_s = v.loc[valid]
        price = float(c_s.iloc[-1])

        s20   = float(_sma(c_s, 20).iloc[-1])
        s50   = float(_sma(c_s, 50).iloc[-1])
        rsi_v = float(_rsi(c_s).iloc[-1])
        rs    = _rs_at(c_s, bm, date, days=20)
        vr    = _vol_ratio_at(v_s, 20)

        if strategy == "emre":
            checks = [
                (rs is not None) and (not np.isnan(rs)) and rs > 0,
                price > s20 and price > s50,
                (not np.isnan(vr)) and vr >= 0.9,
                (not np.isnan(rsi_v)) and rsi_v < 80,
            ]
            score = sum(checks)
            max_score = 4
            
            allowed_sectors = _get_historical_macro_sectors(date)
            is_ok = get_sector(ticker) in allowed_sectors
            if is_ok: score += 1
            max_score += 1
                
            return score, max_score

        elif strategy == "momentum":
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
    except Exception: pass
    return 0, 4

def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())

def run_backtest(strategy, stock_data, benchmark_df, start_capital=100_000, top_n=5):
    bm = benchmark_df['Close'].squeeze()
    start_date = pd.Timestamp('2024-06-01')
    end_date   = pd.Timestamp(bm.index[-1])

    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    if len(bm_idx) < 10: return None, None, None, [], pd.DataFrame()

    rebal_dates = _month_starts(bm_idx, start_date, end_date)
    if len(rebal_dates) < 2: return None, None, None, [], pd.DataFrame()

    cash = float(start_capital); holdings = {}; pv_log = []; trades = []; monthly_rows = []   

    for i, rdate in enumerate(rebal_dates):
        port_val = cash
        current_prices = {}
        
        for tkr, pos in holdings.items():
            if tkr in stock_data:
                c = stock_data[tkr]['Close'].squeeze()
                vd = c.index[c.index <= rdate]
                p = float(c.loc[vd[-1]]) if len(vd) else pos['buy_price']
                current_prices[tkr] = p
                port_val += pos['shares'] * p
        pv_log.append({'date': rdate, 'value': port_val})

        candidates = []
        for tkr, df in stock_data.items():
            if df is None or len(df) < 25: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate, tkr)
            c = df['Close'].squeeze(); vd = c.index[c.index <= rdate]
            if len(vd):
                price = float(c.loc[vd[-1]])
                rs_val = _rs_at(c.loc[vd], benchmark_df['Close'].squeeze(), rdate, days=20)
                rs_val = rs_val if (rs_val is not None and not np.isnan(rs_val)) else -999
                candidates.append({'ticker': tkr, 'score': sc, 'max': mx, 'price': price, 'rs': rs_val})

        candidates.sort(key=lambda x: (x['score'], x['rs']), reverse=True)
        
        top5 = []; sector_counts = {}
        for cand in candidates:
            sect = get_sector(cand['ticker'])
            if strategy == "emre":
                if sector_counts.get(sect, 0) < 2:
                    top5.append(cand)
                    sector_counts[sect] = sector_counts.get(sect, 0) + 1
            else:
                top5.append(cand)
            if len(top5) == top_n: break

        target_tickers = {q['ticker']: q for q in top5}
        
        row = {'Ay': rdate.strftime('%b %Y'), 'Portföy Büyüklüğü': port_val}
        for rank, q in enumerate(top5, 1):
            row[f'#{rank}'] = f"{q['ticker'].replace('.IS','')} ({q['score']}/{q['max']})"
        monthly_rows.append(row)

        alloc_per_stock = port_val / len(top5) if top5 else 0

        # Portföyden Çıkanları Sat
        for tkr in list(holdings.keys()):
            if tkr not in target_tickers:
                pos = holdings[tkr]
                sell_p = current_prices.get(tkr, pos['buy_price'])
                pnl = (sell_p / pos['buy_price'] - 1) * 100
                val = pos['shares'] * sell_p
                cash += val
                trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🔴 SATIŞ (Çıkış)', 'İşlem Hacmi ₺': f"{val:.2f}", 'Fiyat ₺': f"{sell_p:.2f}", 'Anlık Net K/Z': f"{pnl:+.1f}%"})
                del holdings[tkr]

        # Kalanları Dengele ve Yenileri Al
        for tkr, q in target_tickers.items():
            if tkr in holdings:
                pos = holdings[tkr]
                curr_p = current_prices[tkr]
                curr_val = pos['shares'] * curr_p
                diff = curr_val - alloc_per_stock
                pnl_now = (curr_p / pos['buy_price'] - 1) * 100
                
                if abs(diff) > 50:
                    action = '🟠 KÂR AL (Azalt)' if diff > 0 else '🔵 EKLEME (Artır)'
                    trade_amt = abs(diff)
                    cash += diff
                    
                    if diff < 0:
                        total_shares = alloc_per_stock / curr_p
                        new_avg_cost = (pos['shares'] * pos['buy_price'] + trade_amt) / total_shares
                        holdings[tkr]['buy_price'] = new_avg_cost
                    
                    holdings[tkr]['shares'] = alloc_per_stock / curr_p
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': action, 'İşlem Hacmi ₺': f"{trade_amt:.2f}", 'Fiyat ₺': f"{curr_p:.2f}", 'Anlık Net K/Z': f"{pnl_now:+.1f}%"})
            else:
                if q['price'] > 0:
                    holdings[tkr] = {'shares': alloc_per_stock / q['price'], 'buy_price': q['price'], 'buy_date': rdate, 'score': q['score'], 'max': q['max']}
                    cash -= alloc_per_stock
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🟢 ALIŞ (Giriş)', 'İşlem Hacmi ₺': f"{alloc_per_stock:.2f}", 'Fiyat ₺': f"{q['price']:.2f}", 'Anlık Net K/Z': '-'})

    last_date = bm_idx[-1]
    final_val = cash
    active_now = []
    for tkr, pos in holdings.items():
        if tkr in stock_data:
            c = stock_data[tkr]['Close'].squeeze()
            vd = c.index[c.index <= last_date]
            cur_p = float(c.loc[vd[-1]]) if len(vd) else pos['buy_price']
            final_val += pos['shares'] * cur_p
            active_now.append({'ticker': tkr.replace('.IS',''), 'buy_date': pos['buy_date'].strftime('%d.%m.%Y'), 'buy_price': pos['buy_price'], 'current_price': cur_p, 'pnl_pct': (cur_p / pos['buy_price'] - 1) * 100, 'score': pos['score'], 'max': pos['max']})
    pv_log.append({'date': last_date, 'value': final_val})

    pv_df = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    monthly_df = pd.DataFrame(monthly_rows)
    if len(monthly_df) > 1:
        monthly_df['Aylık Değişim %'] = monthly_df['Portföy Büyüklüğü'].pct_change() * 100
        monthly_df['Aylık Değişim %'] = monthly_df['Aylık Değişim %'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else '-')
        monthly_df['Portföy Büyüklüğü'] = monthly_df['Portföy Büyüklüğü'].apply(lambda x: f"₺{x:,.0f}")

    bm_s = bm.loc[start_date:last_date]
    if len(bm_s) > 0:
        bm_norm = (bm_s / float(bm_s.iloc[0])) * start_capital
    else:
        bm_norm = None

    return pv_df, bm_norm, trades_df, active_now, monthly_df

def calc_stats(pv_df, bm_norm, start_capital):
    pv = pv_df['value']
    total = (pv.iloc[-1] / start_capital - 1) * 100
    n_yrs = max((pv_df.index[-1] - pv_df.index[0]).days / 365, 0.01)
    cagr = ((pv.iloc[-1] / start_capital) ** (1/n_yrs) - 1) * 100
    dd = (pv / pv.cummax() - 1) * 100
    max_dd = dd.min()
    bm_ret = (bm_norm.iloc[-1] / start_capital - 1) * 100 if bm_norm is not None else None
    alpha = total - bm_ret if bm_ret is not None else None
    return {
        'Toplam Getiri': f"{total:+.1f}%",
        'CAGR': f"{cagr:+.1f}%",
        'Max Drawdown': f"{max_dd:.1f}%",
        'BIST100 Getirisi': f"{bm_ret:+.1f}%" if bm_ret is not None else 'N/A',
        'Alpha': f"{alpha:+.1f}%" if alpha is not None else 'N/A',
        'Son Değer': f"₺{pv.iloc[-1]:,.0f}"
    }

STRAT_COLORS = {'emre': '#22c55e', 'momentum': '#a78bfa'}
STRAT_LABELS = {'emre': '🟢 Emre\'nin Makro Stratejisi', 'momentum': '🟣 Momentum'}

def build_perf_chart(results_map, start_capital):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    bm_drawn = False
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        color = STRAT_COLORS.get(strat, '#94a3b8'); label = STRAT_LABELS.get(strat, strat)
        pv = pv_df['value']
        cumret = (pv / start_capital - 1) * 100
        fig.add_trace(go.Scatter(x=pv_df.index, y=cumret, name=label, line=dict(color=color, width=2.2)), row=1, col=1)
        dd = (pv / pv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=pv_df.index, y=dd, name=f"DD {label}", line=dict(color=color, width=1.2), fill='tozeroy', opacity=0.4, showlegend=False), row=2, col=1)
        if not bm_drawn and bm_norm is not None:
            bm_ret = (bm_norm / start_capital - 1) * 100
            fig.add_trace(go.Scatter(x=bm_norm.index, y=bm_ret, name='BIST 100', line=dict(color='#ef4444', width=2, dash='dot')), row=1, col=1)
            bm_drawn = True

    fig.add_hline(y=0, line=dict(color='#475569', width=0.8, dash='dash'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#475569', width=0.6), row=2, col=1)
    fig.update_layout(height=500, paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14', font=dict(family='JetBrains Mono', color='#94a3b8', size=11), legend=dict(orientation='h', y=1.03, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11)), margin=dict(l=10, r=10, t=20, b=10), hovermode='x unified', yaxis=dict(ticksuffix='%', gridcolor='#1e2535'), yaxis2=dict(ticksuffix='%', gridcolor='#1e2535'), xaxis=dict(gridcolor='#1e2535'), xaxis2=dict(gridcolor='#1e2535'))
    return fig
