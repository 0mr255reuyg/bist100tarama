import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sectors import get_sector

def _sma(s, n):  return s.rolling(n).mean()
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
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
    return float(v.iloc[-1]) / v.iloc[-n-1:-1].mean()

def _rs_at(c_s, bm, date, days=20):
    bm_valid = bm.index[bm.index <= date]
    if len(c_s) < days or len(bm_valid) < days: return np.nan
    bm_s = bm.loc[bm_valid]
    return ((float(c_s.iloc[-1]) / float(c_s.iloc[-days])) - 1) - ((float(bm_s.iloc[-1]) / float(bm_s.iloc[-days])) - 1)

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
        c = df['Close'].squeeze(); h = df['High'].squeeze(); lo = df['Low'].squeeze(); v = df['Volume'].squeeze(); bm = bm_df['Close'].squeeze()
        valid = c.index[c.index <= date]
        if len(valid) < 55: return 0, 5
        c_s = c.loc[valid]; h_s = h.loc[valid]; lo_s = lo.loc[valid]; v_s = v.loc[valid]
        price = float(c_s.iloc[-1])
        s20 = float(_sma(c_s, 20).iloc[-1]); s50 = float(_sma(c_s, 50).iloc[-1])
        rsi_v = float(_rsi(c_s).iloc[-1]); rs = _rs_at(c_s, bm, date, days=20); vr = _vol_ratio_at(v_s, 20)

        if strategy == "emre":
            checks = [(rs is not None) and (not np.isnan(rs)) and rs > 0, price > s20 and price > s50, (not np.isnan(vr)) and vr >= 0.9, (not np.isnan(rsi_v)) and rsi_v < 80]
            score = sum(checks); max_score = 4
            if get_sector(ticker) in _get_historical_macro_sectors(date): score += 1
            max_score += 1
            return score, max_score
        elif strategy == "momentum":
            _, _, hl = _macd(c_s)
            return sum([price > s50, float(hl.iloc[-1]) > 0 and float(hl.iloc[-1]) > float(hl.iloc[-2])]), 4
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
    if len(bm_idx) < 10: return None, None, None, [], pd.DataFrame()
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
            if df is None or len(df) < 25: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate, tkr)
            c = df['Close'].squeeze(); vd = c.index[c.index <= rdate]
            if len(vd): candidates.append({'ticker': tkr, 'score': sc, 'max': mx, 'price': float(c.loc[vd[-1]]), 'rs': _rs_at(c.loc[vd], bm, rdate, days=20) or -999})

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

        # Çıkan Hisselerin Tam Satışı
        for tkr in list(holdings.keys()):
            if tkr not in target_tickers:
                pos = holdings[tkr]; sell_p = current_prices.get(tkr, pos['buy_price'])
                cash += pos['shares'] * sell_p
                trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🔴 SATIŞ (Çıkış)', 'İşlem Hacmi ₺': f"{(pos['shares']*sell_p):.2f}", 'Fiyat ₺': f"{sell_p:.2f}", 'Anlık Net K/Z': f"{((sell_p/pos['buy_price']-1)*100):+.1f}%"})
                del holdings[tkr]

        # Kalan Hisselerde Kâr Al / Ekleme Rebalans Hareketleri
        for tkr, q in target_tickers.items():
            if tkr in holdings:
                pos = holdings[tkr]; curr_p = current_prices[tkr]; curr_val = pos['shares'] * curr_p; diff = curr_val - alloc_per_stock
                if abs(diff) > 50:
                    action = '条 🟠 KÂR AL (Azalt)' if diff > 0 else '🎰 🔵 EKLEME (Artır)'
                    cash += diff
                    if diff < 0: holdings[tkr]['buy_price'] = (pos['shares'] * pos['buy_price'] + abs(diff)) / (alloc_per_stock / curr_p)
                    holdings[tkr]['shares'] = alloc_per_stock / curr_p
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': action, 'İşlem Hacmi ₺': f"{abs(diff):.2f}", 'Fiyat ₺': f"{curr_p:.2f}", 'Anlık Net K/Z': f"{((curr_p/pos['buy_price']-1)*100):+.1f}%"})
            else:
                if q['price'] > 0:
                    holdings[tkr] = {'shares': alloc_per_stock / q['price'], 'buy_price': q['price'], 'buy_date': rdate, 'score': q['score'], 'max': q['max']}
                    cash -= alloc_per_stock
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem Türü': '🚀 🟢 ALIŞ (Giriş)', 'İşlem Hacmi ₺': f"{alloc_per_stock:.2f}", 'Fiyat ₺': f"{q['price']:.2f}", 'Anlık Net K/Z': '-'})

    pv_df = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    monthly_df = pd.DataFrame(monthly_rows)
    if len(monthly_df) > 1:
        monthly_df['Aylık Değişim %'] = monthly_df['Portföy Büyüklüğü'].pct_change() * 100
        monthly_df['Aylık Değişim %'] = monthly_df['Aylık Değişim %'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else '-')
        monthly_df['Portföy Büyüklüğü'] = monthly_df['Portföy Büyüklüğü'].apply(lambda x: f"₺{x:,.0f}")
    return pv_df, ((bm.loc[start_date:bm.index[-1]] / float(bm.loc[start_date])) * start_capital), pd.DataFrame(trades), [], monthly_df
