"""
Backtest Engine — BIST 100 Strateji Tarayıcı
Her ayın ilk borsa günü en yüksek puanlı TOP 5 hisseyi al.
Öbür ay top 5'te yoksa sat. 2 yıl · günlük · 100k₺.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sectors import get_sector


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
    (score, max_score)
    """
    try:
        c  = df['Close'].squeeze()
        h  = df['High'].squeeze()
        lo = df['Low'].squeeze()
        v  = df['Volume'].squeeze()
        bm = bm_df['Close'].squeeze()

        valid = c.index[c.index <= date]
        if len(valid) < 55: return 0, 4 # Veri yetersizse 0 puan dön
        c_s  = c.loc[valid]; h_s = h.loc[valid]
        lo_s = lo.loc[valid]; v_s = v.loc[valid]
        price = float(c_s.iloc[-1])
        vr    = _vol_ratio_at(v_s)

        if strategy == "emre":
            s20   = float(_sma(c_s, 20).iloc[-1])
            s50   = float(_sma(c_s, 50).iloc[-1])
            rsi_v = float(_rsi(c_s).iloc[-1])
            rs    = _rs_at(c_s, bm, date, days=20)
            vol5  = float(v_s.iloc[-5:].mean())
            vol20 = float(v_s.iloc[-20:].mean()) if len(v_s) >= 20 else vol5
            
            checks = [
                (rs is not None) and (not np.isnan(rs)) and rs > 0,
                price > s20 and price > s50,
                vol5 >= vol20 * 0.85,
                (not np.isnan(rsi_v)) and rsi_v < 80,
            ]
            return sum(checks), 4

        elif strategy == "momentum":
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

    except Exception:
        pass
    return 0, 4


# ── AY BAŞLARI ────────────────────────────────────────────────────────────────
def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())


# ── ANA BACKTEST ──────────────────────────────────────────────────────────────
def run_backtest(strategy, stock_data, benchmark_df, start_capital=100_000, top_n=5):
    """
    Her ay başında en yüksek puanlı top_n hisseyi al. Eksik kriter olsa bile 5'e tamamla.
    Döner: pv_df, bm_norm, trades_df, active_now, monthly_table
    """
    bm = benchmark_df['Close'].squeeze()
    
    start_date = pd.Timestamp('2024-06-01')
    end_date   = pd.Timestamp(bm.index[-1])

    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    if len(bm_idx) < 10:
        return None, None, None, [], pd.DataFrame()

    rebal_dates = _month_starts(bm_idx, start_date, end_date)
    if len(rebal_dates) < 2:
        return None, None, None, [], pd.DataFrame()

    cash     = float(start_capital)
    holdings = {}   
    pv_log   = []
    trades   = []
    monthly_rows = []   

    for i, rdate in enumerate(rebal_dates):
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
            if df is None or len(df) < 25: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate)
            
            c = df['Close'].squeeze()
            bm_c = benchmark_df['Close'].squeeze()
            vd = c.index[c.index <= rdate]
            
            if len(vd):
                price = float(c.loc[vd[-1]])
                c_s   = c.loc[vd]
                rs_val = _rs_at(c_s, bm_c, rdate, days=20)
                rs_val = rs_val if (rs_val is not None and not np.isnan(rs_val)) else -999
                
                fk_val = float(df['F/K'].iloc[-1]) if 'F/K' in df.columns else 999.0
                net_kar = float(df['Net_Kar'].iloc[-1]) if 'Net_Kar' in df.columns else 0.0

                candidates.append({
                    'ticker': tkr, 'score': sc, 'max': mx, 'price': price, 
                    'rs': rs_val, 'fk': fk_val, 'net_kar': net_kar
                })

        # Eşitlik Bozucu
        candidates.sort(key=lambda x: (
            x['score'], 
            x['net_kar'], 
            -x['fk'] if x['fk'] > 0 else -999, 
            x['rs']
        ), reverse=True)
        
        # Max 2 Hisse / Sektör Filtresi (TÜM STRATEJİLER İÇİN GEÇERLİ)
        top5 = []
        sector_counts = {}
        for cand in candidates:
            sect = get_sector(cand['ticker'])
            if sector_counts.get(sect, 0) < 2:
                top5.append(cand)
                sector_counts[sect] = sector_counts.get(sect, 0) + 1
            if len(top5) == top_n:
                break

        target_tickers = {q['ticker'] for q in top5}

        row = {'Ay': rdate.strftime('%b %Y'), 'Port. Değer': port_val}
        for rank, q in enumerate(top5, 1):
            row[f'#{rank}'] = f"{q['ticker'].replace('.IS','')} ({q['score']}/{q['max']})"
        monthly_rows.append(row)

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

    monthly_df = pd.DataFrame(monthly_rows)
    if len(monthly_df) > 1:
        monthly_df['Aylık P&L'] = monthly_df['Port. Değer'].pct_change() * 100
        monthly_df['Aylık P&L'] = monthly_df['Aylık P&L'].apply(
            lambda x: f"{x:+.1f}%" if not pd.isna(x) else '-'
        )
        monthly_df['Port. Değer'] = monthly_df['Port. Değer'].apply(
            lambda x: f"₺{x:,.0f}"
        )

    bm_s    = bm.loc[start_date:last_date]
    if len(bm_s) > 0:
        bm_norm = (bm_s / float(bm_s.iloc[0])) * start_capital
    else:
        bm_norm = None

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
    'emre':     '#f59e0b', 
    'rs':       '#38bdf8',
    'momentum': '#a78bfa',
    'trend':    '#4ade80',
}
STRAT_LABELS = {
    'emre':     '🟠 Emre Stratejisi',
    'rs':       '🔵 Rölatif Güç',
    'momentum': '🟣 Momentum',
    'trend':    '🟢 Trend Sürücüsü',
}


# ── PERFORMANS GRAFİĞİ ────────────────────────────────────────────────────────
def build_perf_chart(results_map, start_capital):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.04)

    bm_drawn = False
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        color  = STRAT_COLORS.get(strat, '#94a3b8')
        label  = STRAT_LABELS.get(strat, strat)
        pv     = pv_df['value']
        cumret = (pv / start_capital - 1) * 100

        fig.add_
