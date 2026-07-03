"""
Backtest Engine — BIST 100 Strateji Tarayıcı
Kümülatif Eşit Ağırlıklı Rebalans ve İşlem Log Defteri Entegrasyonu
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sectors import get_sector

# ── 1. İNDİKATÖRLER VE YARDIMCI FONKSİYONLAR ──────────────────────────────────
def _sma(s, n):  return s.rolling(n).mean()
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

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

def _month_starts(index, start, end):
    dates = index[(index >= start) & (index <= end)]
    df_tmp = pd.DataFrame({'d': dates, 'ym': [x.strftime('%Y-%m') for x in dates]})
    return list(df_tmp.groupby('ym')['d'].first())

# Bazı stratejiler için minimum kalite eşiği ve hedef pozisyon aralığı.
# Burada olmayan stratejiler (emre, momentum) eski davranışı korur: skora
# bakılmaksızın her zaman tam top_n (5) hisse seçilir. "claude" için eşiğin
# altında kalan adaylar havuza hiç girmez; 4-5 arası pozisyon hedeflenir.
STRATEGY_MIN_SCORE = {"claude": 4}
STRATEGY_TARGET_RANGE = {"claude": (4, 5)}

def _pick_candidates(candidates, strategy, top_n=5, max_per_sector=2):
    """Skor+RS'ye göre zaten sıralanmış adaylardan portföyü seçer.
    Sektör başına en fazla `max_per_sector` hisse kuralı her zaman geçerlidir.
    'claude' stratejisinde ek olarak minimum skor eşiği ve 4-5 pozisyon
    hedefi uygulanır; diğerlerinde davranış (her zaman top_n hisse) değişmez."""
    def _fill(pool, limit):
        picked, sector_counts = [], {}
        for cand in pool:
            sect = get_sector(cand['ticker'])
            if sector_counts.get(sect, 0) < max_per_sector:
                picked.append(cand); sector_counts[sect] = sector_counts.get(sect, 0) + 1
            if len(picked) == limit: break
        return picked

    min_score = STRATEGY_MIN_SCORE.get(strategy)
    if min_score is None:
        return _fill(candidates, top_n)

    target_min, target_max = STRATEGY_TARGET_RANGE.get(strategy, (top_n, top_n))
    qualified = [c for c in candidates if c['score'] >= min_score]
    picked = _fill(qualified, target_max)
    if len(picked) < target_min:
        relaxed = [c for c in candidates if c['score'] >= max(min_score - 1, 0)]
        picked = _fill(relaxed, target_min)
    return picked

# ── 1b. TARİHSEL TLREF MAKRO REJİMİ ───────────────────────────────────────────
# app.py'deki canlı get_macro_regime() ile AYNI mantık (SMA8/54, ADX, D+/D-,
# Plato tespiti); burada geçmişteki her rebalans tarihi için ayrı ayrı
# hesaplanır ki backtest, canlı stratejiyle tutarlı bir "makro rüzgar" kriteri
# kullansın. tlref_weekly verilmezse (ör. EVDS API key yoksa) None döner ve
# _score_at basit fiyat-trend proxy'sine düşer.
_REGIME_SECTOR_MAP = {
    "risk_off": ["Gıda ve Perakende", "İletişim", "Sağlık"],
    "risk_on":  ["Teknoloji ve Yazılım", "Enerji", "Otomotiv", "Sanayi ve Kimya"],
    "plato":    ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm"],
}

def _calc_adx_hist(h, l, c, n=14):
    up = h.diff(); down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=h.index).ewm(span=n, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=h.index).ewm(span=n, adjust=False).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=n, adjust=False).mean()
    return adx, plus_di, minus_di

def _regime_sectors_at(tlref_weekly, date):
    """Belirli bir geçmiş tarihte (o tarihe kadarki veriyle) hangi rejimin
    aktif olduğunu hesaplayıp o rejimin olumlu sektörlerini döndürür."""
    if tlref_weekly is None or tlref_weekly.empty:
        return None
    df_v = tlref_weekly.loc[tlref_weekly.index <= date]
    if len(df_v) < 55:
        return None
    c = df_v['Close'].squeeze(); h = df_v['High'].squeeze(); l = df_v['Low'].squeeze()
    sma8 = float(c.rolling(8).mean().iloc[-1])
    sma54 = float(c.rolling(54).mean().iloc[-1])
    adx, plus_di, minus_di = _calc_adx_hist(h, l, c)
    adx_val = float(adx.iloc[-1]); p_di = float(plus_di.iloc[-1]); m_di = float(minus_di.iloc[-1])
    spread_pct = abs(sma8 - sma54) / sma54 * 100 if sma54 else 0.0
    is_plato = spread_pct < 3 and adx_val < 20

    if is_plato:
        return _REGIME_SECTOR_MAP["plato"]
    elif sma8 > sma54 and p_di > m_di and adx_val >= 20:
        return _REGIME_SECTOR_MAP["risk_off"]
    elif sma8 < sma54 and m_di > p_di and adx_val >= 20:
        return _REGIME_SECTOR_MAP["risk_on"]
    return _REGIME_SECTOR_MAP["plato"]

# ── 2. KRİTER SKORU (GEÇMİŞ TARİH İÇİN) ───────────────────────────────────────
def _max_score_for(strategy):
    return {"emre": 5, "momentum": 4, "claude": 6}.get(strategy, 5)

def _score_at(strategy, df, bm_df, date, ticker=None, regime_sectors=None):
    try:
        c = df['Close'].squeeze(); h = df['High'].squeeze()
        lo = df['Low'].squeeze(); v = df['Volume'].squeeze()
        bm = bm_df['Close'].squeeze()

        valid = c.index[c.index <= date]
        if len(valid) < 55: return 0, _max_score_for(strategy)
        c_s = c.loc[valid]; h_s = h.loc[valid]
        lo_s = lo.loc[valid]; v_s = v.loc[valid]
        price = float(c_s.iloc[-1])
        vr = _vol_ratio_at(v_s)

        if strategy == "emre":
            s20 = float(_sma(c_s, 20).iloc[-1])
            s50 = float(_sma(c_s, 50).iloc[-1])
            rsi_v = float(_rsi(c_s).iloc[-1])
            rs = _rs_at(c_s, bm, date, days=20)

            if regime_sectors is not None and ticker is not None:
                # Canlı stratejiyle tutarlı: gerçek TLREF rejimine göre sektör hizalaması
                macro_bonus = 1 if get_sector(ticker) in regime_sectors else 0
            else:
                # TLREF verisi yoksa (ör. EVDS key girilmemiş) basit fiyat-trend proxy'si
                macro_bonus = 1 if price > s50 else 0

            checks = [
                (rs is not None) and (not np.isnan(rs)) and rs > 0,
                price > s20 and price > s50,
                (not np.isnan(vr)) and vr >= 0.9,
                (not np.isnan(rsi_v)) and rsi_v < 80,
            ]
            total_score = sum(checks) + macro_bonus
            return min(total_score, 5), 5

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

        elif strategy == "claude":
            # Claude Stratejisi — canlı score_claude (app.py) ile birebir aynı 6 kriter:
            # çoklu-onaylı trend yapısı, kalıcı (20g+60g) RS, getiri/risk verimliliği,
            # sağlıklı RSI bandı, hacim katılımı, gerçek TLREF/TCMB rejimine göre makro rüzgar.
            sma50_hist = _sma(c_s, 50)
            sma50_valid = sma50_hist.dropna()
            if len(sma50_valid) < 7:
                return 0, 6
            s50 = float(sma50_hist.iloc[-1])
            s50_prev = float(sma50_valid.iloc[-6])
            rsi_v = float(_rsi(c_s).iloc[-1])
            rs20 = _rs_at(c_s, bm, date, days=20)
            rs60 = _rs_at(c_s, bm, date, days=60)
            a14 = _atr(h_s, lo_s, c_s)
            atr_now = float(a14.iloc[-1])
            atr_pct = (atr_now / price * 100) if price > 0 else np.nan
            ret_20 = (price / float(c_s.iloc[-21]) - 1) * 100 if len(c_s) > 21 else np.nan
            eff_ratio = (ret_20 / (atr_pct * (20 ** 0.5))) if (not np.isnan(atr_pct) and atr_pct > 0 and not np.isnan(ret_20)) else np.nan

            if regime_sectors is not None and ticker is not None:
                macro_ok = get_sector(ticker) in regime_sectors
            else:
                macro_ok = price > s50  # TLREF verisi yoksa basit proxy

            trend_ok = price > s50 and s50 > s50_prev
            rs_ok = (rs20 is not None and not np.isnan(rs20) and rs20 > 0) and \
                    (rs60 is not None and not np.isnan(rs60) and rs60 > 0)

            checks = [
                trend_ok,
                rs_ok,
                (not np.isnan(eff_ratio)) and eff_ratio > 0.3,
                45 <= rsi_v <= 72,
                (not np.isnan(vr)) and vr >= 1.0,
                macro_ok,
            ]
            return sum(checks), 6

    except Exception:
        pass
    return 0, _max_score_for(strategy)

# ── 3. ANA BACKTEST MOTORU (REBALANS & LOG DEFTERİ) ───────────────────────────
def run_backtest(strategy, stock_data, benchmark_df, tlref_weekly=None, start_capital=100_000, top_n=5):
    bm = benchmark_df['Close'].squeeze()
    start_date = pd.Timestamp('2024-06-01')
    end_date   = pd.Timestamp(bm.index[-1])
    bm_idx = bm.index[(bm.index >= start_date) & (bm.index <= end_date)]
    
    if len(bm_idx) < 10: return None, None, None, [], pd.DataFrame()
    rebal_dates = _month_starts(bm_idx, start_date, end_date)
    if len(rebal_dates) < 2: return None, None, None, [], pd.DataFrame()

    cash = float(start_capital)
    holdings = {}   
    pv_log = []
    trades = []
    monthly_rows = []   

    for i, rdate in enumerate(rebal_dates):
        # 1. Portföyün o anki toplam değerini hesapla
        port_val = cash
        for tkr, pos in holdings.items():
            if tkr in stock_data:
                c = stock_data[tkr]['Close'].squeeze()
                vd = c.index[c.index <= rdate]
                if len(vd): port_val += pos['shares'] * float(c.loc[vd[-1]])
        pv_log.append({'date': rdate, 'value': port_val})

        # 1b. Bu rebalans tarihindeki TLREF makro rejimi (tüm hisseler için ortak)
        regime_sectors = _regime_sectors_at(tlref_weekly, rdate)

        # 2. Hisse Havuzunu Tara ve Puanla
        candidates = []
        for tkr, df in stock_data.items():
            if df is None or len(df) < 55: continue
            sc, mx = _score_at(strategy, df, benchmark_df, rdate, ticker=tkr, regime_sectors=regime_sectors)
            c = df['Close'].squeeze()
            vd = c.index[c.index <= rdate]
            if len(vd):
                price = float(c.loc[vd[-1]])
                rs_val = _rs_at(c.loc[vd], bm, rdate, days=20)
                rs_val = rs_val if (rs_val is not None and not np.isnan(rs_val)) else -999
                candidates.append({'ticker': tkr, 'score': sc, 'max': mx, 'price': price, 'rs': rs_val})

        # Puan ve RS'ye göre sırala (Eşitlik bozucu)
        candidates.sort(key=lambda x: (x['score'], x['rs']), reverse=True)

        # Max 2 Sektör Kuralı ile Portföy Seçimi ("claude" için ek olarak
        # min. kalite eşiği + 4-5 pozisyon hedefi; diğerlerinde eski davranış)
        top5 = _pick_candidates(candidates, strategy, top_n=top_n, max_per_sector=2)

        target_tickers = {q['ticker'] for q in top5}
        target_alloc = port_val / max(1, len(target_tickers)) if target_tickers else 0

        # Aylık Portföy Özet Satırı
        row = {'Ay': rdate.strftime('%b %Y'), 'Port. Değer': port_val}
        for rank, q in enumerate(top5, 1):
            row[f'#{rank}'] = f"{q['ticker'].replace('.IS','')} ({q['score']}/{q['max']})"
        monthly_rows.append(row)

        # 3. SATIŞ ve REBALANS (Log Defteri Mantığı)
        # 3A. Tamamen Çıkış Yapılanlar (Listede artık olmayanlar)
        for tkr in list(holdings.keys()):
            if tkr not in target_tickers:
                pos = holdings[tkr]
                c = stock_data[tkr]['Close'].squeeze()
                vd = c.index[c.index <= rdate]
                sell_p = float(c.loc[vd[-1]]) if len(vd) else pos['avg_cost']
                pnl = (sell_p / pos['avg_cost'] - 1) * 100
                cash += pos['shares'] * sell_p
                trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem': '🔴 TAMAMEN SATIŞ', 'Maliyet ₺': f"{pos['avg_cost']:.2f}", 'Fiyat ₺': f"{sell_p:.2f}", 'P&L': f"{pnl:+.1f}%"})
                del holdings[tkr]

        # 3B. Tutulanların Dengelenmesi (Kâr Al veya Ekleme) ve Yeni Alışlar
        for q in top5:
            tkr = q['ticker']
            price = q['price']
            
            if tkr in holdings:
                pos = holdings[tkr]
                current_val = pos['shares'] * price
                diff = target_alloc - current_val
                
                if diff > (target_alloc * 0.05): # %5'ten fazla geride kaldıysa ekle
                    buy_amt = min(diff, cash)
                    if buy_amt > 50:
                        new_shares = buy_amt / price
                        total_cost = (pos['shares'] * pos['avg_cost']) + buy_amt
                        pos['shares'] += new_shares
                        pos['avg_cost'] = total_cost / pos['shares']
                        cash -= buy_amt
                        trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem': '🔵 EKLEME', 'Maliyet ₺': f"{pos['avg_cost']:.2f}", 'Fiyat ₺': f"{price:.2f}", 'P&L': "-"})
                
                elif diff < -(target_alloc * 0.05): # %5'ten fazla şiştiyse kâr al
                    sell_amt = abs(diff)
                    sell_shares = sell_amt / price
                    pnl = (price / pos['avg_cost'] - 1) * 100
                    pos['shares'] -= sell_shares
                    cash += sell_amt
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem': '🟠 KÂR AL', 'Maliyet ₺': f"{pos['avg_cost']:.2f}", 'Fiyat ₺': f"{price:.2f}", 'P&L': f"{pnl:+.1f}%"})
            
            else: # 3C. Yepyeni Alışlar
                buy_amt = min(target_alloc, cash)
                if buy_amt > 50 and price > 0:
                    shares = buy_amt / price
                    holdings[tkr] = {'shares': shares, 'avg_cost': price, 'buy_date': rdate, 'score': q['score'], 'max': q['max']}
                    cash -= buy_amt
                    trades.append({'Ay': rdate.strftime('%b %Y'), 'Hisse': tkr.replace('.IS',''), 'İşlem': '🟢 YENİ ALIŞ', 'Maliyet ₺': f"{price:.2f}", 'Fiyat ₺': f"{price:.2f}", 'P&L': "-"})

    # Dönem Sonu Değerlemesi
    last_date = bm_idx[-1]
    final_val = cash
    active_now = []
    for tkr, pos in holdings.items():
        if tkr in stock_data:
            c = stock_data[tkr]['Close'].squeeze()
            vd = c.index[c.index <= last_date]
            cur_p = float(c.loc[vd[-1]]) if len(vd) else pos['avg_cost']
            final_val += pos['shares'] * cur_p
            pnl = (cur_p / pos['avg_cost'] - 1) * 100
            active_now.append({'ticker': tkr.replace('.IS',''), 'buy_date': pos['buy_date'].strftime('%d.%m.%Y'), 'buy_price': pos['avg_cost'], 'current_price': cur_p, 'pnl_pct': pnl, 'score': pos['score'], 'max': pos['max']})
    pv_log.append({'date': last_date, 'value': final_val})

    pv_df = pd.DataFrame(pv_log).drop_duplicates('date').set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    monthly_df = pd.DataFrame(monthly_rows)
    if len(monthly_df) > 1:
        monthly_df['Aylık P&L'] = monthly_df['Port. Değer'].pct_change() * 100
        monthly_df['Aylık P&L'] = monthly_df['Aylık P&L'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else '-')
        monthly_df['Port. Değer'] = monthly_df['Port. Değer'].apply(lambda x: f"₺{x:,.0f}")

    bm_s = bm.loc[start_date:last_date]
    bm_norm = (bm_s / float(bm_s.iloc[0])) * start_capital if len(bm_s) > 0 else None

    return pv_df, bm_norm, trades_df, active_now, monthly_df

# ── 4. İSTATİSTİK VE GRAFİK YARDIMCILARI ──────────────────────────────────────
def calc_stats(pv_df, bm_norm, start_capital):
    pv = pv_df['value']; total = (pv.iloc[-1] / start_capital - 1) * 100
    n_yrs = max((pv_df.index[-1] - pv_df.index[0]).days / 365, 0.01)
    cagr = ((pv.iloc[-1] / start_capital) ** (1/n_yrs) - 1) * 100
    dd = (pv / pv.cummax() - 1) * 100
    max_dd = dd.min()
    bm_ret = (bm_norm.iloc[-1] / start_capital - 1) * 100 if bm_norm is not None else None
    alpha = total - bm_ret if bm_ret is not None else None
    return {'Toplam Getiri': f"{total:+.1f}%", 'CAGR': f"{cagr:+.1f}%", 'Max Drawdown': f"{max_dd:.1f}%", 'BIST100 Getirisi': f"{bm_ret:+.1f}%" if bm_ret is not None else 'N/A', 'Alpha': f"{alpha:+.1f}%" if alpha is not None else 'N/A', 'Son Değer': f"₺{pv.iloc[-1]:,.0f}"}

STRAT_COLORS = { 'emre': '#f59e0b', 'momentum': '#a78bfa', 'claude': '#38bdf8' }
STRAT_LABELS = { 'emre': "🟠 Emre'nin Makro Stratejisi", 'momentum': '🟣 Momentum', 'claude': '🔵 Claude Stratejisi' }

def build_perf_chart(results_map, start_capital):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    bm_drawn = False
    for strat, (pv_df, bm_norm) in results_map.items():
        if pv_df is None: continue
        color = STRAT_COLORS.get(strat, '#94a3b8'); label = STRAT_LABELS.get(strat, strat)
        pv = pv_df['value']; cumret = (pv / start_capital - 1) * 100
        fig.add_trace(go.Scatter(x=pv_df.index, y=cumret, name=label, line=dict(color=color, width=2.2), hovertemplate=f"<b>{label}</b><br>%{{y:+.1f}}%<extra></extra>"), row=1, col=1)
        dd = (pv / pv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=pv_df.index, y=dd, name=f"DD {label}", line=dict(color=color, width=1.2), fill='tozeroy', opacity=0.4, showlegend=False, hovertemplate="DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
        if not bm_drawn and bm_norm is not None:
            bm_ret = (bm_norm / start_capital - 1) * 100
            fig.add_trace(go.Scatter(x=bm_norm.index, y=bm_ret, name='BIST 100', line=dict(color='#ef4444', width=2, dash='dot'), hovertemplate="BIST100: %{y:+.1f}%<extra></extra>"), row=1, col=1)
            bm_drawn = True

    fig.add_hline(y=0, line=dict(color='#333', width=0.8, dash='dash'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#333', width=0.6), row=2, col=1)
    fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', color='#a3a3a3', size=11), legend=dict(orientation='h', y=1.03, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11, color="#fff")), margin=dict(l=10, r=10, t=20, b=10), hovermode='x unified')
    fig.update_yaxes(ticksuffix='%', gridcolor='#1e1e1e', row=1, col=1); fig.update_yaxes(ticksuffix='%', gridcolor='#1e1e1e', row=2, col=1)
    fig.update_xaxes(gridcolor='#1e1e1e', row=1, col=1); fig.update_xaxes(gridcolor='#1e1e1e', row=2, col=1)
    return fig
