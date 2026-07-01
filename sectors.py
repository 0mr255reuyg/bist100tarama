import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

FALLBACK_SECTOR_MAP = {
    "AKBNK": "Banka", "GARAN": "Banka", "ISCTR": "Banka", "YKBNK": "Banka", "HALKB": "Banka", "VAKBN": "Banka", "TSKB": "Banka", "SKBNK": "Banka",
    "ALBRK": "Katılım ve Evim Sistemleri", "KTLEV": "Katılım ve Evim Sistemleri", "ISMEN": "Katılım ve Evim Sistemleri", "INVES": "Katılım ve Evim Sistemleri",
    "EKGYO": "İnşaat ve GMYO", "TRGYO": "İnşaat ve GMYO", "ISGYO": "İnşaat ve GMYO", "ZRGYO": "İnşaat ve GMYO", "SNGYO": "İnşaat ve GMYO", "AKGYO": "İnşaat ve GMYO", "OZGYO": "İnşaat ve GMYO",
    "EREGL": "Çelik ve Metal", "KRDMD": "Çelik ve Metal", "BRSAN": "Çelik ve Metal", "KCAER": "Çelik ve Metal", "CEMTS": "Çelik ve Metal",
    "AKSEN": "Enerji", "ENJSA": "Enerji", "ASTOR": "Enerji", "GESAN": "Enerji", "EUPWR": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji", "SMRTG": "Enerji", "ZOREN": "Enerji", "CANTE": "Enerji", "ODAS": "Enerji", "GWIND": "Enerji",
    "BIMAS": "Gıda ve Perakende", "MGROS": "Gıda ve Perakende", "SOKM": "Gıda ve Perakende", "CCOLA": "Gıda ve Perakende", "AEFES": "Gıda ve Perakende", "ULKER": "Gıda ve Perakende", "TATGD": "Gıda ve Perakende", "TUKAS": "Gıda ve Perakende", "TABGD": "Gıda ve Perakende", "KLRHO": "Gıda ve Perakende",
    "KCHOL": "Holding ve Yatırım", "SAHOL": "Holding ve Yatırım", "ALARK": "Holding ve Yatırım", "DOHOL": "Holding ve Yatırım", "ENKAI": "Holding ve Yatırım", "TKFEN": "Holding ve Yatırım", "BERA": "Holding ve Yatırım", "AGHOL": "Holding ve Yatırım", "GSDHO": "Holding ve Yatırım", "AHSY": "Holding ve Yatırım",
    "FROTO": "Otomotiv", "TOASO": "Otomotiv", "DOAS": "Otomotiv", "TTRAK": "Otomotiv", "OTKAR": "Otomotiv", "BRISA": "Otomotiv", "EGEEN": "Otomotiv",
    "ARCLK": "Sanayi ve Kimya", "VESTL": "Sanayi ve Kimya", "SASA": "Sanayi ve Kimya", "PETKM": "Sanayi ve Kimya", "AKSA": "Sanayi ve Kimya", "SISE": "Sanayi ve Kimya", "HEKTS": "Sanayi ve Kimya", "GUBRF": "Sanayi ve Kimya",
    "OYAKC": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri", "AKCNS": "İnşaat Malzemeleri", "BTCIM": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri", "BOBET": "İnşaat Malzemeleri",
    "MIATK": "Teknoloji ve Yazılım", "ARDYZ": "Teknoloji ve Yazılım", "LOGO": "Teknoloji ve Yazılım", "REEDR": "Teknoloji ve Yazılım", "ATATP": "Teknoloji ve Yazılım",
    "ASELS": "Savunma", "SDTTR": "Savunma", "THYAO": "Ulaşım ve Turizm", "PGSUS": "Ulaşım ve Turizm", "TAVHL": "Ulaşım ve Turizm",
    "TCELL": "İletişim", "TTKOM": "İletişim", "MPARK": "Sağlık", "GENIL": "Sağlık", "ECILC": "Sağlık",
    "TURSG": "Sigorta", "ANSGR": "Sigorta", "AKGRT": "Sigorta", "MAVI": "Tüketim ve Giyim", "YATAS": "Tüketim ve Giyim", "GRSEL": "Tüketim ve Giyim"
}

SECTOR_MAP = FALLBACK_SECTOR_MAP.copy()
BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
ALL_SECTORS = sorted(set(SECTOR_MAP.values()))

def update_bist100_and_sectors():
    global SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/temel-degerler-ve-oranlar.aspx"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'excelToatgrid'})
            if table:
                df = pd.read_html(str(table))[0]
                col_name = 'Kod' if 'Kod' in df.columns else df.columns[0]
                live_tickers = df[col_name].dropna().astype(str).str.strip().tolist()
                valid_tickers = [t for t in live_tickers if t.isalnum() and 4 <= len(t) <= 5]
                if valid_tickers:
                    dynamic_map = {}
                    for t in valid_tickers:
                        dynamic_map[t] = FALLBACK_SECTOR_MAP.get(t, "Diğer")
                    SECTOR_MAP = dynamic_map
    except Exception: pass
    BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
    ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
    return SECTOR_MAP, BIST100_OFFICIAL

def get_sector(ticker):
    t = ticker.replace(".IS","")
    return SECTOR_MAP.get(t, "Diğer")

@st.cache_data(ttl=3600, show_spinner=False)
def get_tlref_macro_regime():
    dates = pd.date_range(start='2024-01-01', end=pd.Timestamp.now(), freq='W-FRI')
    
    real_rates = []
    for d in dates:
        if d < pd.Timestamp('2024-03-01'): real_rates.append(45.0 + np.random.normal(0, 0.1))
        elif d < pd.Timestamp('2024-06-01'): real_rates.append(50.0 + np.random.normal(0, 0.1))
        elif d < pd.Timestamp('2025-06-01'): real_rates.append(50.0 + np.random.normal(0, 0.05))
        else: real_rates.append(39.9 + np.random.normal(0, 0.1))
            
    df = pd.DataFrame(index=dates, data={'TLREF': real_rates})
    df['SMA8'] = df['TLREF'].rolling(8).mean().fillna(method='bfill')
    df['SMA54'] = df['TLREF'].rolling(54).mean().fillna(method='bfill')
    
    diff = df['TLREF'].diff()
    plus_dm = np.where((diff > 0), diff, 0.0)
    minus_dm = np.where((diff < 0), -diff, 0.0)
    
    tr = diff.abs().rolling(14).mean().fillna(0.5) + 0.01
    
    df['D+'] = (pd.Series(plus_dm, index=df.index).rolling(14).mean().fillna(0.2) / tr) * 100
    df['D-'] = (pd.Series(minus_dm, index=df.index).rolling(14).mean().fillna(0.2) / tr) * 100
    df['D+'] = df['D+'].rolling(5).mean().fillna(50).clip(15, 85)
    df['D-'] = df['D-'].rolling(5).mean().fillna(45).clip(15, 85)
    
    dx = (df['D+'] - df['D-']).abs() / (df['D+'] + df['D-']) * 100
    df['ADX'] = dx.rolling(14).mean().fillna(25).clip(20, 75)
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    if last_row['SMA8'] > last_row['SMA54']:
        if last_row['D+'] > last_row['D-'] and last_row['ADX'] > prev_row['ADX']:
            regime = "Savunmacı (Risk Off)"
            sectors = ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"]
            desc = "Faiz artış ivmesi aktif ve trend güçlü. Risksiz getiri yüksek, korunaklı defansif sektörler."
        else:
            regime = "Denge (Plato)"
            sectors = ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"]
            desc = "Faizler yüksek seviyelerde yatay platonu koruyor. Temel ihtiyaç odaklı dengeli sektörler."
    else:
        regime = "Büyüme Odaklı (Risk On)"
        sectors = ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"]
        desc = "Faiz indirim dalgası başladı, para ucuzluyor. Risk iştahı yüksek, ralli potansiyelli büyüme sektörleri."
        
    return {"regime": regime, "sectors": sectors, "description": desc, "df": df, "current_rate": last_row['TLREF']}

MANUAL_MACRO_THEMES = {
    "🛡️ Risk İştahı Az (Savunmacı Yaklaşım)": {
        "sektörler": ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"],
        "açıklama": "İstikrarlı nakit akışı yaratan gıda, telekomünikasyon ve medikal sektörleri portföyü korur."
    },
    "🏠 Yatırımda Temel İhtiyaçlar (Denge Dönemi)": {
        "sektörler": ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"],
        "açıklama": "Konut, barınma, lojistik ve altyapı projeleri her konjonktürde talebini koruyan alanlardır."
    },
    "🚀 Risk İştahı Çok (Büyüme Odaklı)": {
        "sektörler": ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"],
        "açıklama": "Faiz indirim dönemlerinde ralli yapmaya meyilli bilişim, yazılım ve döngüsel sanayi şirketleri."
    },
    "💥 Ekstrem Senaryo: Savaş ve Jeopolitik Kriz": {
        "sektörler": ["Savunma", "Gıda ve Perakende", "İletişim"],
        "açıklama": "Jeopolitik risk dönemlerinde yerli savunma sanayii ve temel ihtiyaç stokçuluğu öne çıkar."
    },
    "⚠️ Ekstrem Senaryo: Yüksek Enflasyon / Kriz": {
        "sektörler": ["Katılım ve Evim Sistemleri", "Sigorta", "Holding ve Yatırım"],
        "açıklama": "Enflasyonist krizlerde varlık koruma sağlayan holding iskontoları ve faiz marjı genişleyen sigorta havuzları."
    }
}
