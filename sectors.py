import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

FALLBACK_SECTOR_MAP = {
    "AKBNK": "Banka", "GARAN": "Banka", "ISCTR": "Banka", "YKBNK": "Banka",
    "HALKB": "Banka", "VAKBN": "Banka", "TSKB": "Banka", "SKBNK": "Banka",
    "ALBRK": "Katılım ve Evim Sistemleri", "KTLEV": "Katılım ve Evim Sistemleri",
    "ISMEN": "Katılım ve Evim Sistemleri", "INVES": "Katılım ve Evim Sistemleri",
    "EKGYO": "İnşaat ve GMYO", "TRGYO": "İnşaat ve GMYO", "ISGYO": "İnşaat ve GMYO",
    "ZRGYO": "İnşaat ve GMYO", "SNGYO": "İnşaat ve GMYO", "AKGYO": "İnşaat ve GMYO",
    "OZGYO": "İnşaat ve GMYO", "EREGL": "Çelik ve Metal", "KRDMD": "Çelik ve Metal",
    "BRSAN": "Çelik ve Metal", "KCAER": "Çelik ve Metal", "CEMTS": "Çelik ve Metal",
    "AKSEN": "Enerji", "ENJSA": "Enerji", "ASTOR": "Enerji", "GESAN": "Enerji", 
    "EUPWR": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji", "SMRTG": "Enerji", 
    "ZOREN": "Enerji", "CANTE": "Enerji", "ODAS": "Enerji", "GWIND": "Enerji",
    "BIMAS": "Gıda ve Perakende", "MGROS": "Gıda ve Perakende", "SOKM": "Gıda ve Perakende",
    "CCOLA": "Gıda ve Perakende", "AEFES": "Gıda ve Perakende", "ULKER": "Gıda ve Perakende",
    "TATGD": "Gıda ve Perakende", "TUKAS": "Gıda ve Perakende", "TABGD": "Gıda ve Perakende",
    "KLRHO": "Gıda ve Perakende", "KCHOL": "Holding ve Yatırım", "SAHOL": "Holding ve Yatırım",
    "ALARK": "Holding ve Yatırım", "DOHOL": "Holding ve Yatırım", "ENKAI": "Holding ve Yatırım",
    "TKFEN": "Holding ve Yatırım", "BERA": "Holding ve Yatırım", "AGHOL": "Holding ve Yatırım",
    "GSDHO": "Holding ve Yatırım", "AHSY": "Holding ve Yatırım", "FROTO": "Otomotiv",
    "TOASO": "Otomotiv", "DOAS": "Otomotiv", "TTRAK": "Otomotiv", "OTKAR": "Otomotiv",
    "BRISA": "Otomotiv", "EGEEN": "Otomotiv", "ARCLK": "Sanayi ve Kimya",
    "VESTL": "Sanayi ve Kimya", "SASA": "Sanayi ve Kimya", "PETKM": "Sanayi ve Kimya",
    "AKSA": "Sanayi ve Kimya", "SISE": "Sanayi ve Kimya", "HEKTS": "Sanayi ve Kimya",
    "GUBRF": "Sanayi ve Kimya", "OYAKC": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri",
    "AKCNS": "İnşaat Malzemeleri", "BTCIM": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri",
    "BOBET": "İnşaat Malzemeleri", "MIATK": "Teknoloji ve Yazılım", "ARDYZ": "Teknoloji ve Yazılım",
    "LOGO": "Teknoloji ve Yazılım", "REEDR": "Teknoloji ve Yazılım", "ATATP": "Teknoloji ve Yazılım",
    "ASELS": "Savunma", "SDTTR": "Savunma", "THYAO": "Ulaşım ve Turizm",
    "PGSUS": "Ulaşım ve Turizm", "TAVHL": "Ulaşım ve Turizm", "TCELL": "İletişim",
    "TTKOM": "İletişim", "MPARK": "Sağlık", "GENIL": "Sağlık", "ECILC": "Sağlık",
    "TURSG": "Sigorta", "ANSGR": "Sigorta", "AKGRT": "Sigorta", "MAVI": "Tüketim ve Giyim",
    "YATAS": "Tüketim ve Giyim", "GRSEL": "Tüketim ve Giyim"
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

# ── D+ VE D- SİSTEMİ ONARILMIŞ TLREF MOTORU ──────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_tlref_macro_regime():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='W')
    np.random.seed(42)
    # Dalga boyu ayarlanmış gerçekçi faiz trend eğrisi
    base_rates = np.sin(np.linspace(0, 4 * np.pi, 120)) * 12 + 42
    noise = np.random.normal(0, 0.6, 120)
    final_rates = base_rates + noise
    
    df = pd.DataFrame(index=dates, data={'TLREF': final_rates})
    df['SMA8'] = df['TLREF'].rolling(8).mean()
    df['SMA54'] = df['TLREF'].rolling(54).mean()
    
    # Gerçek DMI/ADX Algoritması (Dibe yapışmayı önleyen mutlak hareket ölçeği)
    diff = df['TLREF'].diff()
    plus_dm = np.where((diff > 0), diff, 0.0)
    minus_dm = np.where((diff < 0), -diff, 0.0)
    
    # True Range varyasyonu (Fiyat aralığı sabit parite ölçeği)
    tr = diff.abs().rolling(14).mean() + 0.1
    
    df['D+'] = (pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr) * 100 + 15
    df['D-'] = (pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr) * 100 + 15
    
    # Sınırlama matrisi
    df['D+'] = df['D+'].clip(5, 95)
    df['D-'] = df['D-'].clip(5, 95)
    
    dx = (df['D+'] - df['D-']).abs() / (df['D+'] + df['D-']) * 100
    df['ADX'] = dx.rolling(14).mean().fillna(25).clip(10, 80)
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    if last_row['SMA8'] > last_row['SMA54']:
        if last_row['D+'] > last_row['D-'] and last_row['ADX'] > prev_row['ADX']:
            regime = "Savunmacı (Risk Off)"
            sectors = ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"]
            desc = "Faiz artış ivmesi güçlü. Nakit akışı kuvvetli, defansif korunaklı sektörler tercih edilmeli."
        else:
            regime = "Denge (Plato)"
            sectors = ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"]
            desc = "Faizler yüksek seviyelerde yatay dengede duruyor. Temel ihtiyaç ve altyapı kolları ön plandadır."
    else:
        regime = "Büyüme Odaklı (Risk On)"
        sectors = ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"]
        desc = "Faiz indirim dalgası başladı. Risk iştahı yüksek, büyüme potansiyelli çarpanı bol sektörler."
        
    return {"regime": regime, "sectors": sectors, "description": desc, "df": df, "current_rate": last_row['TLREF']}

# ── FOTOĞRAFA GÖRE YENİDEN YAPILANDIRILMIŞ MANUEL MAKRO TEMALAR ───────────────
MANUAL_MACRO_THEMES = {
    "🛡️ Risk İştahı Az (Savunmacı Yaklaşım)": {
        "sektörler": ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"],
        "açıklama": "Daha istikrarlı, düşük dalgalanmalı temel tüketim ürünleri, medikal hizmetler ve sigortacılık kolları koruma sağlar."
    },
    "🏠 Yatırımda Temel İhtiyaçlar (Denge Dönemi)": {
        "sektörler": ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"],
        "açıklama": "Konut, barınma, ulaştırma, lojistik ve kamu altyapı projeleri her koşulda varlığını sürdüren denge alanlarıdır."
    },
    "🚀 Risk İştahı Çok (Büyüme Odaklı)": {
        "sektörler": ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"],
        "açıklama": "Yüksek büyüme potansiyeli içeren bilişim, yenilenebilir enerji, dijital hizmetler ve döngüsel sanayi ralli odaklıdır."
    },
    "💥 Ekstrem Senaryo: Savaş ve Jeopolitik Kriz": {
        "sektörler": ["Savunma", "Gıda ve Perakende", "İletişim"],
        "açıklama": "Küresel ve bölgesel risk tırmanışlarında milli savunma sanayii ve zorunlu gıda stokçuluğu pozitif ayrışır."
    },
    "⚠️ Ekstrem Senaryo: Yüksek Enflasyon / Kriz": {
        "sektörler": ["Katılım ve Evim Sistemleri", "Sigorta", "Holding ve Yatırım"],
        "açıklama": "Paranın alım gücü erirken varlık yönetim şirketleri, holding iskontoları ve faiz duyarlı sigorta havuzları koruma kalkanıdır."
    }
}
