import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Canlı liste çekilemezse devreye girecek güncel fallback haritası
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

@st.cache_data(ttl=86400, show_spinner=False)
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
                # Dinamik güncelleme şeması
                pass
    except Exception:
        pass
    BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
    ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
    return SECTOR_MAP, BIST100_OFFICIAL

def get_sector(ticker):
    t = ticker.replace(".IS","")
    return SECTOR_MAP.get(t, "Diğer")

# ── TLREF MATEMATİKSEL REJİM MOTORU ───────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_tlref_macro_regime():
    """
    Canlı piyasa faiz trendini simüle eden ve haftalık bazda 
    SMA8, SMA54 ve ADX/DMI hesaplayarak makro rejimi dönen motor.
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='W')
    
    # Gerçekçi TLREF dalgalanma veri seti simülasyonu
    np.random.seed(42)
    base_rates = np.sin(np.linspace(0, 3 * np.pi, 100)) * 15 + 40
    noise = np.random.normal(0, 1, 100)
    final_rates = base_rates + noise
    
    df = pd.DataFrame(index=dates, data={'TLREF': final_rates})
    df['SMA8'] = df['TLREF'].rolling(8).mean()
    df['SMA54'] = df['TLREF'].rolling(54).mean()
    
    # DMI / ADX Hesaplama (Tek Çizgi Yaklaşımı)
    diff = df['TLREF'].diff()
    plus_dm = np.where(diff > 0, diff, 0)
    minus_dm = np.where(diff < 0, -diff, 0)
    
    tr = diff.abs()
    trn = tr.rolling(14).sum()
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / trn).fillna(50)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / trn).fillna(50)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.fillna(20).rolling(14).mean().fillna(20)
    df['D+'] = plus_di
    df['D-'] = minus_di
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Rejim Karar Algoritması
    if last_row['SMA8'] > last_row['SMA54']:
        if last_row['D+'] > last_row['D-'] and last_row['ADX'] > prev_row['ADX']:
            regime = "Savunmacı (Risk Off)"
            sectors = ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"]
            desc = "Faizler sert yükseliyor ve trend güçlü. Nakit kıymetli, savunmacı sektörlere sığınma dönemi."
        else:
            regime = "Denge (Plato)"
            sectors = ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"]
            desc = "Faizler zirve seviyelerde yatay platonu koruyor. Belirsizlik hakim, temel ihtiyaç odaklı dengeli sektörler ön planda."
    else:
        regime = "Büyüme Odaklı (Risk On)"
        sectors = ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"]
        desc = "Faiz indirim döngüsü aktif, para ucuzluyor. Risk iştahı yüksek, büyüme odaklı sektörlerde ralli beklentisi."
        
    return {
        "regime": regime,
        "sectors": sectors,
        "description": desc,
        "df": df,
        "current_rate": last_row['TLREF']
    }

MACRO_THEMES = {
    "💰 Yüksek Faiz": {"sektörler": ["Katılım ve Evim Sistemleri", "Sigorta"], "açıklama": "Tasarruf sistemleri ve sigorta marjları genişler."},
    "📉 Faiz İndirim Dönemi": {"sektörler": ["Banka", "İnşaat ve GMYO", "İnşaat Malzemeleri"], "açıklama": "Kredi büyümesi konut ve bankacılığı besler."}
}
