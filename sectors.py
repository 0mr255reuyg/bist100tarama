import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Canlı liste çekilemezse veya bağlantı koparsa devreye girecek 1 Temmuz sonrası güncel yedek harita
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

# Uygulama çalışma zamanı haritaları
SECTOR_MAP = FALLBACK_SECTOR_MAP.copy()
BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
ALL_SECTORS = sorted(set(SECTOR_MAP.values()))

@st.cache_data(ttl=86400, show_spinner=False)
def update_bist100_and_sectors():
    """
    İş Yatırım Temel Değerler tablosundan aktif işlem gören tüm güncel BIST 100 
    hisselerini dinamik olarak çeker ve listeyi günceller.
    """
    global SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/temel-degerler-ve-oranlar.aspx"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'excelToatgrid'})
            if table:
                df = pd.read_html(str(table))[0]
                # İlk sütun veya 'Kod' başlıklı sütun üzerinden hisse kodlarını temizle
                col_name = 'Kod' if 'Kod' in df.columns else df.columns[0]
                live_tickers = df[col_name].dropna().astype(str).str.strip().tolist()
                
                # Sadece geçerli kısa kodları (harf ve sayıdan oluşan) filtrele
                valid_tickers = [t for t in live_tickers if t.isalnum() and 4 <= len(t) <= 5]
                
                if valid_tickers:
                    dynamic_map = {}
                    for t in valid_tickers:
                        # Yeni bir hisse listeye girdiyse 'Diğer' olarak ata, mevcutsa sektörünü koru
                        dynamic_map[t] = FALLBACK_SECTOR_MAP.get(t, "Diğer")
                    SECTOR_MAP = dynamic_map
    except Exception:
        # Bağlantı hatası durumunda mevcut yedek harita kesintisiz devam eder
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
    Haftalık bazda paranın maliyetini simüle eden ve hareketli ortalamalar ile
    trend şiddetini hesaplayarak dinamik rejim kararı veren ana motor.
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='W')
    
    # Gerçekçi ve döngüsel faiz dalgalanması modeli (Kuant Simülasyonu)
    np.random.seed(42)
    base_rates = np.sin(np.linspace(0, 3 * np.pi, 100)) * 15 + 40
    noise = np.random.normal(0, 1, 100)
    final_rates = base_rates + noise
    
    df = pd.DataFrame(index=dates, data={'TLREF': final_rates})
    df['SMA8'] = df['TLREF'].rolling(8).mean()
    df['SMA54'] = df['TLREF'].rolling(54).mean()
    
    # DMI / ADX İndikatör Hesaplamaları
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
    
    # Kurguladığımız 3'lü Risk İştahı ve Rejim Algoritması
    if last_row['SMA8'] > last_row['SMA54']:
        if last_row['D+'] > last_row['D-'] and last_row['ADX'] > prev_row['ADX']:
            regime = "Savunmacı (Risk Off)"
            sectors = ["Gıda ve Perakende", "Sigorta", "Sağlık", "İletişim"]
            desc = "Faizler güçlü bir momentumla yükseliyor. Risksiz getiri cazip, piyasada defansif ve nakit akışı güçlü sektörler koruma sağlar."
        else:
            regime = "Denge (Plato)"
            sectors = ["İnşaat ve GMYO", "İnşaat Malzemeleri", "Ulaşım ve Turizm", "Holding ve Yatırım"]
            desc = "Faizler yüksek seviyelerde yatay bir platoda sabitlendi. Belirsizlik hakim, temel ihtiyaç ve altyapı odaklı dengeli sektörler ön planda."
    else:
        regime = "Büyüme Odaklı (Risk On)"
        sectors = ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya", "Enerji"]
        desc = "Faiz indirim döngüsü aktif, borçlanma maliyetleri düşüyor. Piyasanın risk iştahı yüksek, büyüme odaklı şirketlerde güçlü ralli beklentisi."
        
    return {
        "regime": regime,
        "sectors": sectors,
        "description": desc,
        "df": df,
        "current_rate": last_row['TLREF']
    }
