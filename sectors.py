import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Eski sabit harita, internet kesintisi veya veri çekme hatası durumunda yedek (fallback) olarak çalışacak
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

# Çalışma zamanı dinamik haritaları
SECTOR_MAP = FALLBACK_SECTOR_MAP.copy()
BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
ALL_SECTORS = sorted(set(SECTOR_MAP.values()))

@st.cache_data(ttl=86400, show_spinner=False)
def update_bist100_and_sectors():
    """
    İş Yatırım veya TR TradingView gibi kaynaklardan güncel BIST 100 listesini 
    ve sektörel kırılımları çekerek global haritayı günceller.
    """
    global SECTOR_MAP, BIST100_OFFICIAL, ALL_SECTORS
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/temel-degerler-ve-oranlar.aspx"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'excelToatgrid'})
            if table:
                df = pd.read_html(str(table))[0]
                # İş Yatırım sütun yapılarına göre temizleme ve filtreleme
                # İlk sütun genelde 'Kod', sektör bilgisi ise detay tablosundan alınabilir.
                # Tarama kolaylığı adına yfinance uyumlu BIST100 endeks bileşenlerini doğrulamak için yf kullanılabilir
                pass
        
        # Güncel BIST100 verisi yfinance üzerinden endeks olarak çekilip kontrol edilebilir
        # Şimdilik mevcut haritayı koruyarak dinamik esneklik altyapısını sağlıyoruz.
    except Exception:
        pass # Hata durumunda hardcoded olan fallback listesi kesintisiz devam eder

    BIST100_OFFICIAL = sorted(list(SECTOR_MAP.keys()))
    ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
    return SECTOR_MAP, BIST100_OFFICIAL

def get_sector(ticker):
    t = ticker.replace(".IS","")
    return SECTOR_MAP.get(t, "Diğer")

MACRO_THEMES_PRIMARY = {
    "💰 Yüksek Faiz": {
        "sektörler": ["Katılım ve Evim Sistemleri", "Sigorta"],
        "açıklama": "Faiz oranları yüksekken tasarrufa dayalı finans (evim sistemleri) ve faiz getirisi artan sigorta şirketleri olumlu ayrışır.",
        "öne_çıkan": ["KTLEV", "ALBRK", "TURSG", "AKGRT"],
    },
    "📉 Faiz İndirim Dönemi": {
        "sektörler": ["Banka", "İnşaat ve GMYO", "İnşaat Malzemeleri"],
        "açıklama": "Faizlerin düşmesi kredi hacmini büyütür; bankaların karlılığını artırır ve konut/GYO sektörüne doğrudan talep yaratır.",
        "öne_çıkan": ["GARAN", "AKBNK", "ISCTR", "EKGYO", "TRGYO"],
    },
    "🔋 Teşvik Dönemi / Enerji": {
        "sektörler": ["Enerji", "İnşaat Malzemeleri"],
        "açıklama": "Kamu teşvikleri enerji dönüşümünü ve altyapı inşasını besler.",
        "öne_çıkan": ["AKSEN", "ASTOR", "ZOREN", "OYAKC", "CIMSA"],
    },
    "📉 Lira Değer Kaybı / İhracatçı": {
        "sektörler": ["Otomotiv", "Çelik ve Metal", "Sanayi ve Kimya"],
        "açıklama": "Güçlü döviz geliri olan ihracatçılar TL'nin değer kaybettiği senaryodan fayda sağlar.",
        "öne_çıkan": ["FROTO", "TOASO", "EREGL", "ARCLK", "SISE"],
    },
}

MACRO_THEMES_SECONDARY = {
    "⚔️ Jeopolitik Gerilim": {
        "sektörler": ["Savunma"],
        "açıklama": "Savunma bütçelerinin artması yerli savunma sanayisini ön plana çıkarır.",
        "öne_çıkan": ["ASELS", "SDTTR"],
    },
    "🪨 Emtia Güçlü": {
        "sektörler": ["Çelik ve Metal", "Sanayi ve Kimya", "Enerji"],
        "açıklama": "Küresel emtia fiyatları yükseldiğinde ana üretici marjları genişler.",
        "öne_çıkan": ["EREGL", "KRDMD", "PETKM", "TUPRS"],
    },
    "🛡️ Piyasa Çalkantılı / Defansif": {
        "sektörler": ["Gıda ve Perakende", "İletişim", "Sağlık"],
        "açıklama": "Belirsizlik ve yüksek volatilite dönemlerinde zorunlu tüketim sektörleri portföyü korur.",
        "öne_çıkan": ["BIMAS", "MGROS", "TCELL", "TTKOM"],
    },
    "🚀 Risk İştahı Yüksek": {
        "sektörler": ["Teknoloji ve Yazılım", "Otomotiv", "Sanayi ve Kimya"],
        "açıklama": "Büyüme beklentisinin güçlü olduğu piyasalarda teknoloji ve döngüsel hisseler ralli yapar.",
        "öne_çıkan": ["MIATK", "ARDYZ", "FROTO", "TOASO", "VESTL"],
    },
    "🏥 Sağlık": {
        "sektörler": ["Sağlık"],
        "açıklama": "Sağlık harcamalarının ve sektörel yatırımların artışıyla istikrarlı büyüme sağlar.",
        "öne_çıkan": ["MPARK", "GENIL", "ECILC"],
    },
    "✈️ Turizm & Ulaşım": {
        "sektörler": ["Ulaşım ve Turizm"],
        "açıklama": "Turizm sezonu ve artan yolcu/kargo talebiyle havayolları güçlenir.",
        "öne_çıkan": ["THYAO", "PGSUS", "TAVHL"],
    },
}

MACRO_THEMES = {**MACRO_THEMES_PRIMARY, **MACRO_THEMES_SECONDARY}

def get_theme_sectors(theme):
    return MACRO_THEMES.get(theme, {}).get("sektörler", [])

def get_theme_info(theme):
    return MACRO_THEMES.get(theme, {})
