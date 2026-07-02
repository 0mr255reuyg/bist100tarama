"""
sectors.py
BIST 100 bileşen listesi ve Makro Tema eşleştirmeleri.
NOT: get_live_bist100_and_sectors() şu an her zaman statik FALLBACK_SECTOR_MAP'i
döndürüyor — henüz gerçek bir canlı kaynağa bağlı değil. "canlı" isimlendirmesi
ileride gerçek bir çekim eklenirse anlamlı olsun diye bırakıldı; şu an yanıltıcı
olmaması için burada açıkça belirtiliyor.
"""
import requests
from bs4 import BeautifulSoup
import streamlit as st

# 1. YEDEK LİSTE (FALLBACK) - Sistem canlı veriye ulaşamazsa (ya da henüz canlı
#    kaynak bağlanmadığı için her zaman) burayı kullanır
FALLBACK_SECTOR_MAP = {
    "AKBNK": "Banka", "GARAN": "Banka", "ISCTR": "Banka", "YKBNK": "Banka",
    "HALKB": "Banka", "VAKBN": "Banka", "TSKB": "Banka", "SKBNK": "Banka",
    "ALBRK": "Katılım ve Evim Sistemleri", "KTLEV": "Katılım ve Evim Sistemleri",
    "ISMEN": "Katılım ve Evim Sistemleri", "INVES": "Katılım ve Evim Sistemleri",
    "EKGYO": "İnşaat ve GMYO", "TRGYO": "İnşaat ve GMYO", "ISGYO": "İnşaat ve GMYO",
    "ZRGYO": "İnşaat ve GMYO", "SNGYO": "İnşaat ve GMYO", "AKGYO": "İnşaat ve GMYO",
    "OZGYO": "İnşaat ve GMYO",
    "EREGL": "Çelik ve Metal", "KRDMD": "Çelik ve Metal", "BRSAN": "Çelik ve Metal", 
    "KCAER": "Çelik ve Metal", "CEMTS": "Çelik ve Metal",
    "AKSEN": "Enerji", "ENJSA": "Enerji", "ASTOR": "Enerji", "GESAN": "Enerji", 
    "EUPWR": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji", "SMRTG": "Enerji", 
    "ZOREN": "Enerji", "CANTE": "Enerji", "ODAS": "Enerji", "GWIND": "Enerji",
    "TUPRS": "Enerji",
    "BIMAS": "Gıda ve Perakende", "MGROS": "Gıda ve Perakende", "SOKM": "Gıda ve Perakende",
    "CCOLA": "Gıda ve Perakende", "AEFES": "Gıda ve Perakende", "ULKER": "Gıda ve Perakende",
    "TATGD": "Gıda ve Perakende", "TUKAS": "Gıda ve Perakende", "TABGD": "Gıda ve Perakende",
    "KLRHO": "Gıda ve Perakende",
    "KCHOL": "Holding ve Yatırım", "SAHOL": "Holding ve Yatırım", "ALARK": "Holding ve Yatırım",
    "DOHOL": "Holding ve Yatırım", "ENKAI": "Holding ve Yatırım", "TKFEN": "Holding ve Yatırım",
    "BERA": "Holding ve Yatırım", "AGHOL": "Holding ve Yatırım", "GSDHO": "Holding ve Yatırım",
    "AHSY": "Holding ve Yatırım",
    "FROTO": "Otomotiv", "TOASO": "Otomotiv", "DOAS": "Otomotiv", "TTRAK": "Otomotiv",
    "OTKAR": "Otomotiv", "BRISA": "Otomotiv", "EGEEN": "Otomotiv",
    "ARCLK": "Sanayi ve Kimya", "VESTL": "Sanayi ve Kimya", "SASA": "Sanayi ve Kimya",
    "PETKM": "Sanayi ve Kimya", "AKSA": "Sanayi ve Kimya", "SISE": "Sanayi ve Kimya",
    "HEKTS": "Sanayi ve Kimya", "GUBRF": "Sanayi ve Kimya",
    "OYAKC": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri", "AKCNS": "İnşaat Malzemeleri",
    "BTCIM": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri", "BOBET": "İnşaat Malzemeleri",
    "MIATK": "Teknoloji ve Yazılım", "ARDYZ": "Teknoloji ve Yazılım", "LOGO": "Teknoloji ve Yazılım",
    "REEDR": "Teknoloji ve Yazılım", "ATATP": "Teknoloji ve Yazılım",
    "ASELS": "Savunma", "SDTTR": "Savunma",
    "THYAO": "Ulaşım ve Turizm", "PGSUS": "Ulaşım ve Turizm", "TAVHL": "Ulaşım ve Turizm",
    "TCELL": "İletişim", "TTKOM": "İletişim",
    "MPARK": "Sağlık", "GENIL": "Sağlık", "ECILC": "Sağlık",
    "TURSG": "Sigorta", "ANSGR": "Sigorta", "AKGRT": "Sigorta",
    "MAVI": "Tüketim ve Giyim", "YATAS": "Tüketim ve Giyim", "GRSEL": "Tüketim ve Giyim",
}

# 2. DİNAMİK VERİ ÇEKME MOTORU
@st.cache_data(ttl=86400, show_spinner=False)
def get_live_bist100_and_sectors():
    """
    Canlı bir kaynağa bağlanmayı dener. Şu anda gerçek bir istek atılmıyor;
    her zaman güvenli/statik fallback listeyi döndürür (0 hata garantisi için).
    Gerçek canlı çekim (KAP/İş Yatırım/borsapy vb.) ileride buraya eklenebilir.
    """
    try:
        # TODO: Gerçek canlı kaynak entegre edilecekse örnek iskelet:
        # response = requests.get("URL", timeout=5)
        # soup = BeautifulSoup(response.text, "html.parser")
        # ... parse edilip live_map doldurulur ...
        live_map = FALLBACK_SECTOR_MAP.copy()
        return sorted(set(live_map.keys())), live_map
    except Exception:
        # Herhangi bir bağlantı veya parse hatasında sistem hissettirmeden yedeğe geçer
        return sorted(set(FALLBACK_SECTOR_MAP.keys())), FALLBACK_SECTOR_MAP

# Aktif listeler bu fonksiyondan beslenir
BIST100_OFFICIAL, SECTOR_MAP = get_live_bist100_and_sectors()

def get_sector(ticker):
    t = ticker.replace(".IS", "")
    return SECTOR_MAP.get(t, "Diğer")

# 3. MAKRO TEMALAR VE 3'LÜ REJİM ALTYAPISI
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

ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
