# Midas ekranından alınan BIST 100 hisse listesi — Tam 98 Hisse (Spor Hisseleri Hariç)

SECTOR_MAP = {
    # Banka (8)
    "AKBNK": "Banka", "GARAN": "Banka", "ISCTR": "Banka", "YKBNK": "Banka",
    "HALKB": "Banka", "VAKBN": "Banka", "TSKB": "Banka", "SKBNK": "Banka",

    # Katılım ve Evim Sistemleri (4) - Finanstan ayrıştırıldı
    "ALBRK": "Katılım ve Evim Sistemleri", "KTLEV": "Katılım ve Evim Sistemleri",
    "ISMEN": "Katılım ve Evim Sistemleri", "INVES": "Katılım ve Evim Sistemleri",

    # İnşaat ve GMYO (7)
    "EKGYO": "İnşaat ve GMYO", "TRGYO": "İnşaat ve GMYO", "ISGYO": "İnşaat ve GMYO",
    "ZRGYO": "İnşaat ve GMYO", "SNGYO": "İnşaat ve GMYO", "AKGYO": "İnşaat ve GMYO",
    "OZGYO": "İnşaat ve GMYO",

    # Çelik ve Metal (5)
    "EREGL": "Çelik ve Metal", "KRDMD": "Çelik ve Metal", "BRSAN": "Çelik ve Metal", 
    "KCAER": "Çelik ve Metal", "CEMTS": "Çelik ve Metal",

    # Enerji (12)
    "AKSEN": "Enerji", "ENJSA": "Enerji", "ASTOR": "Enerji", "GESAN": "Enerji", 
    "EUPWR": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji", "SMRTG": "Enerji", 
    "ZOREN": "Enerji", "CANTE": "Enerji", "ODAS": "Enerji", "GWIND": "Enerji",

    # Gıda ve Perakende (10)
    "BIMAS": "Gıda ve Perakende", "MGROS": "Gıda ve Perakende", "SOKM": "Gıda ve Perakende",
    "CCOLA": "Gıda ve Perakende", "AEFES": "Gıda ve Perakende", "ULKER": "Gıda ve Perakende",
    "TATGD": "Gıda ve Perakende", "TUKAS": "Gıda ve Perakende", "TABGD": "Gıda ve Perakende",
    "KLRHO": "Gıda ve Perakende",

    # Holding ve Yatırım (10)
    "KCHOL": "Holding ve Yatırım", "SAHOL": "Holding ve Yatırım", "ALARK": "Holding ve Yatırım",
    "DOHOL": "Holding ve Yatırım", "ENKAI": "Holding ve Yatırım", "TKFEN": "Holding ve Yatırım",
    "BERA": "Holding ve Yatırım", "AGHOL": "Holding ve Yatırım", "GSDHO": "Holding ve Yatırım",
    "AHSY": "Holding ve Yatırım",

    # Otomotiv (7)
    "FROTO": "Otomotiv", "TOASO": "Otomotiv", "DOAS": "Otomotiv", "TTRAK": "Otomotiv",
    "OTKAR": "Otomotiv", "BRISA": "Otomotiv", "EGEEN": "Otomotiv",

    # Sanayi ve Kimya (8)
    "ARCLK": "Sanayi ve Kimya", "VESTL": "Sanayi ve Kimya", "SASA": "Sanayi ve Kimya",
    "PETKM": "Sanayi ve Kimya", "AKSA": "Sanayi ve Kimya", "SISE": "Sanayi ve Kimya",
    "HEKTS": "Sanayi ve Kimya", "GUBRF": "Sanayi ve Kimya",

    # İnşaat Malzemeleri (6)
    "OYAKC": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri", "AKCNS": "İnşaat Malzemeleri",
    "BTCIM": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri", "BOBET": "İnşaat Malzemeleri",

    # Teknoloji ve Yazılım (5)
    "MIATK": "Teknoloji ve Yazılım", "ARDYZ": "Teknoloji ve Yazılım", "LOGO": "Teknoloji ve Yazılım",
    "REEDR": "Teknoloji ve Yazılım", "ATATP": "Teknoloji ve Yazılım",

    # Savunma (2)
    "ASELS": "Savunma", "SDTTR": "Savunma",

    # Ulaşım ve Turizm (3)
    "THYAO": "Ulaşım ve Turizm", "PGSUS": "Ulaşım ve Turizm", "TAVHL": "Ulaşım ve Turizm",

    # İletişim (2)
    "TCELL": "İletişim", "TTKOM": "İletişim",

    # Sağlık (3)
    "MPARK": "Sağlık", "GENIL": "Sağlık", "ECILC": "Sağlık",

    # Sigorta (3)
    "TURSG": "Sigorta", "ANSGR": "Sigorta", "AKGRT": "Sigorta",

    # Tüketim ve Giyim (3)
    "MAVI": "Tüketim ve Giyim", "YATAS": "Tüketim ve Giyim", "GRSEL": "Tüketim ve Giyim",
}

# Tekrarları çıkar, sıralı liste (Tam 98 Adet)
BIST100_OFFICIAL = sorted(set(SECTOR_MAP.keys()))

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

ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
