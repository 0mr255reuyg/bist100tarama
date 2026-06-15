# BIST 100 Sektör Haritası
SECTOR_MAP = {
    # Bankacılık & Finans
    "AKBNK": "Bankacılık", "GARAN": "Bankacılık", "HALKB": "Bankacılık",
    "ISCTR": "Bankacılık", "SKBNK": "Bankacılık", "TSKB": "Bankacılık",
    "VAKBN": "Bankacılık", "YKBNK": "Bankacılık", "ALBRK": "Bankacılık",
    "KLGYO": "GYO", "EKGYO": "GYO", "ISGYO": "GYO", "ALGYO": "GYO",
    "AGESA": "Sigorta & Finans", "AKGRT": "Sigorta & Finans",
    "ANSGR": "Sigorta & Finans", "SAHOL": "Holding", "KCHOL": "Holding",
    "DOHOL": "Holding", "AGHOL": "Holding",

    # Sanayi & Üretim
    "ARCLK": "Beyaz Eşya", "VESTL": "Beyaz Eşya",
    "FROTO": "Otomotiv", "TOASO": "Otomotiv", "OTKAR": "Otomotiv",
    "TTRAK": "Otomotiv", "ASUZU": "Otomotiv",
    "EREGL": "Demir Çelik", "KRDMD": "Demir Çelik",
    "BRSAN": "Demir Çelik", "CEMTS": "Çimento", "CIMSA": "Çimento",
    "BSOKE": "Çimento", "BTCIM": "Çimento", "BUCIM": "Çimento",
    "CMENT": "Çimento",

    # Enerji
    "AKSEN": "Enerji", "ZOREN": "Enerji", "ODAS": "Enerji",
    "BIOEN": "Enerji", "AYEN": "Enerji", "ASTOR": "Enerji",
    "ENKAI": "Enerji & İnşaat", "TUPRS": "Rafineri", "PETKM": "Petrokimya",
    "PTOFS": "Petrol Dağıtım",

    # Madencilik
    "KOZAA": "Madencilik", "KOZAL": "Madencilik", "CONSE": "Madencilik",
    "IPEKE": "Madencilik",

    # Teknoloji
    "ASELS": "Savunma & Teknoloji", "LOGO": "Teknoloji", "NETAS": "Teknoloji",
    "ARDYZ": "Teknoloji", "ARENA": "Teknoloji",

    # Gıda & İçecek
    "AEFES": "Gıda & İçecek", "CCOLA": "Gıda & İçecek", "ULKER": "Gıda & İçecek",
    "TATGD": "Gıda & İçecek", "PNSUT": "Gıda & İçecek",

    # Perakende
    "BIMAS": "Perakende", "MGROS": "Perakende", "SOKM": "Perakende",
    "BIZIM": "Perakende", "MAVI": "Perakende", "CRFSA": "Perakende",

    # Havacılık & Ulaşım
    "THYAO": "Havacılık", "PGSUS": "Havacılık", "TAVHL": "Havacılık",
    "CLEBI": "Havacılık",

    # Telekom
    "TCELL": "Telekom", "TTKOM": "Telekom",

    # Kimya & İlaç
    "GUBRF": "Kimya & Gübre", "SASA": "Kimya", "ALKIM": "Kimya",
    "DEVA": "İlaç", "SELEC": "Kimya",

    # İnşaat & GYO
    "EKGYO": "GYO", "KONTR": "İnşaat",

    # Diğer
    "KARSN": "Makine", "TKFEN": "İnşaat & Enerji",
    "BAGFS": "Kimya & Gübre", "AKCNS": "Çimento",
    "AKSA":  "Kimya", "SISE":  "Cam & Kimya",
    "ANACM": "Cam & Kimya", "TRKCM": "Cam & Kimya",
    "ISGSY": "Sigorta & Finans", "ALFAS": "Sigorta & Finans",
    "BRYAT": "Turizm", "BNTAS": "Lojistik",
    "BOBET": "Gıda & İçecek", "BRISA": "Otomotiv",
    "BIRGY": "Enerji", "DOAS": "Otomotiv",
    "ECILC": "Holding", "EGEEN": "Tekstil",
    "KORDS": "Tekstil", "CMBTN": "Çimento",
}

# Makro tema → sektör grupları
MACRO_THEMES = {
    "📉 Faiz Düşüyor":       ["Bankacılık", "GYO", "Holding", "Sigorta & Finans"],
    "🪨 Emtia Güçlü":        ["Madencilik", "Enerji", "Rafineri", "Petrokimya", "Kimya & Gübre"],
    "🌊 Piyasa Çalkantılı":  ["Gıda & İçecek", "Perakende", "Telekom"],
    "🚀 Risk İştahı Yüksek": ["Savunma & Teknoloji", "Teknoloji", "Otomotiv", "Beyaz Eşya"],
}

def get_sector(ticker):
    t = ticker.replace(".IS", "")
    return SECTOR_MAP.get(t, "Diğer")

def get_theme_sectors(theme):
    return MACRO_THEMES.get(theme, [])
