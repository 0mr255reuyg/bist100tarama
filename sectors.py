# ── TAM BIST HİSSE & SEKTÖR HARİTASI ────────────────────────────────────────
# Kaynak: Kullanıcının hisse tarayıcısından alınan güncel liste
# Sektör isimleri orijinal uygulama sınıflandırmasına göre düzenlendi

SECTOR_MAP = {
    # Çelik
    "EREGL": "Çelik", "KRDMD": "Çelik", "BRSAN": "Çelik", "CVKMD": "Çelik",
    "ISDMR": "Çelik", "BRYAT": "Çelik",

    # Enerji
    "AKSEN": "Enerji", "ENJSA": "Enerji", "CANTE": "Enerji", "ENERY": "Enerji",
    "IZENR": "Enerji", "TRENJ": "Enerji", "ZOREN": "Enerji", "MAGEN": "Enerji",
    "ODAS":  "Enerji", "KONTR": "Enerji", "ASTOR": "Enerji", "AKFYE": "Enerji",
    "AYDEM": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji",
    # Not: KCHOL ve TUPRS listedeki enerji sınıfından çıkarıldı (holding/rafineri)

    # Finans (Bankacılık + Finansal Hizmetler)
    "DSTKF": "Finans", "SKBNK": "Finans", "TSKB": "Finans", "GARAN": "Finans",
    "ISCTR": "Finans", "YKBNK": "Finans", "SAHOL": "Finans", "ISMEN": "Finans",
    "AKBNK": "Finans", "KTLEV": "Finans", "TUREX": "Finans", "HALKB": "Finans",
    "VAKBN": "Finans", "ALBRK": "Finans", "TRGYO": "Finans",

    # Gıda ve İçecek
    "CCOLA": "Gıda ve İçecek", "AGHOL": "Gıda ve İçecek", "TUKAS": "Gıda ve İçecek",
    "AEFES": "Gıda ve İçecek", "EFOR":  "Gıda ve İçecek", "TABGD": "Gıda ve İçecek",
    "KLRHO": "Gıda ve İçecek", "BALSU": "Gıda ve İçecek", "BIMAS": "Gıda ve İçecek",
    "ULKER": "Gıda ve İçecek", "OBAMS": "Gıda ve İçecek", "SOKM":  "Gıda ve İçecek",
    "MGROS": "Gıda ve İçecek", "TATGD": "Gıda ve İçecek",

    # Ham Madde (Kimya, Petrokimya, Metal)
    "PETKM": "Ham Madde", "GUBRF": "Ham Madde", "SASA":  "Ham Madde",
    "TRMET": "Ham Madde", "AKSA":  "Ham Madde", "HEKTS": "Ham Madde",
    "TRALT": "Ham Madde", "TUPRS": "Ham Madde", "ECILC": "Ham Madde",

    # Hizmet
    "GRTHO": "Hizmet",

    # İletişim
    "TCELL": "İletişim", "TTKOM": "İletişim",

    # İnşaat Malzemeleri (Çimento, Cam, Yapı)
    "EUREN": "İnşaat Malzemeleri", "OYAKC": "İnşaat Malzemeleri",
    "QUAGR": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri",
    "BTCIM": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri",
    "SISE":  "İnşaat Malzemeleri", "AKCNS": "İnşaat Malzemeleri",

    # İnşaat ve GMYO (GYO + İnşaat Şirketleri)
    "KUYAS": "İnşaat ve GMYO", "EKGYO": "İnşaat ve GMYO", "PSGYO": "İnşaat ve GMYO",
    "DAPGM": "İnşaat ve GMYO", "TKFEN": "İnşaat ve GMYO", "GLRMK": "İnşaat ve GMYO",
    "ALARK": "İnşaat ve GMYO", "RALYH": "İnşaat ve GMYO", "ENKAI": "İnşaat ve GMYO",
    "TRGYO": "İnşaat ve GMYO", "SNGYO": "İnşaat ve GMYO", "AKGYO": "İnşaat ve GMYO",
    "OZGYO": "İnşaat ve GMYO", "VKGYO": "İnşaat ve GMYO",

    # Otomotiv
    "FROTO": "Otomotiv", "OTKAR": "Otomotiv", "TOASO": "Otomotiv",
    "DOAS":  "Otomotiv", "TTRAK": "Otomotiv", "BRISA": "Otomotiv",
    "FMTAS": "Otomotiv",

    # Sağlık
    "ECILC": "Sağlık", "GENIL": "Sağlık", "MPARK": "Sağlık",

    # Sanayi (Beyaz Eşya, Elektronik, Üretim)
    "EUPWR": "Sanayi", "ARCLK": "Sanayi", "VESTL": "Sanayi",
    "SARKY": "Sanayi", "ASTOR": "Sanayi",

    # Savunma
    "ASELS": "Savunma", "ALTNY": "Savunma", "MIATK": "Savunma",
    "SDTTR": "Savunma", "ATATP": "Savunma",

    # Sigorta
    "ANSGR": "Sigorta", "TURSG": "Sigorta", "AKGRT": "Sigorta",

    # Teknoloji
    "PATEK": "Teknoloji", "GESAN": "Teknoloji", "REEDR": "Teknoloji",
    "ARDYZ": "Teknoloji", "LOGO":  "Teknoloji", "NETAS": "Teknoloji",

    # Tüketim (Perakende, Giyim, Dayanıklı Tüketim)
    "MAVI":  "Tüketim", "DOAS":  "Tüketim", "DOHOL": "Tüketim",
    "GRSEL": "Tüketim", "KCHOL": "Tüketim", "BERA":  "Tüketim",
    "GSDHO": "Tüketim",

    # Ulaşım
    "PASEU": "Ulaşım", "THYAO": "Ulaşım", "PGSUS": "Ulaşım",
    "TAVHL": "Ulaşım", "TMSN":  "Ulaşım",

    # Yaşam / Spor / Eğlence
    "GSRAY": "Yaşam", "FENER": "Yaşam",

    # Holdingler
    "SAHOL": "Holding", "KCHOL": "Holding", "DOHOL": "Holding",
    "AGHOL": "Holding", "BERA":  "Holding", "GSDHO": "Holding",
    "PAHOL": "Holding",
}

# Çakışan sektör önceliği — bir hisse birden fazla kategoride görünüyorsa
# en spesifik sektör önceliklidir (SECTOR_MAP'te son tanım geçerli olur Python'da)
# Bu yüzden Holding gibi geniş kategoriler en sona eklendi

# ── TAM HİSSE LİSTESİ (sektör haritasından + ek hisseler) ───────────────────
BIST100_OFFICIAL = sorted(set([
    # Çelik
    "EREGL","KRDMD","BRSAN","CVKMD","ISDMR","BRYAT",
    # Enerji
    "AKSEN","ENJSA","CANTE","ENERY","IZENR","TRENJ","ZOREN","MAGEN",
    "ODAS","KONTR","AKFYE","AYDEM","CWENE","ALFAS","ASTOR",
    # Finans
    "DSTKF","SKBNK","TSKB","GARAN","ISCTR","YKBNK","SAHOL","ISMEN",
    "AKBNK","KTLEV","TUREX","HALKB","VAKBN","ALBRK",
    # Gıda ve İçecek
    "CCOLA","AGHOL","TUKAS","AEFES","EFOR","TABGD","KLRHO","BALSU",
    "BIMAS","ULKER","OBAMS","SOKM","MGROS","TATGD",
    # Ham Madde
    "PETKM","GUBRF","SASA","TRMET","AKSA","HEKTS","TRALT","TUPRS",
    # Hizmet
    "GRTHO",
    # İletişim
    "TCELL","TTKOM",
    # İnşaat Malzemeleri
    "EUREN","OYAKC","QUAGR","BSOKE","BTCIM","CIMSA","SISE","AKCNS",
    # İnşaat ve GMYO
    "KUYAS","EKGYO","PSGYO","DAPGM","TKFEN","GLRMK","ALARK","RALYH",
    "ENKAI","TRGYO","SNGYO","AKGYO","OZGYO","VKGYO",
    # Otomotiv
    "FROTO","OTKAR","TOASO","DOAS","TTRAK","BRISA","FMTAS",
    # Sağlık
    "GENIL","MPARK","ECILC",
    # Sanayi
    "EUPWR","ARCLK","VESTL","SARKY",
    # Savunma
    "ASELS","ALTNY","MIATK","SDTTR","ATATP",
    # Sigorta
    "ANSGR","TURSG","AKGRT",
    # Teknoloji
    "PATEK","GESAN","REEDR","ARDYZ","LOGO","NETAS",
    # Tüketim
    "MAVI","DOHOL","GRSEL","BERA","GSDHO","KCHOL",
    # Ulaşım
    "PASEU","THYAO","PGSUS","TAVHL","TMSN",
    # Yaşam
    "GSRAY","FENER",
    # Holding
    "PAHOL",
]))

# Hisse → sektör eşlemesi (çakışma varsa en spesifik sektör)
_FINAL_SECTOR_MAP = {}
for ticker, sector in SECTOR_MAP.items():
    _FINAL_SECTOR_MAP[ticker] = sector

def get_sector(ticker):
    t = ticker.replace(".IS", "")
    return _FINAL_SECTOR_MAP.get(t, "Diğer")

# ── MAKRO TEMALAR ─────────────────────────────────────────────────────────────
MACRO_THEMES_PRIMARY = {
    "💰 Yüksek Faiz": {
        "sektörler": ["Finans", "İnşaat ve GMYO"],
        "açıklama": "Yüksek faiz ortamında banka marjları genişler, mevduat gelirleri artar. GYO'lar baskı altında olsa da seçici fırsatlar doğar.",
        "öne_çıkan": ["AKBNK","GARAN","ISCTR","HALKB","EKGYO","TSKB"],
    },
    "⚔️ Jeopolitik Gerilim": {
        "sektörler": ["Savunma"],
        "açıklama": "Savunma harcamaları ve ihracat artışıyla savunma sanayi şirketleri öne çıkar.",
        "öne_çıkan": ["ASELS","MIATK","SDTTR","ALTNY"],
    },
    "🔋 Teşvik Dönemi / Enerji": {
        "sektörler": ["Enerji", "İnşaat Malzemeleri"],
        "açıklama": "Kamu yatırımları ve enerji dönüşümü teşvikleri enerji ve inşaat malzemesi sektörünü besler.",
        "öne_çıkan": ["AKSEN","ENJSA","ASTOR","ZOREN","OYAKC","CIMSA"],
    },
    "📉 Lira Değer Kaybı / İhracatçı": {
        "sektörler": ["Otomotiv","Çelik","Ham Madde","Sanayi"],
        "açıklama": "Döviz geliri olan ihracatçı şirketler TL değer kaybından doğrudan fayda sağlar.",
        "öne_çıkan": ["FROTO","TOASO","EREGL","KRDMD","TUPRS","ARCLK"],
    },
}

MACRO_THEMES_SECONDARY = {
    "📉 Faiz İndirim Dönemi": {
        "sektörler": ["İnşaat ve GMYO","Finans","İnşaat Malzemeleri"],
        "açıklama": "Faiz düşünce borçlanma maliyeti azalır, konut ve GYO değerlenir, banka kredi hacmi büyür.",
        "öne_çıkan": ["EKGYO","TRGYO","ENKAI","KCHOL","GARAN"],
    },
    "🪨 Emtia Güçlü": {
        "sektörler": ["Çelik","Ham Madde","Enerji"],
        "açıklama": "Global emtia fiyatlarının yükselmesi üretici şirket marjlarını genişletir.",
        "öne_çıkan": ["EREGL","KRDMD","GUBRF","SASA","AKSEN"],
    },
    "🛡️ Piyasa Çalkantılı / Defansif": {
        "sektörler": ["Gıda ve İçecek","İletişim","Sigorta"],
        "açıklama": "Volatil dönemlerde savunmacı sektörler portföyü korur, talep esnekliği düşüktür.",
        "öne_çıkan": ["BIMAS","MGROS","TCELL","TTKOM","ULKER"],
    },
    "🚀 Risk İştahı Yüksek": {
        "sektörler": ["Savunma","Teknoloji","Sanayi","Otomotiv"],
        "açıklama": "Büyüme beklentisinin yüksek olduğu dönemlerde teknoloji ve siklikallar öne çıkar.",
        "öne_çıkan": ["ASELS","ARDYZ","CWENE","FROTO","TOASO","VESTL"],
    },
    "🏥 Sağlık & Demografik Büyüme": {
        "sektörler": ["Sağlık"],
        "açıklama": "Yaşlanan nüfus ve sağlık harcamaları artışıyla sağlık sektörü uzun vadeli büyür.",
        "öne_çıkan": ["MPARK","GENIL","ECILC"],
    },
    "✈️ Turizm & Ulaşım": {
        "sektörler": ["Ulaşım","Yaşam","Hizmet"],
        "açıklama": "Turizm sezonu ve seyahat talebinin artmasıyla havacılık ve ulaşım sektörü güçlenir.",
        "öne_çıkan": ["THYAO","PGSUS","TAVHL","GSRAY","FENER"],
    },
}

MACRO_THEMES = {**MACRO_THEMES_PRIMARY, **MACRO_THEMES_SECONDARY}

def get_theme_sectors(theme):
    return MACRO_THEMES.get(theme, {}).get("sektörler", [])

def get_theme_info(theme):
    return MACRO_THEMES.get(theme, {})

ALL_SECTORS = sorted(set(_FINAL_SECTOR_MAP.values()))
