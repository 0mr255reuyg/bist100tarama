# Midas ekranından alınan güncel BIST hisse listesi — sektör sektör

SECTOR_MAP = {
    # Çelik (4)
    "EREGL":"Çelik","KRDMD":"Çelik","BRSAN":"Çelik","CVKMD":"Çelik",

    # Enerji (10)
    "AKSEN":"Enerji","ENJSA":"Enerji","CANTE":"Enerji","ENERY":"Enerji",
    "IZENR":"Enerji","TRENJ":"Enerji","ZOREN":"Enerji","MAGEN":"Enerji",
    "ODAS":"Enerji","AKFYE":"Enerji",

    # Finans (14)
    "GARAN":"Finans","AKBNK":"Finans","ISCTR":"Finans","YKBNK":"Finans",
    "HALKB":"Finans","VAKBN":"Finans","ALBRK":"Finans","TSKB":"Finans",
    "SKBNK":"Finans","ISMEN":"Finans","KTLEV":"Finans","TUREX":"Finans",
    "DSTKF":"Finans","SAHOL":"Finans",

    # Gıda ve İçecek (12)
    "BIMAS":"Gıda ve İçecek","MGROS":"Gıda ve İçecek","SOKM":"Gıda ve İçecek",
    "AEFES":"Gıda ve İçecek","CCOLA":"Gıda ve İçecek","ULKER":"Gıda ve İçecek",
    "TATGD":"Gıda ve İçecek","AGHOL":"Gıda ve İçecek","TUKAS":"Gıda ve İçecek",
    "TABGD":"Gıda ve İçecek","KLRHO":"Gıda ve İçecek","OBAMS":"Gıda ve İçecek",

    # Ham Madde (8)
    "PETKM":"Ham Madde","GUBRF":"Ham Madde","SASA":"Ham Madde",
    "TRMET":"Ham Madde","AKSA":"Ham Madde","HEKTS":"Ham Madde",
    "TRALT":"Ham Madde","TUPRS":"Ham Madde",

    # İletişim (2)
    "TCELL":"İletişim","TTKOM":"İletişim",

    # İnşaat Malzemeleri (6)
    "OYAKC":"İnşaat Malzemeleri","BSOKE":"İnşaat Malzemeleri",
    "BTCIM":"İnşaat Malzemeleri","CIMSA":"İnşaat Malzemeleri",
    "SISE":"İnşaat Malzemeleri","AKCNS":"İnşaat Malzemeleri",

    # İnşaat ve GMYO (8)
    "EKGYO":"İnşaat ve GMYO","TRGYO":"İnşaat ve GMYO","ALARK":"İnşaat ve GMYO",
    "ENKAI":"İnşaat ve GMYO","TKFEN":"İnşaat ve GMYO","SNGYO":"İnşaat ve GMYO",
    "AKGYO":"İnşaat ve GMYO","OZGYO":"İnşaat ve GMYO",

    # Otomotiv (6)
    "FROTO":"Otomotiv","TOASO":"Otomotiv","OTKAR":"Otomotiv",
    "DOAS":"Otomotiv","TTRAK":"Otomotiv","BRISA":"Otomotiv",

    # Sağlık (3)
    "MPARK":"Sağlık","ECILC":"Sağlık","GENIL":"Sağlık",

    # Sanayi (4)
    "ARCLK":"Sanayi","VESTL":"Sanayi","EUPWR":"Sanayi","SARKY":"Sanayi",

    # Savunma (4)
    "ASELS":"Savunma","MIATK":"Savunma","SDTTR":"Savunma","ATATP":"Savunma",

    # Sigorta (3)
    "ANSGR":"Sigorta","TURSG":"Sigorta","AKGRT":"Sigorta",

    # Teknoloji (4)
    "ARDYZ":"Teknoloji","LOGO":"Teknoloji","NETAS":"Teknoloji","REEDR":"Teknoloji",

    # Holding (5)
    "KCHOL":"Holding","DOHOL":"Holding","BERA":"Holding","GSDHO":"Holding","AGHOL":"Holding",

    # Ulaşım (4)
    "THYAO":"Ulaşım","PGSUS":"Ulaşım","TAVHL":"Ulaşım","TMSN":"Ulaşım",

    # Tüketim (3)
    "MAVI":"Tüketim","GRSEL":"Tüketim","DOHOL":"Tüketim",
}

# Tekrarları çıkar, sıralı liste
BIST100_OFFICIAL = sorted(set(SECTOR_MAP.keys()))

def get_sector(ticker):
    t = ticker.replace(".IS","")
    return SECTOR_MAP.get(t, "Diğer")

MACRO_THEMES_PRIMARY = {
    "💰 Yüksek Faiz": {
        "sektörler": ["Finans","İnşaat ve GMYO"],
        "açıklama": "Yüksek faiz ortamında banka marjları genişler.",
        "öne_çıkan": ["AKBNK","GARAN","ISCTR","HALKB","EKGYO"],
    },
    "⚔️ Jeopolitik Gerilim": {
        "sektörler": ["Savunma"],
        "açıklama": "Savunma harcamaları artar, yerli savunma şirketleri öne çıkar.",
        "öne_çıkan": ["ASELS","MIATK","SDTTR"],
    },
    "🔋 Teşvik Dönemi / Enerji": {
        "sektörler": ["Enerji","İnşaat Malzemeleri"],
        "açıklama": "Kamu teşvikleri enerji ve inşaat malzemesi sektörünü besler.",
        "öne_çıkan": ["AKSEN","ENJSA","ZOREN","OYAKC","CIMSA"],
    },
    "📉 Lira Değer Kaybı / İhracatçı": {
        "sektörler": ["Otomotiv","Çelik","Ham Madde","Sanayi"],
        "açıklama": "Döviz geliri olan ihracatçılar TL değer kaybından fayda sağlar.",
        "öne_çıkan": ["FROTO","TOASO","EREGL","KRDMD","ARCLK"],
    },
}

MACRO_THEMES_SECONDARY = {
    "📉 Faiz İndirim Dönemi": {
        "sektörler": ["İnşaat ve GMYO","Finans","İnşaat Malzemeleri"],
        "açıklama": "Faiz düşünce konut ve GYO değerlenir.",
        "öne_çıkan": ["EKGYO","TRGYO","ENKAI","GARAN"],
    },
    "🪨 Emtia Güçlü": {
        "sektörler": ["Çelik","Ham Madde","Enerji"],
        "açıklama": "Emtia fiyatları yükselince üretici marjları genişler.",
        "öne_çıkan": ["EREGL","KRDMD","GUBRF","AKSEN"],
    },
    "🛡️ Piyasa Çalkantılı / Defansif": {
        "sektörler": ["Gıda ve İçecek","İletişim","Sigorta"],
        "açıklama": "Volatil dönemlerde savunmacı sektörler portföyü korur.",
        "öne_çıkan": ["BIMAS","MGROS","TCELL","TTKOM"],
    },
    "🚀 Risk İştahı Yüksek": {
        "sektörler": ["Savunma","Teknoloji","Sanayi","Otomotiv"],
        "açıklama": "Büyüme beklentisinde teknoloji ve siklikallar öne çıkar.",
        "öne_çıkan": ["ASELS","ARDYZ","FROTO","TOASO","VESTL"],
    },
    "🏥 Sağlık": {
        "sektörler": ["Sağlık"],
        "açıklama": "Sağlık harcamaları artışıyla uzun vadeli büyüme.",
        "öne_çıkan": ["MPARK","GENIL","ECILC"],
    },
    "✈️ Turizm & Ulaşım": {
        "sektörler": ["Ulaşım"],
        "açıklama": "Turizm sezonu ve seyahat talebinin artmasıyla güçlenir.",
        "öne_çıkan": ["THYAO","PGSUS","TAVHL"],
    },
}

MACRO_THEMES = {**MACRO_THEMES_PRIMARY, **MACRO_THEMES_SECONDARY}

def get_theme_sectors(theme):
    return MACRO_THEMES.get(theme, {}).get("sektörler", [])

def get_theme_info(theme):
    return MACRO_THEMES.get(theme, {})

ALL_SECTORS = sorted(set(SECTOR_MAP.values()))
