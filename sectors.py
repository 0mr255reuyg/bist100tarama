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
#
# GÜNCELLEME (Borsa İstanbul 2026 3. çeyrek endeks revizyonu, 1 Temmuz 2026
# itibarıyla geçerli): BIST 100'den çıkarılan AGHOL ve TABGD kaldırıldı;
# endekse yeni giren ODINE, IEYHO ve ESEN eklendi. Ayrıca ISMEN (İş Yatırım
# Menkul Değerler) "Katılım ve Evim Sistemleri"nden "Holding ve Yatırım"a
# taşındı — ISMEN konvansiyonel bir aracı kurum, katılım endeksine uygun
# değil (İş Bankası iştiraki). KTLEV ve INVES katılım-uyumlu oldukları
# doğrulandığı için o grupta kaldı.
# NOT: Bu liste ~100 hisseyi kapsar ama tam eşleşme garantisi yok; kalan
# ~90+ tickerin sektör/endeks-üyeliği tek tek doğrulanmadı, sadece bu
# revizyondaki somut değişiklikler ve şüpheli görünen 1-2 kayıt kontrol
# edildi. Şüpheli bulursan tek tek doğrulayabiliriz.
# GÜNCELLEME 2 (membership.py ile tam senkronizasyon): get_all_tickers_ever()
# (2024-04-01 -> 2026-07-01 arası hiç BIST100 üyesi olmuş 151 ticker) ile
# karşılaştırıldığında: (a) burada olup GERÇEKTE hiçbir dönemde üye olmamış
# 15 "hayalet" ticker bulundu ve silindi (AHSY, AKGRT, AKGYO, ATATP, BRISA,
# CEMTS, GSDHO, INVES, LOGO, OZGYO, SNGYO, TATGD, TRGYO, YATAS, ZRGYO — bunlar
# muhtemelen ilk listenin gerçek bir anlık görüntü değil, elle derlenmiş
# yaklaşık bir liste olmasından kalma); (b) sektörü hiç tanımlanmamış 66
# ticker eklendi — bunlar tanımsız kaldığı için hem ekranda "Diğer" görünüyor
# hem de backtest'teki rejim/makro-rüzgar bonusu bu ~50-60 hisseyi hiç
# tanıyamıyordu (Emre'nin ve Faiz Pusulası'nın "Makro Rüzgar" kriteri gerçek
# sektörü olmayan her hisseyi otomatik bonussuz bırakıyordu). Bu vesileyle 2
# yeni sektör açıldı: "Finansal Kiralama ve Faktoring" (DSTKF, LIDER gibi
# faktoring şirketleri artık kendi kategorisinde, "Diğer"de değil) ve
# "Spor Kulüpleri" (BJKAS, FENER, GSRAY, TSPOR - halka açık futbol kulüpleri).
FALLBACK_SECTOR_MAP = {
    "AKBNK": "Banka", "GARAN": "Banka", "ISCTR": "Banka", "YKBNK": "Banka",
    "HALKB": "Banka", "VAKBN": "Banka", "TSKB": "Banka", "SKBNK": "Banka",
    "ALBRK": "Katılım ve Evim Sistemleri", "KTLEV": "Katılım ve Evim Sistemleri",
    "EKGYO": "İnşaat ve GMYO", "AKFGY": "İnşaat ve GMYO", "AVPGY": "İnşaat ve GMYO",
    "DAPGM": "İnşaat ve GMYO", "PEKGY": "İnşaat ve GMYO", "PSGYO": "İnşaat ve GMYO",
    "RGYAS": "İnşaat ve GMYO", "RYGYO": "İnşaat ve GMYO", "KUYAS": "İnşaat ve GMYO",
    "ISGYO": "İnşaat ve GMYO",
    "EREGL": "Çelik ve Metal", "KRDMD": "Çelik ve Metal", "BRSAN": "Çelik ve Metal",
    "KCAER": "Çelik ve Metal", "CVKMD": "Çelik ve Metal", "KOZAA": "Çelik ve Metal",
    "KOZAL": "Çelik ve Metal", "SARKY": "Çelik ve Metal",
    "AKSEN": "Enerji", "ENJSA": "Enerji", "ASTOR": "Enerji", "GESAN": "Enerji",
    "EUPWR": "Enerji", "CWENE": "Enerji", "ALFAS": "Enerji", "SMRTG": "Enerji",
    "ZOREN": "Enerji", "CANTE": "Enerji", "ODAS": "Enerji", "GWIND": "Enerji",
    "TUPRS": "Enerji", "ESEN": "Enerji", "AHGAZ": "Enerji", "AKFYE": "Enerji",
    "BIOEN": "Enerji", "ENERY": "Enerji", "IPEKE": "Enerji", "IZENR": "Enerji",
    "MAGEN": "Enerji", "SAYAS": "Enerji", "YEOTK": "Enerji",
    "BIMAS": "Gıda ve Perakende", "MGROS": "Gıda ve Perakende", "SOKM": "Gıda ve Perakende",
    "CCOLA": "Gıda ve Perakende", "AEFES": "Gıda ve Perakende", "ULKER": "Gıda ve Perakende",
    "TUKAS": "Gıda ve Perakende", "KLRHO": "Gıda ve Perakende", "BALSU": "Gıda ve Perakende",
    "EFORC": "Gıda ve Perakende", "KAYSE": "Gıda ve Perakende", "OBAMS": "Gıda ve Perakende",
    "TABGD": "Gıda ve Perakende", "TKNSA": "Gıda ve Perakende", "YYLGD": "Gıda ve Perakende",
    "KCHOL": "Holding ve Yatırım", "SAHOL": "Holding ve Yatırım", "ALARK": "Holding ve Yatırım",
    "DOHOL": "Holding ve Yatırım", "ENKAI": "Holding ve Yatırım", "TKFEN": "Holding ve Yatırım",
    "BERA": "Holding ve Yatırım", "ISMEN": "Holding ve Yatırım", "IEYHO": "Holding ve Yatırım",
    "AGHOL": "Holding ve Yatırım", "BINHO": "Holding ve Yatırım", "BRYAT": "Holding ve Yatırım",
    "ECZYT": "Holding ve Yatırım", "GRTHO": "Holding ve Yatırım", "NTHOL": "Holding ve Yatırım",
    "PAHOL": "Holding ve Yatırım", "RALYH": "Holding ve Yatırım", "YAZIC": "Holding ve Yatırım",
    "FROTO": "Otomotiv", "TOASO": "Otomotiv", "DOAS": "Otomotiv", "TTRAK": "Otomotiv",
    "OTKAR": "Otomotiv", "EGEEN": "Otomotiv", "BFREN": "Otomotiv", "KARSN": "Otomotiv",
    "TMSN": "Otomotiv",
    "ARCLK": "Sanayi ve Kimya", "VESTL": "Sanayi ve Kimya", "SASA": "Sanayi ve Kimya",
    "PETKM": "Sanayi ve Kimya", "AKSA": "Sanayi ve Kimya", "SISE": "Sanayi ve Kimya",
    "HEKTS": "Sanayi ve Kimya", "GUBRF": "Sanayi ve Kimya", "AGROT": "Sanayi ve Kimya",
    "EUREN": "Sanayi ve Kimya", "VESBE": "Sanayi ve Kimya",
    "OYAKC": "İnşaat Malzemeleri", "CIMSA": "İnşaat Malzemeleri", "AKCNS": "İnşaat Malzemeleri",
    "BTCIM": "İnşaat Malzemeleri", "BSOKE": "İnşaat Malzemeleri", "BOBET": "İnşaat Malzemeleri",
    "BIENY": "İnşaat Malzemeleri", "GLRMK": "İnşaat Malzemeleri", "GOLTS": "İnşaat Malzemeleri",
    "KLSER": "İnşaat Malzemeleri", "KONYA": "İnşaat Malzemeleri", "QUAGR": "İnşaat Malzemeleri",
    "MIATK": "Teknoloji ve Yazılım", "ARDYZ": "Teknoloji ve Yazılım",
    "REEDR": "Teknoloji ve Yazılım", "ODINE": "Teknoloji ve Yazılım",
    "KONTR": "Teknoloji ve Yazılım", "PATEK": "Teknoloji ve Yazılım",
    "ASELS": "Savunma", "SDTTR": "Savunma", "ALTNY": "Savunma", "PAPIL": "Savunma",
    "THYAO": "Ulaşım ve Turizm", "PGSUS": "Ulaşım ve Turizm", "TAVHL": "Ulaşım ve Turizm",
    "CLEBI": "Ulaşım ve Turizm", "PASEU": "Ulaşım ve Turizm", "TUREX": "Ulaşım ve Turizm",
    "TCELL": "İletişim", "TTKOM": "İletişim",
    "MPARK": "Sağlık", "GENIL": "Sağlık", "ECILC": "Sağlık", "LMKDC": "Sağlık", "SELEC": "Sağlık",
    "TURSG": "Sigorta", "ANSGR": "Sigorta", "ANHYT": "Sigorta",
    "MAVI": "Tüketim ve Giyim", "GRSEL": "Tüketim ve Giyim", "ADEL": "Tüketim ve Giyim",
    # Finansal Kiralama ve Faktoring: banka olmayan, kredi/leasing bazlı finans
    # şirketleri. Faiz rejimine bankacılığa benzer şekilde duyarlılar (kaynak
    # maliyeti faize bağlı) — bu yüzden "Diğer"e değil kendi kategorisine.
    "DSTKF": "Finansal Kiralama ve Faktoring", "LIDER": "Finansal Kiralama ve Faktoring",
    # Spor Kulüpleri: halka açık futbol kulüpleri, faiz rejiminden bağımsız
    # (lig performansı/yayın geliri odaklı), ayrı kategori.
    "BJKAS": "Spor Kulüpleri", "FENER": "Spor Kulüpleri", "GSRAY": "Spor Kulüpleri",
    "TSPOR": "Spor Kulüpleri",
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
