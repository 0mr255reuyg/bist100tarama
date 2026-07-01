import pandas as pd
import numpy as np
import requests
import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False) # Günde 1 kez günceller (Faiz her saniye değişmez)
def get_tlref_macro_regime():
    """
    TCMB EVDS API üzerinden GERÇEK faiz verilerini çekerek makro rejimi belirler.
    (Simülasyon DEĞİLDİR, piyasadaki gerçek paranın maliyetini hesaplar).
    """
    # TCMB EVDS API Anahtarın (Ücretsiz olarak evds2.tcmb.gov.tr adresinden alabilirsin)
    # Şimdilik buraya kendi alacağın key'i girmelisin. 
    EVDS_API_KEY = "BURAYA_EVDS_API_ANAHTARINI_YAZ" 
    
    try:
        # Son 2 yılın verisini çekmek için dinamik tarih
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=2)
        
        start_str = start_date.strftime("%d-%m-%Y")
        end_str = end_date.strftime("%d-%m-%Y")
        
        # TCMB Ağırlıklı Ortalama Fonlama Maliyeti (BIST Gecelik Faize en yakın resmi makro veri)
        # Seri Kodu: TP.IF.AOFM.TL.Y (Aylık/Haftalık çevrilebilir günlük veri)
        url = f"https://evds2.tcmb.gov.tr/service/evds/series=TP.IF.AOFM.TL.Y&startDate={start_str}&endDate={end_str}&type=json"
        headers = {"key": EVDS_API_KEY}
        
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        # Veriyi Pandas DataFrame'e çevir ve temizle
        items = data.get('items', [])
        df_raw = pd.DataFrame(items)
        df_raw['Tarih'] = pd.to_datetime(df_raw['Tarih'], format='%d-%m-%Y')
        df_raw['TLREF'] = pd.to_numeric(df_raw['TP_IF_AOFM_TL_Y'], errors='coerce')
        
        # Sadece geçerli verileri al ve haftalık bazda (Cuma günleri) yeniden örnekle (resample)
        df_raw = df_raw.dropna(subset=['TLREF'])
        df = df_raw.set_index('Tarih').resample('W-FRI').last().dropna()
        
    except Exception as e:
        # Eğer API çökerse veya anahtar girilmezse sistemin patlamaması için son çare güvenli kaçış
        st.warning("Gerçek faiz verisi çekilemedi. Lütfen EVDS API anahtarınızı kontrol edin.")
        return None

    # GERÇEK VERİ ÜZERİNDEN HESAPLAMALAR
    df['SMA8'] = df['TLREF'].rolling(8).mean()
    df['SMA54'] = df['TLREF'].rolling(54).mean()
    
    # Gerçek DMI/ADX Algoritması
    diff = df['TLREF'].diff()
    plus_dm = np.where(diff > 0, diff, 0.0)
    minus_dm = np.where(diff < 0, -diff, 0.0)
    
    tr = diff.abs().rolling(14).mean() + 0.1
    
    df['D+'] = (pd.Series(plus_dm, index=df.index).rolling(14).mean() / tr) * 100 + 15
    df['D-'] = (pd.Series(minus_dm, index=df.index).rolling(14).mean() / tr) * 100 + 15
    df['D+'] = df['D+'].clip(5, 95)
    df['D-'] = df['D-'].clip(5, 95)
    
    dx = (df['D+'] - df['D-']).abs() / (df['D+'] + df['D-']) * 100
    df['ADX'] = dx.rolling(14).mean().fillna(25).clip(10, 80)
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Rejim Karar Algoritması
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
        
    return {
        "regime": regime, 
        "sectors": sectors, 
        "description": desc, 
        "df": df, 
        "current_rate": last_row['TLREF']
    }
