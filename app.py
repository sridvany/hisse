import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Ayarları
st.set_page_config(page_title="BIST Trading Terminal", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; }
    input { text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ BIST Profesyonel Analiz Terminali")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Veri Ayarları")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon", value="XU100").strip().upper()
    
    st.divider()
    # Geniş tarih aralığı
    g_start = st.date_input("Başlangıç", value=datetime(2020, 1, 1))
    g_end = st.date_input("Bitiş", value=datetime.now())
    
    st.divider()
    run_analysis = st.button("ANALİZİ BAŞLAT")

if run_analysis and symbol_raw:
    with st.spinner('Veriler işleniyor...'):
        t = f"{symbol_raw}.IS" if not symbol_raw.endswith(".IS") else symbol_raw
        ct = f"{compare_raw}.IS" if not compare_raw.endswith(".IS") else compare_raw
        
        df = yf.download(t, start=g_start, end=g_end)
        df_c = yf.download(ct, start=g_start, end=g_end)

        if not df.empty:
            # MultiIndex Fix
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if isinstance(df_c.columns, pd.MultiIndex): df_c.columns = df_c.columns.get_level_values(0)

            # --- HESAPLAMALAR ---
            df['Daily Range'] = (df['High'] - df['Low']).round(2)
            df['Pct'] = df['Close'].pct_change() * 100
            # Amihud Skoru (Görünür olması için 10^7 ile çarpıldı)
            df['Amihud'] = (df['Pct'].abs() / df['Volume'] * 10000000).round(4)

            # --- 1. ANA GRAFİK & AYRAÇ ---
            st.subheader(f"📊 {symbol_raw} Candlestick & Volume Profile")
            fig1 = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
            
            # VRP
            v_bins = pd.cut(df['Close'], bins=20)
            v_prof = df.groupby(v_bins, observed=True)['Volume'].sum()
            fig1.add_trace(go.Bar(x=v_prof.values, y=[i.mid for i in v_prof.index], orientation='h', marker_color='rgba(255, 75, 75, 0.2)'), row=1, col=2)
            
            fig1.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis=dict(rangeslider_visible=True))
            st.plotly_chart(fig1, use_container_width=True)

            # --- 2. DETAY VERİ TABLOSU ---
            st.divider()
            st.subheader("📅 Detay Veri Listesi")
            df_out = df.tail(60).copy()
            df_out['Değişim %'] = df_out['Pct'].apply(lambda x: f"🟢 +%{x:.2f}" if x > 0 else f"🔴 -%{abs(x):.2f}" if x < 0 else "⚪ 0.00")
            st.dataframe(df_out[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Range', 'Amihud', 'Değişim %']].sort_index(ascending=False), use_container_width=True, height=300)

            # --- 3. DUAL AXIS ANALİZLER (AYRAÇLI) ---
            st.divider()
            st.subheader("📈 Amihud (Sol) vs Daily Range (Sağ)")
            
            def make_dual(y1, n1, c1, y2, n2, c2, dash=None):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=y1, name=n1, line=dict(color=c1, width=2), yaxis="y1"))
                fig.add_trace(go.Scatter(x=df.index, y=y2, name=n2, line=dict(color=c2, width=2, dash=dash), yaxis="y2"))
                fig.update_layout(
                    template="plotly_dark", height=450,
                    yaxis=dict(title=n1, titlefont=dict(color=c1), tickfont=dict(color=c1), autorange=True),
                    yaxis2=dict(title=n2, titlefont=dict(color=c2), tickfont=dict(color=c2), anchor="x", overlaying="y", side="right", autorange=True),
                    xaxis=dict(rangeslider_visible=True), # İSTEDİĞİN AYRAÇ BURADA
                    hovermode="x unified", margin=dict(t=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Amihud vs Daily Range
            make_dual(df['Amihud'], "Amihud (Likidite)", "#00FFCC", df['Daily Range'], "Daily Range", "#FFD700", dash='dot')
            
            # Daily Range vs Close
            st.subheader("📈 Daily Range vs Close")
            make_dual(df['Close'], "Fiyat (Close)", "#FFFFFF", df['Daily Range'], "Daily Range", "#FFD700")

            # Amihud vs Close
            st.subheader("🧪 Amihud vs Close")
            make_dual(df['Close'], "Fiyat (Close)", "#FFFFFF", df['Amihud'], "Amihud", "#00FFCC")

else:
    st.info("👈 Analizi başlatmak için sembol girin ve butona basın.")
