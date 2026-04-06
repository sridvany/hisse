import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Ayarları
st.set_page_config(page_title="BIST Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; height: 3em; }
    input { text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ BIST Gelişmiş Analiz Terminali (Likidite Fix)")

with st.sidebar:
    st.header("🔍 Global Veri Çek")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon Sembolü", value="XU100").strip().upper()
    
    st.divider()
    st.subheader("📅 Veri Havuzu Aralığı")
    g_start = st.date_input("Başlangıç Tarihi", value=datetime.now() - timedelta(days=365*2))
    g_end = st.date_input("Bitiş Tarihi", value=datetime.now())
    
    st.divider()
    run_analysis = st.button("VERİLERİ ANALİZ ET")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Önce bir sembol girin.")
    else:
        with st.spinner('Grafikler hazırlanıyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            df = yf.download(ticker, start=g_start, end=g_end, interval="1d")
            df_comp = yf.download(comp_ticker, start=g_start, end=g_end, interval="1d")

            if df.empty:
                st.error("Veri bulunamadı.")
            else:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                if isinstance(df_comp.columns, pd.MultiIndex): df_comp.columns = df_comp.columns.get_level_values(0)

                # --- HESAPLAMALAR ---
                df['Daily Range'] = (df['High'] - df['Low']).round(2)
                df['Pct_Change'] = df['Close'].pct_change() * 100
                
                # Amihud Fix: Çarpanı 10^8 yaparak değeri görünür kılıyoruz
                # Formül: (|Getiri| / Hacim) * 100.000.000
                df['Amihud'] = (df['Pct_Change'].abs() / df['Volume'] * 100000000).round(4)

                # --- 1. ANA GRAFİK ---
                st.subheader(f"📊 {symbol_raw} - Fiyat ve Hacim Profili")
                fig_main = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
                fig_main.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
                
                # VRP
                bins = 20
                df['PriceBin'] = pd.cut(df['Close'], bins=bins)
                v_prof = df.groupby('PriceBin', observed=True)['Volume'].sum()
                fig_main.add_trace(go.Bar(x=v_prof.values, y=[i.mid for i in v_prof.index], orientation='h', marker_color='rgba(255, 75, 75, 0.2)', name="Hacim"), row=1, col=2)

                fig_main.update_xaxes(rangeslider_visible=True)
                fig_main.update_layout(height=600, template="plotly_dark", showlegend=False, dragmode='pan')
                st.plotly_chart(fig_main, use_container_width=True, config={'scrollZoom': True})

                # --- 2. DUAL AXIS ANALİZLER (DÜZELTİLMİŞ EKSENLER) ---
                st.divider()
                st.subheader("📉 Likidite ve Volatilite Trend Analizi")
                
                def create_dual_fixed(title, y1, name1, col1, y2, name2, col2, dash=None):
                    f = go.Figure()
                    f.add_trace(go.Scatter(x=df.index, y=y1, name=name1, line=dict(color=col1, width=2.5), yaxis="y1"))
                    f.add_trace(go.Scatter(x=df.index, y=y2, name=name2, line=dict(color=col2, width=2.5, dash=dash), yaxis="y2"))
                    
                    f.update_layout(
                        title=title, template="plotly_dark", height=500,
                        # Y1 Ekseni (Amihud için özel ayar: autorange=True sayesinde 0'a yapışmaz)
                        yaxis=dict(
                            title=dict(text=name1, font=dict(color=col1)), 
                            tickfont=dict(color=col1),
                            autorange=True, 
                            fixedrange=False
                        ),
                        # Y2 Ekseni (Daily Range veya Close)
                        yaxis2=dict(
                            title=dict(text=name2, font=dict(color=col2)), 
                            tickfont=dict(color=col2), 
                            anchor="x", overlaying="y", side="right",
                            autorange=True,
                            fixedrange=False
                        ),
                        hovermode="x unified",
                        xaxis=dict(rangeslider_visible=True),
                        dragmode='pan'
                    )
                    st.plotly_chart(f, use_container_width=True, config={'scrollZoom': True})

                # Grafikleri Oluştur
                create_dual_fixed("Amihud (Sol) vs Daily Range (Sağ)", df['Amihud'], "Amihud Skoru", "#00FFCC", df['Daily Range'], "Daily Range", "#FFD700", dash='dot')
                create_dual_fixed("Fiyat (Sol) vs Daily Range (Sağ)", df['Close'], "Fiyat", "#FFFFFF", df['Daily Range'], "Daily Range", "#FFD700")
                create_dual_fixed("Fiyat (Sol) vs Amihud (Sağ)", df['Close'], "Fiyat", "#FFFFFF", df['Amihud'], "Amihud Skoru", "#00FFCC")

                # --- 3. TABLO ---
                st.divider()
                st.subheader("📅 Detay Veri Listesi")
                df['Değişim %'] = df['Pct_Change'].apply(lambda x: f"🟢 +%{x:.2f}" if x > 0 else f"🔴 -%{abs(x):.2f}" if x < 0 else "⚪ 0.00")
                st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Range', 'Amihud', 'Değişim %']].sort_index(ascending=False), use_container_width=True, height=400)

else:
    st.info("👈 Analizi başlatmak için bir sembol girin.")
