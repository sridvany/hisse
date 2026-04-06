import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sayfa Ayarları
st.set_page_config(page_title="BIST Analiz", layout="wide", initial_sidebar_state="expanded")

# Arayüzü özelleştirmek için küçük bir dokunuş
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stTextInput>div>div>input { text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 BIST Profesyonel Günlük Trade Paneli")

# --- SIDEBAR (Sol Menü) ---
with st.sidebar:
    st.header("🔍 Analiz Ayarları")
    
    # "Hisse" kelimesi kaldırıldı, sadece "Sembol" yapıldı.
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO, ASELS...").strip().upper()
    compare_raw = st.text_input("Korelasyon (Endeks/Hisse)", value="XU100").strip().upper()
    
    st.divider()
    
    # Çalıştır butonu
    run_analysis = st.button("ANALİZİ BAŞLAT")

# Sembol Formatlama Fonksiyonu
def format_bist(s):
    if not s: return None
    # Eğer kullanıcı zaten .IS eklediyse dokunma, eklemediyse ekle
    return f"{s}.IS" if not s.endswith(".IS") else s

# --- ANA EKRAN MANTIĞI ---
if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Lütfen önce bir sembol giriniz.")
    else:
        with st.spinner('Veriler çekiliyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            # Veri Çekme (60 günlük veri çekip analiz ediyoruz)
            df = yf.download(ticker, period="60d", interval="1d")
            df_comp = yf.download(comp_ticker, period="60d", interval="1d")

            if df.empty or len(df) < 5:
                st.error(f"❌ {symbol_raw} için veri bulunamadı. Sembolün doğruluğunu kontrol edin.")
            else:
                # --- 1. GRAFİK VE VOLUME PROFILE ---
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.subheader(f"📊 {symbol_raw} Teknik Görünüm")
                    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                                        column_widths=[0.85, 0.15], horizontal_spacing=0.01)

                    # Mum Grafiği
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], 
                        low=df['Low'], close=df['Close'], name="Fiyat"
                    ), row=1, col=1)

                    # Manuel Hacim Profili (VRP) Hesaplama
                    bins = 20
                    df['PriceBin'] = pd.cut(df['Close'], bins=bins)
                    vprofile = df.groupby('PriceBin', observed=True)['Volume'].sum()
                    bin_centers = [i.mid for i in vprofile.index]

                    fig.add_trace(go.Bar(
                        x=vprofile.values, y=bin_centers, orientation='h',
                        marker_color='rgba(100, 150, 250, 0.4)', name="Hacim Profili"
                    ), row=1, col=2)

                    fig.update_layout(xaxis_rangeslider_visible=False, height=550, 
                                      template="plotly_dark", showlegend=False,
                                      margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("🔗 Korelasyon")
                    # Son 30 günlük kapanışları hizala
                    combined = pd.concat([df['Close'], df_comp['Close']], axis=1).dropna().tail(30)
                    combined.columns = ['Hisse', 'Endeks']
                    
                    if not combined.empty:
                        p_corr = combined['Hisse'].corr(combined['Endeks'], method='pearson')
                        s_corr = combined['Hisse'].corr(combined['Endeks'], method='spearman')
                        
                        st.metric("Pearson (Lineer)", f"{p_corr:.2f}")
                        st.metric("Spearman (Trend)", f"{s_corr:.2f}")
                        st.caption(f"Veriler {compare_raw} ile son 30 gün için kıyaslanmıştır.")
                    else:
                        st.warning("Veri yetersiz.")

                # --- 2. LİSTE (30 GÜNLÜK) ---
                st.divider()
                st.subheader("📅 Son 30 Günlük Fiyat Hareketleri")
                
                res_df = df.tail(30).copy()
                res_df['Günlük Değişim %'] = (res_df['Close'].pct_change() * 100).round(2)
                
                def color_sign(val):
                    if val > 0: return f"🟢 +%{val}"
                    if val < 0: return f"🔴 -%{abs(val)}"
                    return "⚪ 0.00"

                res_df['Durum'] = res_df['Günlük Değişim %'].apply(color_sign)
                
                # Tablo Görünümü
                table_to_show = res_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Durum']].sort_index(ascending=False)
                st.dataframe(table_to_show, use_container_width=True)

else:
    # Boş Başlangıç Ekranı
    st.info("👈 Analize başlamak için sol tarafa bir sembol yazın ve butona tıklayın.")
    st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=1000", caption="BIST Data Terminal")
