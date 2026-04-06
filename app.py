import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sayfa Ayarları
st.set_page_config(page_title="BIST Analiz", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stTextInput>div>div>input { text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 BIST Profesyonel Günlük Trade Paneli")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Analiz Ayarları")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO, ASELS...").strip().upper()
    compare_raw = st.text_input("Korelasyon (Endeks/Hisse)", value="XU100").strip().upper()
    st.divider()
    run_analysis = st.button("ANALİZİ BAŞLAT")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Lütfen önce bir sembol giriniz.")
    else:
        with st.spinner('Veriler çekiliyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            # Veri Çekme
            df = yf.download(ticker, period="60d", interval="1d")
            df_comp = yf.download(comp_ticker, period="60d", interval="1d")

            if df.empty or len(df) < 5:
                st.error(f"❌ {symbol_raw} için veri bulunamadı.")
            else:
                # KRİTİK DÜZELTME: MultiIndex sütunları temizle (Hatanın kaynağı burası)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if isinstance(df_comp.columns, pd.MultiIndex):
                    df_comp.columns = df_comp.columns.get_level_values(0)

                # Veriyi 1 Boyutlu (Series) hale getir
                close_prices = df['Close'].squeeze()
                low_prices = df['Low'].squeeze()
                high_prices = df['High'].squeeze()
                open_prices = df['Open'].squeeze()
                volumes = df['Volume'].squeeze()

                # --- 1. GRAFİK VE VOLUME PROFILE ---
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.subheader(f"📊 {symbol_raw} Teknik Görünüm")
                    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                                        column_widths=[0.85, 0.15], horizontal_spacing=0.01)

                    fig.add_trace(go.Candlestick(
                        x=df.index, open=open_prices, high=high_prices, 
                        low=low_prices, close=close_prices, name="Fiyat"
                    ), row=1, col=1)

                    # Hacim Profili Hesaplama (Hatayı önlemek için Series kullanıyoruz)
                    bins = 20
                    df['PriceBin'] = pd.cut(close_prices, bins=bins)
                    vprofile = df.groupby('PriceBin', observed=True)['Volume'].sum()
                    bin_centers = [i.mid for i in vprofile.index]

                    fig.add_trace(go.Bar(
                        x=vprofile.values, y=bin_centers, orientation='h',
                        marker_color='rgba(100, 150, 250, 0.4)', name="Hacim Profili"
                    ), row=1, col=2)

                    fig.update_layout(xaxis_rangeslider_visible=False, height=550, 
                                      template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("🔗 Korelasyon")
                    # Korelasyon için Series olarak hizala
                    combined = pd.concat([close_prices, df_comp['Close'].squeeze()], axis=1).dropna().tail(30)
                    combined.columns = ['Hisse', 'Endeks']
                    
                    if not combined.empty:
                        p_corr = combined['Hisse'].corr(combined['Endeks'], method='pearson')
                        s_corr = combined['Hisse'].corr(combined['Endeks'], method='spearman')
                        
                        st.metric("Pearson (Lineer)", f"{p_corr:.2f}")
                        st.metric("Spearman (Trend)", f"{s_corr:.2f}")
                    else:
                        st.warning("Veri yetersiz.")

                # --- 2. LİSTE ---
                st.divider()
                st.subheader("📅 Son 30 Günlük Fiyat Hareketleri")
                
                res_df = df.tail(30).copy()
                # Değişim yüzdesini hesapla
                diffs = close_prices.tail(31).pct_change() * 100
                res_df['Günlük Değişim %'] = diffs.tail(30).values.round(2)
                
                def color_sign(val):
                    if val > 0: return f"🟢 +%{val}"
                    if val < 0: return f"🔴 -%{abs(val)}"
                    return "⚪ 0.00"

                res_df['Durum'] = res_df['Günlük Değişim %'].apply(color_sign)
                
                final_table = res_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Durum']].sort_index(ascending=False)
                st.dataframe(final_table, use_container_width=True)

else:
    st.info("👈 Analize başlamak için sol tarafa bir sembol yazın ve butona tıklayın.")
