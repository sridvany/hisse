import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sayfa Ayarları
st.set_page_config(page_title="BIST Pro Analiz", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    input { text-transform: uppercase; }
    .stDataFrame { border: 1px solid #4A4A4A; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 BIST Gelişmiş Trade & Likidite Paneli")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Parametreler")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon Sembolü", value="XU100").strip().upper()
    st.divider()
    run_analysis = st.button("ANALİZİ BAŞLAT")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Lütfen bir sembol giriniz.")
    else:
        with st.spinner(f'{symbol_raw} analiz ediliyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            df = yf.download(ticker, period="60d", interval="1d")
            df_comp = yf.download(comp_ticker, period="60d", interval="1d")

            if df.empty or len(df) < 5:
                st.error(f"❌ {symbol_raw} verisi çekilemedi.")
            else:
                # Sütun Temizliği (MultiIndex Fix)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                if isinstance(df_comp.columns, pd.MultiIndex): df_comp.columns = df_comp.columns.get_level_values(0)

                # --- HESAPLAMALAR ---
                # 1. Daily Range (Günlük Fark)
                df['Daily Range'] = (df['High'] - df['Low']).round(2)
                
                # 2. Günlük Getiri (Mutlak Değer)
                df['Pct_Change'] = df['Close'].pct_change() * 100
                
                # 3. Amihud Illiquidity Ratio (Ölçeklendirilmiş: 10^6)
                # Akademik formül: |Return| / (Volume * Price) -> Biz günlük trade için sadeleştiriyoruz.
                # Değer çok küçük çıkmasın diye 1 milyon ile çarpıyoruz.
                df['Amihud'] = (df['Pct_Change'].abs() / (df['Volume'] / 1000000)).round(4)

                # --- 1. GÖRSELLEŞTİRME ---
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.subheader(f"📊 {symbol_raw} Grafik & VRP")
                    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                                        column_widths=[0.85, 0.15], horizontal_spacing=0.01)
                    
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], 
                        low=df['Low'], close=df['Close'], name="Fiyat"
                    ), row=1, col=1)

                    # Volume Profile
                    bins = 15
                    df['PriceBin'] = pd.cut(df['Close'], bins=bins)
                    vprofile = df.groupby('PriceBin', observed=True)['Volume'].sum()
                    bin_centers = [i.mid for i in vprofile.index]

                    fig.add_trace(go.Bar(
                        x=vprofile.values, y=bin_centers, orientation='h',
                        marker_color='rgba(255, 75, 75, 0.3)', name="Hacim"
                    ), row=1, col=2)

                    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("🔗 İlişki Analizi")
                    st.write(f"**{symbol_raw} / {compare_raw}**")
                    
                    combined = pd.concat([df['Close'], df_comp['Close']], axis=1).dropna().tail(30)
                    combined.columns = ['Hisse', 'Endeks']
                    
                    if not combined.empty:
                        p_corr = combined['Hisse'].corr(combined['Endeks'], method='pearson')
                        s_corr = combined['Hisse'].corr(combined['Endeks'], method='spearman')
                        st.metric("Pearson (Korelasyon)", f"{p_corr:.2f}")
                        st.metric("Spearman (Trend)", f"{s_corr:.2f}")
                        
                        # Ortalama Amihud Bilgisi
                        avg_amihud = df['Amihud'].tail(30).mean()
                        st.metric("Ort. Amihud (30G)", f"{avg_amihud:.3f}")
                        st.caption("Amihud ne kadar düşükse likidite o kadar yüksektir.")
                    else:
                        st.warning("Veri yetersiz.")

                # --- 2. TABLO (SCROLL ÖZELLİĞİ İLE) ---
                st.divider()
                st.subheader("📅 Detaylı Veri Listesi")
                
                # Tablo için Durum Sütunu
                def color_sign(val):
                    if val > 0: return f"🟢 +%{val:.2f}"
                    if val < 0: return f"🔴 -%{abs(val):.2f}"
                    return "⚪ 0.00"

                res_df = df.tail(30).copy()
                res_df['Değişim %'] = res_df['Pct_Change'].apply(color_sign)
                
                # Sütunları düzenleme
                final_cols = ['Open', 'Close', 'Daily Range', 'Amihud', 'Volume', 'Değişim %']
                table_final = res_df[final_cols].sort_index(ascending=False)
                
                # Tabloyu göster
                st.dataframe(table_final, use_container_width=True, height=500)

else:
    st.info("👈 Analize başlamak için sol menüden sembolleri girin.")
