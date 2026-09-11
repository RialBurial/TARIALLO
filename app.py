import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Screener Técnico Pro", layout="wide")
st.title("📈 Mi Screener de Análisis Técnico")
st.markdown("Analiza múltiples acciones de un vistazo y filtra las mejores oportunidades.")

# --- INICIALIZAR LA MEMORIA DE STREAMLIT ---
# Creamos una "caja" en la memoria para guardar nuestros datos y que no se borren
if "datos_guardados" not in st.session_state:
    st.session_state.datos_guardados = pd.DataFrame()

# --- BARRA LATERAL PARA INPUTS ---
st.sidebar.header("1. Configuración de Búsqueda")
tickers_input = st.sidebar.text_input(
    "Introduce los Tickers (separados por coma):", 
    "AAPL, MSFT, TSLA, NVDA, AMZN"
)
analizar_btn = st.sidebar.button("Analizar Mercado")

# --- FUNCIÓN PARA OBTENER Y CALCULAR DATOS ---
@st.cache_data
def obtener_datos_tecnicos(tickers_list):
    resultados = []
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=300)
    
    for ticker in tickers_list:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
                
            close_prices = df['Close'].squeeze()
            precio_actual = close_prices.iloc[-1]
            rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
            sma50 = ta.trend.SMAIndicator(close_prices, window=50).sma_indicator().iloc[-1]
            sma200 = ta.trend.SMAIndicator(close_prices, window=200).sma_indicator().iloc[-1]
            
            macd_obj = ta.trend.MACD(close_prices)
            macd_line = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            
            resultados.append({
                "Ticker": ticker,
                "Precio ($)": round(float(precio_actual), 2),
                "RSI": round(float(rsi), 2),
                "SMA 50": round(float(sma50), 2),
                "SMA 200": round(float(sma200), 2),
                "MACD": round(float(macd_line), 2),
                "Señal MACD": round(float(macd_signal), 2)
            })
        except Exception as e:
            pass
            
    return pd.DataFrame(resultados)

# --- LÓGICA DEL BOTÓN ---
# Si pulsamos el botón, descargamos los datos y los guardamos en la memoria
if analizar_btn:
    lista_tickers = [t.strip().upper() for t in tickers_input.split(",")]
    with st.spinner("Descargando datos del mercado y calculando métricas..."):
        # ¡Aquí está la magia! Guardamos el resultado en st.session_state
        st.session_state.datos_guardados = obtener_datos_tecnicos(lista_tickers)

# --- MOSTRAR DATOS Y FILTROS ---
# Si la memoria no está vacía, mostramos los filtros (independientemente del botón)
if not st.session_state.datos_guardados.empty:
    st.success("¡Datos cargados y listos para filtrar!")
    
    st.subheader("2. Filtra tus oportunidades")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_rsi = st.slider("RSI Máximo (Ej. 30 para sobreventa):", 0, 100, 100)
    with col2:
        solo_tendencia_alcista = st.checkbox("Solo Tendencia Alcista (Precio > SMA 200)")
    with col3:
        solo_macd_alcista = st.checkbox("MACD Cruzando al alza (MACD > Señal)")
        
    # Aplicamos los filtros a los datos que están en la memoria
    df_filtrado = st.session_state.datos_guardados.copy()
    
    df_filtrado = df_filtrado[df_filtrado['RSI'] <= max_rsi]
    
    if solo_tendencia_alcista:
        df_filtrado = df_filtrado[df_filtrado['Precio ($)'] > df_filtrado['SMA 200']]
        
    if solo_macd_alcista:
        df_filtrado = df_filtrado[df_filtrado['MACD'] > df_filtrado['Señal MACD']]
        
    st.write(f"Mostrando {len(df_filtrado)} valores que cumplen tus criterios:")
    st.dataframe(
        df_filtrado, 
        use_container_width=True,
        hide_index=True
    )
