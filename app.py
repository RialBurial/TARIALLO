import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Screener Técnico Pro", layout="wide")
st.title("📈 Mi Screener de Análisis Técnico")
st.markdown("Analiza múltiples acciones de un vistazo y filtra las mejores oportunidades.")

# --- BARRA LATERAL PARA INPUTS ---
st.sidebar.header("1. Configuración de Búsqueda")
# Pedimos al usuario que introduzca los tickers separados por comas
tickers_input = st.sidebar.text_input(
    "Introduce los Tickers (separados por coma):", 
    "AAPL, MSFT, TSLA, NVDA, AMZN"
)

# Botón para ejecutar el análisis
analizar_btn = st.sidebar.button("Analizar Mercado")

# --- FUNCIÓN PARA OBTENER Y CALCULAR DATOS ---
# Usamos cache para no descargar los datos de internet cada vez que tocamos un botón
@st.cache_data
def obtener_datos_tecnicos(tickers_list):
    resultados = []
    
    # Calculamos fechas (últimos 300 días para asegurar que el SMA 200 se calcula bien)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=300)
    
    for ticker in tickers_list:
        try:
            # Descargamos datos de Yahoo Finance
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                continue
                
            # Extraemos la columna de precios de cierre
            close_prices = df['Close'].squeeze()
            
            # --- CÁLCULO DE INDICADORES (Usando la librería 'ta') ---
            precio_actual = close_prices.iloc[-1]
            
            # RSI (14 periodos)
            rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
            
            # Medias Móviles (50 y 200)
            sma50 = ta.trend.SMAIndicator(close_prices, window=50).sma_indicator().iloc[-1]
            sma200 = ta.trend.SMAIndicator(close_prices, window=200).sma_indicator().iloc[-1]
            
            # MACD
            macd_obj = ta.trend.MACD(close_prices)
            macd_line = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            
            # Guardamos los resultados en un diccionario
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
            # Si hay un error con un ticker (ej. no existe), lo saltamos
            pass
            
    # Convertimos la lista de resultados en una tabla (DataFrame)
    return pd.DataFrame(resultados)

# --- LÓGICA PRINCIPAL DE LA APP ---
if analizar_btn:
    # Limpiamos los espacios en blanco de los tickers
    lista_tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    with st.spinner("Descargando datos del mercado y calculando métricas..."):
        df_resultados = obtener_datos_tecnicos(lista_tickers)
    
    if not df_resultados.empty:
        st.success("¡Análisis completado con éxito!")
        
        # --- FILTROS (La magia del Screener) ---
        st.subheader("2. Filtra tus oportunidades")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_rsi = st.slider("RSI Máximo (Ej. 30 para sobreventa):", 0, 100, 100)
        with col2:
            solo_tendencia_alcista = st.checkbox("Solo Tendencia Alcista (Precio > SMA 200)")
        with col3:
            solo_macd_alcista = st.checkbox("MACD Cruzando al alza (MACD > Señal)")
            
        # Aplicamos los filtros matemáticos al DataFrame
        df_filtrado = df_resultados.copy()
        
        df_filtrado = df_filtrado[df_filtrado['RSI'] <= max_rsi]
        
        if solo_tendencia_alcista:
            df_filtrado = df_filtrado[df_filtrado['Precio ($)'] > df_filtrado['SMA 200']]
            
        if solo_macd_alcista:
            df_filtrado = df_filtrado[df_filtrado['MACD'] > df_filtrado['Señal MACD']]
            
        # Mostramos la tabla final
        st.write(f"Mostrando {len(df_filtrado)} valores que cumplen tus criterios:")
        st.dataframe(
            df_filtrado, 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("No se han podido obtener datos. Revisa los tickers introducidos.")