import streamlit as st
import yfinance as yf
import pandas as pd
import ta

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
# Usamos un diseño panorámico (wide) para que el Dashboard respire
st.set_page_config(page_title="Value Screener Pro", layout="wide")

# Inicializamos la memoria para guardar los datos
if "datos_guardados" not in st.session_state:
    st.session_state.datos_guardados = pd.DataFrame()

# --- 2. BARRA LATERAL (BÚSQUEDA) ---
st.sidebar.header("🔍 Búsqueda de Valores")
tickers_input = st.sidebar.text_input(
    "Tickers (separados por coma):", 
    "AAPL, MSFT, TSLA, NVDA, AMZN, META, GOOGL"
)
analizar_btn = st.sidebar.button("Analizar Mercado", type="primary")

# --- 3. MOTOR DE CÁLCULO (LA LÓGICA VALUE) ---
@st.cache_data
def obtener_datos_value(tickers_list):
    resultados = []
    
    for ticker in tickers_list:
        try:
            # Bajamos 2 años de datos para asegurar que hay suficientes para la SMA 200 y 52S
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty:
                continue
                
            close_prices = df['Close'].squeeze()
            high_prices = df['High'].squeeze()
            low_prices = df['Low'].squeeze()
            
            precio_actual = close_prices.iloc[-1]
            
            # 1. RSI Clásico
            rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
            
            # 2. SMA 200 y su Desviación
            sma200 = ta.trend.SMAIndicator(close_prices, window=200).sma_indicator().iloc[-1]
            desviacion_sma200 = ((precio_actual - sma200) / sma200) * 100
            
            # 3. Máximos y Mínimos de 52 Semanas (aprox 252 días de cotización)
            max_52s = high_prices.tail(252).max()
            min_52s = low_prices.tail(252).min()
            
            caida_desde_max = ((precio_actual - max_52s) / max_52s) * 100
            distancia_al_min = ((precio_actual - min_52s) / min_52s) * 100
            
            # Guardamos la fila de esta empresa
            resultados.append({
                "Ticker": ticker,
                "Precio ($)": round(float(precio_actual), 2),
                "RSI": round(float(rsi), 2),
                "Caída Max 52S (%)": round(float(caida_desde_max), 2),
                "Dist. Mínimo 52S (%)": round(float(distancia_al_min), 2),
                "Desviación SMA 200 (%)": round(float(desviacion_sma200), 2)
            })
        except Exception as e:
            pass # Si falla un ticker, seguimos con el siguiente
            
    return pd.DataFrame(resultados)

# --- 4. ACCIÓN DEL BOTÓN ---
if analizar_btn:
    lista_tickers = [t.strip().upper() for t in tickers_input.split(",")]
    with st.spinner("Buscando oportunidades en el mercado..."):
        st.session_state.datos_guardados = obtener_datos_value(lista_tickers)

# --- 5. INTERFAZ VISUAL DEL DASHBOARD ---
if not st.session_state.datos_guardados.empty:
    df = st.session_state.datos_guardados.copy()
    
    # Cálculos para las Tarjetas KPI
    empresas_menos20 = len(df[df['Caída Max 52S (%)'] <= -20])
    empresas_menos30 = len(df[df['Caída Max 52S (%)'] <= -30])
    
    # --- CABECERA Y KPIs (Top Derecha) ---
    col_titulo, col_kpi1, col_kpi2 = st.columns([3, 1, 1])
    
    with col_titulo:
        st.title("📊 Screener Value & Técnico")
        st.markdown("Busca empresas de calidad castigadas por el mercado.")
        
    with col_kpi1:
        st.info(f"📉 **{empresas_menos20}** Valores a -20%")
    with col_kpi2:
        st.error(f"🚨 **{empresas_menos30}** Valores a -30%")
        
    st.divider() # Una línea separadora elegante
    
    # --- FILTROS HORIZONTALES ---
    st.subheader("🛠️ Ajusta tus Criterios de Compra")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        filtro_caida = st.selectbox(
            "Filtrar por Caída desde Máximos (52S):",
            ["Mostrar Todos", "Solo caídas > 20%", "Solo caídas > 30%"]
        )
    with f_col2:
        max_rsi = st.slider("RSI Máximo (Ej. 40 para buscar sobreventa):", 0, 100, 100)
    with f_col3:
        filtro_sma = st.checkbox("Solo precios por DEBAJO de la SMA 200")

    # Aplicamos los filtros
    if filtro_caida == "Solo caídas > 20%":
        df = df[df['Caída Max 52S (%)'] <= -20]
    elif filtro_caida == "Solo caídas > 30%":
        df = df[df['Caída Max 52S (%)'] <= -30]
        
    df = df[df['RSI'] <= max_rsi]
    
    if filtro_sma:
        df = df[df['Desviación SMA 200 (%)'] < 0]
        
    # --- FORMATO CONDICIONAL (COLORES) ---
    def aplicar_colores(columna):
        # Esta función pinta las celdas según nuestras reglas de inversión
        colores = []
        for valor in columna:
            if columna.name == 'RSI':
                color = 'color: #00FF00' if valor < 30 else 'color: #FF0000' if valor > 70 else ''
            elif columna.name == 'Caída Max 52S (%)':
                color = 'color: #00FF00' if valor <= -30 else 'color: #90EE90' if valor <= -20 else 'color: #FF0000' if valor >= 0 else ''
            elif columna.name == 'Dist. Mínimo 52S (%)':
                color = 'color: #00FF00' if valor <= 5 else '' # Verde si está a menos del 5% de su mínimo
            elif columna.name == 'Desviación SMA 200 (%)':
                color = 'color: #00FF00' if valor < 0 else 'color: #FF0000'
            else:
                color = ''
            colores.append(color)
        return colores

    # Mostramos la tabla aplicando la magia de los colores
    st.write(f"**Resultados:** {len(df)} valores encontrados.")
    st.dataframe(
        df.style.apply(aplicar_colores, axis=0), 
        use_container_width=True,
        hide_index=True
    )
