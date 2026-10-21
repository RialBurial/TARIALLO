import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from itertools import combinations
from datetime import datetime, timedelta
import io

# Configuración general de la página
st.set_page_config(
    page_title="Cross-Returns Quant Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Suite Cuantitativa: Radar de Mercado y Retornos Cruzados")
st.markdown(
    "Flujo profesional en **4 Pasos**: **0º Radar del Mercado Tradicional** (encuentra qué empresa, índice o materia prima vigilar hoy), "
    "**1º Descubridor Universal** (encuentra sus predictores cruzados), **2º Escáner de Combinaciones** (crea las mejores parejas/tríos) y "
    "**3º Auditoría de los 3 Modelos** (señal, Z-Score, Régimen SMA 200, Kelly y niveles en precio)."
)

# ---------------------------------------------------------
# CATÁLOGO TRADICIONAL PARA PESTAÑA 0 (~65 ACTIVOS SIN CRIPTO)
# ---------------------------------------------------------
TRADITIONAL_SCREENER_UNIVERSE = {
    # Índices Bursátiles
    "SPY": ("S&P 500 ETF", "Índices Bursátiles"),
    "QQQ": ("Nasdaq 100 ETF", "Índices Bursátiles"),
    "DIA": ("Dow Jones Industrial ETF", "Índices Bursátiles"),
    "IWM": ("Russell 2000 Small Caps", "Índices Bursátiles"),
    "^GDAXI": ("DAX 40 Alemania", "Índices Bursátiles"),
    "^STOXX50E": ("Euro Stoxx 50 Europa", "Índices Bursátiles"),
    "^IBEX": ("IBEX 35 España", "Índices Bursátiles"),
    "^N225": ("Nikkei 225 Japón", "Índices Bursátiles"),
    "EWZ": ("MSCI Brasil ETF", "Índices Bursátiles"),
    "EEM": ("Mercados Emergentes ETF", "Índices Bursátiles"),
    # ETFs Sectoriales Wall Street
    "XLK": ("Sector Tecnología", "ETFs Sectoriales"),
    "SMH": ("Sector Semiconductores", "ETFs Sectoriales"),
    "XLF": ("Sector Financiero / Bancos", "ETFs Sectoriales"),
    "XLE": ("Sector Energía", "ETFs Sectoriales"),
    "XLV": ("Sector Salud", "ETFs Sectoriales"),
    "XLI": ("Sector Industrial", "ETFs Sectoriales"),
    "XLY": ("Consumo Discrecional", "ETFs Sectoriales"),
    "XLP": ("Consumo Básico / Defensivo", "ETFs Sectoriales"),
    "XLU": ("Utilidades / Eléctricas", "ETFs Sectoriales"),
    "XLB": ("Materiales Básicos", "ETFs Sectoriales"),
    "XLRE": ("Sector Inmobiliario (REITs)", "ETFs Sectoriales"),
    "XLC": ("Servicios de Comunicación", "ETFs Sectoriales"),
    "ITA": ("Aeroespacial y Defensa ETF", "ETFs Sectoriales"),
    "XBI": ("Biotecnología ETF", "ETFs Sectoriales"),
    # Acciones Líderes Tecnología y Semis
    "NVDA": ("Nvidia Corp", "Acciones: Tecnología"),
    "AAPL": ("Apple Inc", "Acciones: Tecnología"),
    "MSFT": ("Microsoft Corp", "Acciones: Tecnología"),
    "AMZN": ("Amazon.com", "Acciones: Tecnología"),
    "META": ("Meta Platforms", "Acciones: Tecnología"),
    "GOOGL": ("Alphabet / Google", "Acciones: Tecnología"),
    "TSLA": ("Tesla Inc", "Acciones: Tecnología"),
    "AVGO": ("Broadcom Inc", "Acciones: Tecnología"),
    "AMD": ("Advanced Micro Devices", "Acciones: Tecnología"),
    "TSM": ("Taiwan Semiconductor", "Acciones: Tecnología"),
    "ASML": ("ASML Holding", "Acciones: Tecnología"),
    "NFLX": ("Netflix Inc", "Acciones: Tecnología"),
    "CRM": ("Salesforce Inc", "Acciones: Tecnología"),
    "ORCL": ("Oracle Corp", "Acciones: Tecnología"),
    # Acciones Líderes Finanzas, Salud, Consumo e Industria
    "JPM": ("JPMorgan Chase", "Acciones: Finanzas y Salud"),
    "BAC": ("Bank of America", "Acciones: Finanzas y Salud"),
    "GS": ("Goldman Sachs", "Acciones: Finanzas y Salud"),
    "V": ("Visa Inc", "Acciones: Finanzas y Salud"),
    "MA": ("Mastercard Inc", "Acciones: Finanzas y Salud"),
    "BRK-B": ("Berkshire Hathaway", "Acciones: Finanzas y Salud"),
    "LLY": ("Eli Lilly (Farmacéutica)", "Acciones: Finanzas y Salud"),
    "UNH": ("UnitedHealth Group", "Acciones: Finanzas y Salud"),
    "JNJ": ("Johnson & Johnson", "Acciones: Finanzas y Salud"),
    "WMT": ("Walmart Inc", "Acciones: Consumo e Industria"),
    "COST": ("Costco Wholesale", "Acciones: Consumo e Industria"),
    "HD": ("Home Depot", "Acciones: Consumo e Industria"),
    "MCD": ("McDonald's Corp", "Acciones: Consumo e Industria"),
    "KO": ("Coca-Cola Co", "Acciones: Consumo e Industria"),
    "CAT": ("Caterpillar Inc", "Acciones: Consumo e Industria"),
    "GE": ("GE Aerospace", "Acciones: Consumo e Industria"),
    "XOM": ("Exxon Mobil", "Acciones: Consumo e Industria"),
    "CVX": ("Chevron Corp", "Acciones: Consumo e Industria"),
    # Materias Primas y Renta Fija
    "GLD": ("Oro ETF", "Materias Primas y Bonos"),
    "SLV": ("Plata ETF", "Materias Primas y Bonos"),
    "CPER": ("Cobre ETF", "Materias Primas y Bonos"),
    "USO": ("Petróleo WTI ETF", "Materias Primas y Bonos"),
    "UNG": ("Gas Natural ETF", "Materias Primas y Bonos"),
    "DBA": ("Agricultura Materias Primas", "Materias Primas y Bonos"),
    "URA": ("Uranio y Energía Nuclear ETF", "Materias Primas y Bonos"),
    "TLT": ("Bonos Tesoro EE.UU. +20 Años", "Materias Primas y Bonos"),
    "IEF": ("Bonos Tesoro EE.UU. 7-10 Años", "Materias Primas y Bonos"),
    "HYG": ("Bonos High Yield Corporativos", "Materias Primas y Bonos"),
    "UUP": ("Índice Dólar Bullish ETF", "Materias Primas y Bonos")
}

# ---------------------------------------------------------
# CATÁLOGO MAESTRO UNIVERSAL (PARA PESTAÑA 1: DESCUBRIDOR DE PREDICTORES)
# ---------------------------------------------------------
MASTER_UNIVERSE = {
    **TRADITIONAL_SCREENER_UNIVERSE,
    "^VIX": ("Índice de Volatilidad S&P 500", "Macro y Renta Fija"),
    "SHY": ("Bonos Tesoro EE.UU. 1-3 Años", "Macro y Renta Fija"),
    "LQD": ("Bonos Corporativos Grado Inversión", "Macro y Renta Fija"),
    "EURUSD=X": ("Euro / Dólar", "Materias Primas y Forex"),
    "USDJPY=X": ("Dólar / Yen Japonés", "Materias Primas y Forex"),
    "GBPUSD=X": ("Libra / Dólar", "Materias Primas y Forex"),
    "BTC-USD": ("Bitcoin", "Criptoactivos"),
    "ETH-USD": ("Ethereum", "Criptoactivos"),
    "SOL-USD": ("Solana", "Criptoactivos"),
    "BNB-USD": ("Binance Coin", "Criptoactivos")
}

# ---------------------------------------------------------
# FUNCIÓN PARA EXPORTAR TABLAS A EXCEL (.XLSX)
# ---------------------------------------------------------
def convert_dfs_to_excel(sheets_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets_dict.items():
            clean_name = sheet_name[:31]
            df_sheet.to_excel(writer, index=False, sheet_name=clean_name)
    return output.getvalue()

# ---------------------------------------------------------
# BARRA LATERAL: CONFIGURACIÓN DEL FLUJO DE TRABAJO
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. Selección de Activos")

presets = {
    "Personalizado / Usar Descubridor": {
        "target": "SPY",
        "predictors": "QQQ, IWM, TLT, GLD, ^VIX, XLF, SMH"
    },
    "Índices EE.UU. + Sectores y Macro": {
        "target": "SPY",
        "predictors": "QQQ, IWM, TLT, GLD, ^VIX, XLF, XLK, XLE, SMH"
    },
    "Gigantes Tecnológicos (Acciones)": {
        "target": "NVDA",
        "predictors": "AMD, TSM, MSFT, AAPL, META, GOOGL, AMZN, QQQ, SMH"
    },
    "Índices Globales y Divisas": {
        "target": "^GDAXI",
        "predictors": "SPY, ^STOXX50E, ^IBEX, ^N225, EURUSD=X, GLD, USO"
    },
    "Cripto": {
        "target": "ETH-USD",
        "predictors": "BNB-USD, SOL-USD, BTC-USD, LINK-USD, AVAX-USD"
    }
}

preset_choice = st.sidebar.selectbox("Plantillas rápidas de mercado:", list(presets.keys()))

if "custom_target_str" not in st.session_state:
    st.session_state.custom_target_str = presets[preset_choice]["target"]
if "custom_predictors_str" not in st.session_state:
    st.session_state.custom_predictors_str = presets[preset_choice]["predictors"]
if "last_preset" not in st.session_state:
    st.session_state.last_preset = preset_choice

if st.session_state.last_preset != preset_choice:
    st.session_state.custom_target_str = presets[preset_choice]["target"]
    st.session_state.custom_predictors_str = presets[preset_choice]["predictors"]
    st.session_state.last_preset = preset_choice

target_ticker = st.sidebar.text_input(
    "🎯 Activo Objetivo a Operar (X):",
    key="custom_target_str",
    help="Escribe cualquier ticker de Yahoo Finance o envíalo con 1 clic desde la Pestaña 0."
).strip().upper()

predictors_input = st.sidebar.text_area(
    "🔗 Universo de Activos Candidatos (Y) para Pestaña 2:",
    key="custom_predictors_str",
    help="Puedes escribirlos a mano separados por coma o enviarlos automáticamente desde la Pestaña 1 (Descubridor Universal)."
)

candidate_tickers = [
    t.strip().upper() for t in predictors_input.split(",")
    if t.strip() and t.strip().upper() != target_ticker
]

st.sidebar.header("⏱️ 2. Parámetros y Entrenamiento")

horizon = st.sidebar.slider(
    "Horizonte de Retorno / Predicción (Días):",
    min_value=1, max_value=30, value=7,
    help="Ventana de días para medir retornos pasados y proyectar el retorno futuro (ej. 7 días)."
)

years_back = st.sidebar.slider(
    "Histórico de datos (Años):",
    min_value=1, max_value=10, value=5
)

train_pct = st.sidebar.slider(
    "División Entrenamiento / Prueba (Train %):",
    min_value=50, max_value=90, value=70, step=5,
    help="El primer X% de las fechas entrena la fórmula; el % más reciente valida si funciona hoy (Out-of-Sample)."
)

training_mode = st.sidebar.radio(
    "🧠 Modo de Aprendizaje del Modelo:",
    options=["Estático (Clásico Train/Test)", "Ventana Móvil Adaptativa (Rolling)"],
    index=0,
    help="Estático congela los pesos aprendidos en Train. Ventana Móvil re-entrena el modelo periódicamente con los últimos N días."
)

rolling_window_days = 252
if training_mode == "Ventana Móvil Adaptativa (Rolling)":
    rolling_window_days = st.sidebar.slider(
        "Tamaño de Ventana Móvil (Días bursátiles):",
        min_value=90, max_value=504, value=252, step=21,
        help="252 días equivale a 1 año bursátil completo de memoria móvil."
    )

st.sidebar.header("🛡️ 3. Costes, Umbral y Régimen")

commission_pct = st.sidebar.number_input(
    "Comisión + Spread medio por operación (%):",
    min_value=0.00, max_value=2.00, value=0.05, step=0.01, format="%.2f",
    help="Coste medio estimado por operación (0.05% equivale a 5 puntos básicos, estándar en ETFs/acciones líquidas)."
)

min_threshold_pct = st.sidebar.number_input(
    "Umbral mínimo de predicción para operar (%):",
    min_value=0.00, max_value=10.00, value=0.10, step=0.05, format="%.2f",
    help="Escribe el número mínimo de retorno esperado para abrir trade (ej. 0.10%). Si la predicción es menor, el modelo se queda en liquidez (NEUTRAL)."
)

regime_filter_mode = st.sidebar.selectbox(
    "Filtro de Régimen Macro (SMA 200 de X):",
    options=[
        "Desactivado (Operar Long y Short libremente)",
        "Solo operar a favor de la SMA 200 (Tendencia Mayor)"
    ],
    index=0
)

# ---------------------------------------------------------
# MOTOR DE DATOS ROBUSTO Y EXPONENTE DE HURST
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_aligned_market_data(target_sym, candidate_syms, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    target_raw = yf.download(target_sym, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if target_raw.empty:
        return pd.DataFrame()

    if isinstance(target_raw.columns, pd.MultiIndex):
        target_series = target_raw["Close"][target_sym] if target_sym in target_raw["Close"].columns else target_raw["Close"].iloc[:, 0]
    else:
        target_series = target_raw["Close"]

    target_series = target_series.dropna()
    if target_series.empty:
        return pd.DataFrame()

    master_df = pd.DataFrame({target_sym: target_series})
    master_df.index = pd.to_datetime(master_df.index).tz_localize(None)

    unique_cands = [c for c in dict.fromkeys(candidate_syms) if c != target_sym]
    chunk_size = 15

    for i in range(0, len(unique_cands), chunk_size):
        chunk = unique_cands[i:i + chunk_size]
        try:
            raw = yf.download(chunk, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                closes_chunk = raw["Close"].copy()
            else:
                closes_chunk = pd.DataFrame({chunk[0]: raw["Close"]})

            closes_chunk.index = pd.to_datetime(closes_chunk.index).tz_localize(None)
            combined_idx = master_df.index.union(closes_chunk.index).sort_values()
            aligned_chunk = closes_chunk.reindex(combined_idx).ffill().reindex(master_df.index)

            for col in aligned_chunk.columns:
                if aligned_chunk[col].notna().mean() >= 0.75:
                    master_df[col] = aligned_chunk[col]
        except Exception:
            continue

    master_df = master_df.ffill().dropna()
    return master_df

def calculate_hurst_exponent(price_series, max_lag=20):
    """Calcula el Exponente de Hurst (H) para medir memoria estadística:
    H > 0.55 = Persistencia Tendencial | H < 0.45 = Reversión a la Media | ~0.50 = Ruido Aleatorio"""
    s = np.log(price_series.dropna().values)
    if len(s) < 100:
        return 0.50
    lags = range(2, max_lag + 1)
    tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
    if any(t <= 0 for t in tau):
        return 0.50
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(np.clip(poly[0], 0.05, 0.95))

@st.cache_data(ttl=3600, show_spinner=False)
def run_market_screener(categories_selected, z_window_days=7):
    tickers_to_scan = [
        t for t, info in TRADITIONAL_SCREENER_UNIVERSE.items()
        if info[1] in categories_selected
    ]
    if "SPY" not in tickers_to_scan:
        tickers_to_scan.append("SPY")

    df_closes = load_aligned_market_data("SPY", tickers_to_scan, 2)
    if df_closes.empty or "SPY" not in df_closes.columns:
        return []

    spy_ret_6m = float((df_closes["SPY"].iloc[-1] / df_closes["SPY"].iloc[-126]) - 1.0) * 100.0 if len(df_closes) >= 126 else 0.0

    rows = []
    for sym in tickers_to_scan:
        if sym not in df_closes.columns:
            continue
        series = df_closes[sym].dropna()
        if len(series) < 200:
            continue

        curr_price = float(series.iloc[-1])
        sma50 = float(series.rolling(50).mean().iloc[-1])
        sma200 = float(series.rolling(200).mean().iloc[-1])
        dist_sma200 = ((curr_price / sma200) - 1.0) * 100.0

        if curr_price >= sma200 and curr_price >= sma50:
            regime_str = "🟢 Alcista Fuerte (>SMA50 y 200)"
        elif curr_price >= sma200 and curr_price < sma50:
            regime_str = "🟡 En Pullback (>SMA200, <SMA50)"
        else:
            regime_str = "🔴 Bajista (<SMA200)"

        # Retorno y Z-Score en la ventana elegida (ej. 7 días)
        rets_nd = series.pct_change(z_window_days).dropna()
        curr_ret_nd = float(rets_nd.iloc[-1] * 100.0)
        mean_nd = float(rets_nd.mean())
        std_nd = float(rets_nd.std())
        z_score = float((rets_nd.iloc[-1] - mean_nd) / std_nd) if std_nd > 1e-9 else 0.0

        # Rendimiento a 6 meses (126 sesiones), Sharpe 6m y Alpha vs SPY
        daily_rets_6m = series.pct_change().dropna().iloc[-126:]
        ret_6m = float((series.iloc[-1] / series.iloc[-126]) - 1.0) * 100.0
        alpha_spy_6m = ret_6m - spy_ret_6m
        std_d = float(daily_rets_6m.std())
        sharpe_6m = float((daily_rets_6m.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-9 else 0.0

        # Exponente de Hurst (último año 252 sesiones)
        hurst = calculate_hurst_exponent(series.iloc[-252:])
        if hurst >= 0.55:
            hurst_type = f"📈 Tendencial ({hurst:.2f})"
        elif hurst <= 0.45:
            hurst_type = f"🔄 Péndulo / Reversión ({hurst:.2f})"
        else:
            hurst_type = f"🎲 Ruido Aleatorio ({hurst:.2f})"

        # Ratio de Compresión de Volatilidad (Vol 10d / Vol 126d)
        vol_10d = float(daily_rets_6m.iloc[-10:].std())
        vol_ratio = float(vol_10d / std_d) if std_d > 1e-9 else 1.0

        # Puntuaciones por perfil (0 a 100)
        dip_score = max(0.0, min(100.0, (50.0 if curr_price >= sma200 else 10.0) + (-z_score * 20.0) + (sharpe_6m * 10.0)))
        mom_score = max(0.0, min(100.0, (40.0 if curr_price >= sma50 >= sma200 else 0.0) + (sharpe_6m * 20.0) + (alpha_spy_6m * 1.0)))
        hurst_score = max(0.0, min(100.0, abs(hurst - 0.50) * 250.0 + max(0.0, sharpe_6m * 15.0)))
        squeeze_score = max(0.0, min(100.0, (1.2 - vol_ratio) * 80.0 + (20.0 if curr_price >= sma200 else 0.0)))

        # Diagnóstico automático
        if curr_price >= sma200 and z_score <= -1.2:
            diag = "🏹 BUY THE DIP: Alcista de fondo con sobreventa estadística a corto plazo"
        elif curr_price >= sma50 >= sma200 and sharpe_6m >= 1.2 and z_score < 1.8:
            diag = "🚀 LÍDER MOMENTUM: Tendencia institucional limpia con Alpha positivo"
        elif vol_ratio <= 0.65:
            diag = "💥 COMPRESIÓN EXTREMA: Muelle de volatilidad comprimido (explosión inminente)"
        elif hurst >= 0.57:
            diag = "📈 PERSISTENCIA ALTA: Activo de fuerte memoria tendencial (Hurst alto)"
        elif hurst <= 0.43:
            diag = "🔄 PÉNDULO ESTADÍSTICO: Activo de fuerte reversión a la media (Hurst bajo)"
        elif z_score >= 2.0:
            diag = "⚠️ SOBRECOMPRA EXTREMA: Estirado > +2σ a corto plazo (riesgo de corrección)"
        else:
            diag = "〰️ En Rango Normal / Sin extremo estadístico"

        name_desc, cat_name = TRADITIONAL_SCREENER_UNIVERSE.get(sym, (sym, "Mercado Tradicional"))
        rows.append({
            "Ticker": sym,
            "Nombre": name_desc,
            "Categoría": cat_name,
            "Precio Actual": round(curr_price, 2),
            "Régimen Técnico": regime_str,
            "Dist. SMA 200 (%)": round(dist_sma200, 1),
            f"Retorno ({z_window_days}d %)": round(curr_ret_nd, 2),
            f"Z-Score ({z_window_days}d σ)": round(z_score, 2),
            "Sharpe (6m)": round(sharpe_6m, 2),
            "Alpha vs SPY (6m %)": round(alpha_spy_6m, 1),
            "Hurst (H)": round(hurst, 2),
            "Personalidad (Hurst)": hurst_type,
            "Compresión Vol (10d/6m)": round(vol_ratio, 2),
            "Diagnóstico Cuantitativo": diag,
            "_dip_score": round(dip_score, 1),
            "_mom_score": round(mom_score, 1),
            "_hurst_score": round(hurst_score, 1),
            "_squeeze_score": round(squeeze_score, 1),
            "_above_sma200": curr_price >= sma200,
            "_above_sma50": curr_price >= sma50
        })
    return rows

def evaluate_strategy(
    df_returns, df_closes, target_col, feature_cols, horizon_days, train_ratio,
    fee_pct=0.05, threshold_pct=0.10, is_rolling=False, roll_win=252, use_sma200_filter=False
):
    df = pd.DataFrame(index=df_returns.index)
    df["Target_Future"] = df_returns[target_col].shift(-horizon_days)

    for col in feature_cols:
        df[col] = df_returns[col]

    sma200_series = df_closes[target_col].rolling(200, min_periods=50).mean()
    above_sma200 = (df_closes[target_col] >= sma200_series).reindex(df_returns.index).fillna(True)

    latest_features_series = df_returns[feature_cols].iloc[-1]
    feat_means = df_returns[feature_cols].mean()
    feat_stds = df_returns[feature_cols].std().replace(0, 1e-9)
    latest_feature_zscores = ((latest_features_series - feat_means) / feat_stds).to_dict()

    df_model = df.dropna().copy()
    split_idx = int(len(df_model) * (train_ratio / 100.0))

    X_all = df_model[feature_cols].values
    y_all = df_model["Target_Future"].values
    n_samples = len(df_model)

    if not is_rolling:
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        weights = model.coef_.copy()
        intercept = float(model.intercept_)
        preds = intercept + X_all.dot(weights)
        live_pred = float(intercept + latest_features_series.values.dot(weights))
    else:
        X_train_init = X_all[:split_idx]
        y_train_init = y_all[:split_idx]
        base_model = LinearRegression()
        base_model.fit(X_train_init, y_train_init)
        preds = np.zeros(n_samples)
        preds[:split_idx] = base_model.intercept_ + X_train_init.dot(base_model.coef_)

        step_retrain = 10
        curr_w = base_model.coef_.copy()
        curr_b = float(base_model.intercept_)

        for t in range(split_idx, n_samples, step_retrain):
            train_end = max(30, t - horizon_days)
            train_start = max(0, train_end - roll_win)
            if train_end - train_start >= 25:
                m_roll = LinearRegression()
                m_roll.fit(X_all[train_start:train_end], y_all[train_start:train_end])
                curr_w = m_roll.coef_.copy()
                curr_b = float(m_roll.intercept_)
            end_block = min(t + step_retrain, n_samples)
            preds[t:end_block] = curr_b + X_all[t:end_block].dot(curr_w)

        final_start = max(0, n_samples - roll_win)
        m_live = LinearRegression()
        m_live.fit(X_all[final_start:], y_all[final_start:])
        weights = m_live.coef_.copy()
        intercept = float(m_live.intercept_)
        live_pred = float(intercept + latest_features_series.values.dot(weights))

    df_model["Pred"] = preds

    pred_mean = float(np.mean(preds[:split_idx]))
    pred_std = float(np.std(preds[:split_idx]))
    signal_zscore = (live_pred - pred_mean) / pred_std if pred_std > 1e-9 else 0.0

    thresh_dec = threshold_pct / 100.0
    fee_dec = fee_pct / 100.0

    signals = np.where(
        df_model["Pred"] > thresh_dec, 1.0,
        np.where(df_model["Pred"] < -thresh_dec, -1.0, 0.0)
    )

    if use_sma200_filter:
        bull_mask = above_sma200.loc[df_model.index].values
        signals = np.where((signals > 0) & (~bull_mask), 0.0, signals)
        signals = np.where((signals < 0) & bull_mask, 0.0, signals)

    df_model["Signal"] = signals

    active_mask = np.abs(signals) > 0
    df_model["Strategy_Return"] = (signals * df_model["Target_Future"]) - (active_mask * fee_dec)

    df_model["Equity_Curve"] = (df_model["Strategy_Return"] / horizon_days).cumsum() * 100.0
    df_model["BuyHold_Curve"] = (df_model["Target_Future"] / horizon_days).cumsum() * 100.0

    train_strat = df_model["Strategy_Return"].iloc[:split_idx]
    test_strat = df_model["Strategy_Return"].iloc[split_idx:]
    train_active = train_strat[df_model["Signal"].iloc[:split_idx] != 0]
    test_active = test_strat[df_model["Signal"].iloc[split_idx:] != 0]

    recent_strat = test_strat.iloc[-60:] if len(test_strat) >= 10 else test_strat
    recent_active = recent_strat[recent_strat != 0]

    def get_metrics(full_series, active_series):
        if len(full_series) == 0 or len(active_series) == 0:
            return {
                "mean_pct": 0.0, "win_rate": 0.0, "sharpe": 0.0,
                "total_pnl": 0.0, "max_dd": 0.0, "trades_pct": 0.0, "half_kelly_pct": 0.0
            }
        mean_ret = float(active_series.mean() * 100.0)
        win_rate = float((active_series > 0).mean() * 100.0)
        std_ret = float(full_series.std())
        sharpe = float((full_series.mean() / std_ret) * np.sqrt(252.0 / horizon_days)) if std_ret > 1e-9 else 0.0
        eq = (full_series / horizon_days).cumsum() * 100.0
        running_max = np.maximum.accumulate(np.insert(eq.values, 0, 0.0))[1:]
        dd = eq.values - running_max
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
        total_pnl = float(eq.iloc[-1]) if len(eq) > 0 else 0.0
        trades_pct = float((len(active_series) / len(full_series)) * 100.0)

        wins = active_series[active_series > 0]
        losses = active_series[active_series < 0]
        if len(wins) > 0 and len(losses) > 0:
            w_prob = len(wins) / len(active_series)
            payoff_ratio = float(wins.mean() / abs(losses.mean()))
            full_kelly = w_prob - ((1.0 - w_prob) / payoff_ratio) if payoff_ratio > 0 else 0.0
            half_kelly = max(0.0, min(full_kelly * 0.5, 0.50)) * 100.0
        elif len(wins) > 0:
            half_kelly = 25.0
        else:
            half_kelly = 0.0

        return {
            "mean_pct": mean_ret,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "total_pnl": total_pnl,
            "max_dd": max_dd,
            "trades_pct": trades_pct,
            "half_kelly_pct": half_kelly
        }

    target_vol_pct = float(df_returns[target_col].iloc[split_idx:].std() * 100.0)

    return {
        "weights": weights,
        "intercept": intercept,
        "features": feature_cols,
        "train_metrics": get_metrics(train_strat, train_active),
        "test_metrics": get_metrics(test_strat, test_active),
        "recent_metrics": get_metrics(recent_strat, recent_active),
        "df_model": df_model,
        "split_date": df_model.index[split_idx],
        "live_pred_pct": live_pred * 100.0,
        "signal_zscore": signal_zscore,
        "latest_features": latest_features_series * 100.0,
        "latest_feature_zscores": latest_feature_zscores,
        "target_vol_pct": target_vol_pct
    }

# Descarga inicial para el activo objetivo y los candidatos actuales de la barra lateral
with st.spinner(f"Descargando cotizaciones alineadas al calendario de `{target_ticker}`..."):
    closes_df = load_aligned_market_data(target_ticker, candidate_tickers, years_back)

if closes_df.empty or target_ticker not in closes_df.columns:
    st.error(f"❌ No se encontraron datos para el activo objetivo `{target_ticker}`. Revisa el símbolo en Yahoo Finance.")
    st.stop()

valid_candidates = [c for c in candidate_tickers if c in closes_df.columns]
missing_tickers = set(candidate_tickers) - set(valid_candidates)
if missing_tickers:
    st.warning(f"⚠️ No se pudieron descargar o alinear datos para: {', '.join(missing_tickers)}. Se omitirán del análisis.")

if len(closes_df) < horizon * 10:
    st.error("❌ No hay suficiente histórico para este horizonte. Reduce los días de horizonte o amplía los años.")
    st.stop()

returns_df = closes_df.pct_change(horizon).dropna()
current_target_price = float(closes_df[target_ticker].iloc[-1])

sma200_val = float(closes_df[target_ticker].rolling(200, min_periods=50).mean().iloc[-1])
is_bull_regime = current_target_price >= sma200_val
sma200_dist_pct = ((current_target_price / sma200_val) - 1.0) * 100.0

is_rolling_mode = (training_mode == "Ventana Móvil Adaptativa (Rolling)")
use_sma_filter = (regime_filter_mode != "Desactivado (Operar Long y Short libremente)")

ar_res = evaluate_strategy(
    returns_df, closes_df, target_ticker, [target_ticker], horizon, train_pct,
    commission_pct, min_threshold_pct, is_rolling_mode, rolling_window_days, use_sma_filter
)
ar_test_mean = ar_res["test_metrics"]["mean_pct"]
ar_test_wr = ar_res["test_metrics"]["win_rate"]
ar_test_sharpe = ar_res["test_metrics"]["sharpe"]
ar_test_dd = ar_res["test_metrics"]["max_dd"]

# ---------------------------------------------------------
# FUNCIONES DE CACHÉ PARA PESTAÑA 1 Y PESTAÑA 2
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def run_universal_discovery(target, years, horizon_days, train_ratio, categories_selected, only_pos_test, fee_pct, thresh_pct, is_roll, roll_w, sma_flt):
    universe_tickers = [
        t for t, info in MASTER_UNIVERSE.items()
        if info[1] in categories_selected and t != target
    ]
    df_closes = load_aligned_market_data(target, universe_tickers, years)
    if df_closes.empty or target not in df_closes.columns:
        return [], 0

    df_rets = df_closes.pct_change(horizon_days).dropna()
    ar_base = evaluate_strategy(df_rets, df_closes, target, [target], horizon_days, train_ratio, fee_pct, thresh_pct, is_roll, roll_w, sma_flt)
    ar_ref_mean = ar_base["test_metrics"]["mean_pct"]

    discovered = []
    evaluated_count = 0

    for cand in universe_tickers:
        if cand not in df_rets.columns:
            continue
        evaluated_count += 1
        res = evaluate_strategy(df_rets, df_closes, target, [cand], horizon_days, train_ratio, fee_pct, thresh_pct, is_roll, roll_w, sma_flt)
        w = float(res["weights"][0])
        train_m = res["train_metrics"]["mean_pct"]
        test_m = res["test_metrics"]["mean_pct"]
        test_wr = res["test_metrics"]["win_rate"]
        test_sh = res["test_metrics"]["sharpe"]
        test_dd = res["test_metrics"]["max_dd"]
        diff_ar = test_m - ar_ref_mean
        cand_z = res["latest_feature_zscores"].get(cand, 0.0)

        if only_pos_test and test_m <= 0:
            continue

        name_desc, cat_name = MASTER_UNIVERSE.get(cand, (cand, "Otro"))
        discovered.append({
            "Ticker": cand,
            "Nombre del Activo": name_desc,
            "Categoría": cat_name,
            "Relación Aprendida": "📈 Tendencia (+)" if w > 0 else "🔄 Reversión (-)",
            "Peso (w)": round(w, 4),
            "Z-Score Hoy (σ)": round(cand_z, 2),
            "Retorno Neto Trade (Test %)": round(test_m, 3),
            "Diferencial vs AR (%)": round(diff_ar, 3),
            "Supera a AR": "✅ Sí" if diff_ar > 0 else "No",
            "Win Rate (Test %)": round(test_wr, 1),
            "Sharpe (Test)": round(test_sh, 2),
            "Max Drawdown (Test %)": round(test_dd, 2),
            "Retorno Neto (Train %)": round(train_m, 3)
        })
    return discovered, evaluated_count

@st.cache_data(ttl=3600, show_spinner=False)
def run_full_scanner(returns_data, closes_data, target, candidates, horizon_days, train_ratio, max_r, only_rs, filter_clones, ar_mean_ref, fee_pct, thresh_pct, is_roll, roll_w, sma_flt):
    combos = []
    for r in range(1, max_r + 1):
        combos.extend(list(combinations(candidates, r)))

    corr_matrix = returns_data[candidates].corr().abs() if len(candidates) > 1 else pd.DataFrame()

    rows = []
    clones_filtered = 0

    for combo in combos:
        combo_list = list(combo)

        if filter_clones and len(combo_list) >= 2:
            is_clone = False
            for a, b in combinations(combo_list, 2):
                if corr_matrix.loc[a, b] > 0.90:
                    is_clone = True
                    break
            if is_clone:
                clones_filtered += 1
                continue

        res = evaluate_strategy(returns_data, closes_data, target, combo_list, horizon_days, train_ratio, fee_pct, thresh_pct, is_roll, roll_w, sma_flt)
        w = res["weights"]
        is_rs = (any(val > 0 for val in w) and any(val < 0 for val in w))

        if only_rs and not is_rs:
            continue

        if res["test_metrics"]["mean_pct"] > 0:
            weights_str = ", ".join([f"{f} ({val:+.2f})" for f, val in zip(combo_list, w)])
            pred_now = res["live_pred_pct"]
            sig_z = res["signal_zscore"]
            diff_vs_ar = res["test_metrics"]["mean_pct"] - ar_mean_ref

            if pred_now > thresh_pct:
                sig_str = f"🟢 LONG ({pred_now:+.2f}%)"
            elif pred_now < -thresh_pct:
                sig_str = f"🔴 SHORT ({pred_now:+.2f}%)"
            else:
                sig_str = f"⚪ NEUTRAL ({pred_now:+.2f}%)"

            rows.append({
                "combo_list": combo_list,
                "Activos Predictores": " + ".join(combo_list),
                "Fuerza Relativa (+/-)": "✅ Sí" if is_rs else "No",
                "Pesos (w)": weights_str,
                "Retorno Neto Trade (Test %)": round(res["test_metrics"]["mean_pct"], 3),
                "Diferencial vs AR (%)": round(diff_vs_ar, 3),
                "Win Rate (Test %)": round(res["test_metrics"]["win_rate"], 1),
                "Sharpe (Test)": round(res["test_metrics"]["sharpe"], 2),
                "Max Drawdown (Test %)": round(res["test_metrics"]["max_dd"], 2),
                "Win Rate 90d (%)": round(res["recent_metrics"]["win_rate"], 1),
                "Z-Score Señal Hoy (σ)": round(sig_z, 2),
                f"Señal Hoy ({horizon_days}d)": sig_str
            })
    return rows, len(combos), clones_filtered

# ---------------------------------------------------------
# ESTRUCTURA DE 4 PESTAÑAS (PESTAÑA 0 AL INICIO)
# ---------------------------------------------------------
tab_screener0, tab_discovery, tab_scanner, tab_models = st.tabs([
    "🎯 0. Radar de Oportunidades del Mercado (Sin Cripto)",
    "🌍 1. Descubridor Universal de Activos",
    "🔍 2. Escáner de Combinaciones (Radar de Parejas)",
    "📊 3. Auditoría de los 3 Modelos"
])

# =========================================================
# PESTAÑA 0: RADAR DE OPORTUNIDADES DEL MERCADO TRADICIONAL
# =========================================================
with tab_screener0:
    st.subheader("🎯 Radar Cuantitativo del Mercado Tradicional (Acciones, Índices, Sectores y Materias Primas)")
    st.markdown(
        "Esta pestaña peina **~65 activos líderes del mercado real (0% Cripto)** mediante 4 lentes estadísticas "
        "(**Z-Score**, **Ratio de Sharpe / Alpha**, **Exponente de Hurst $H$** y **Compresión de Volatilidad**) "
        "para decirte qué empresas, índices o materias primas presentan hoy la mejor configuración operativa."
    )

    trad_categories = list(dict.fromkeys([info[1] for info in TRADITIONAL_SCREENER_UNIVERSE.values()]))

    col_s0_1, col_s0_2, col_s0_3 = st.columns([1.5, 1.5, 1])
    hunting_profile = col_s0_1.selectbox(
        "🔭 Lente Estadística / Estrategia de Búsqueda:",
        [
            "🏹 1. Cazador de Gomas Estiradas (Buy the Dip en Tendencia Alcista)",
            "🚀 2. Líderes de Momentum Puro y Alpha Institucional",
            "🧬 3. Detector de Memoria Estadística (Exponente de Hurst Fuerte)",
            "💥 4. Compresión de Volatilidad (Squeeze antes de Explosión)",
            "🛠️ 5. Vista Completa Personalizada (Todos los Activos)"
        ],
        index=0
    )
    selected_trad_cats = col_s0_2.multiselect(
        "Sectores / Categorías a peinar:",
        options=trad_categories,
        default=trad_categories
    )
    z_win_screener = col_s0_3.slider(
        "Ventana para medir Z-Score (Días):",
        min_value=3, max_value=20, value=horizon,
        help="Días usados para medir si un activo está sobrevendido (-Z) o sobrecomprado (+Z) a corto plazo."
    )

    if st.button("🔎 Peinar el Mercado Tradicional Ahora", type="primary"):
        with st.spinner("Calculando Z-Scores, Sharpe, Alpha, Exponente de Hurst y Compresión en ~65 activos tradicionales..."):
            st.session_state.screener0_data = run_market_screener(selected_trad_cats, z_win_screener)
            st.session_state.screener0_zwin = z_win_screener

    if "screener0_data" in st.session_state and st.session_state.screener0_data:
        raw_s0_df = pd.DataFrame(st.session_state.screener0_data)
        z_col_name = f"Z-Score ({st.session_state.screener0_zwin}d σ)"

        # Aplicar filtros y ordenación según el Perfil de Caza elegido
        df_s0 = raw_s0_df.copy()
        if hunting_profile.startswith("🏹 1."):
            st.info(
                "**🏹 Lente Activa — Buy the Dip Cuantitativo:** Muestra activos que están en **Tendencia Alcista Mayor (Precio > SMA 200)** "
                "pero que acaban de sufrir un castigo estadístico a corto plazo (**Z-Score negativo**), ofreciendo una entrada barata a favor de la tendencia."
            )
            df_s0["Score Perfil (0-100)"] = df_s0["_dip_score"]
            df_s0 = df_s0[df_s0["_above_sma200"] & (df_s0[z_col_name] <= -0.3)].sort_values(by=z_col_name, ascending=True)
            if df_s0.empty:
                st.warning("Hoy ningún activo alcista supera el nivel de sobreventa fuerte; mostrando los más cercanos a sobreventa en tendencia alcista:")
                df_s0 = raw_s0_df[raw_s0_df["_above_sma200"]].copy()
                df_s0["Score Perfil (0-100)"] = df_s0["_dip_score"]
                df_s0 = df_s0.sort_values(by=z_col_name, ascending=True)

        elif hunting_profile.startswith("🚀 2."):
            st.info(
                "**🚀 Lente Activa — Líderes de Momentum y Alpha:** Filtra empresas, índices y materias primas que baten al S&P 500 (**Alpha > 0**), "
                "con **Precio > SMA 50 y SMA 200**, ordenados por su limpieza de subida (**Sharpe 6m**) y sin estar en sobrecompra extrema ($\text{Z-Score} < +2.0\sigma$)."
            )
            df_s0["Score Perfil (0-100)"] = df_s0["_mom_score"]
            df_s0 = df_s0[df_s0["_above_sma200"] & df_s0["_above_sma50"] & (df_s0[z_col_name] < 2.0)].sort_values(by="Sharpe (6m)", ascending=False)

        elif hunting_profile.startswith("🧬 3."):
            st.info(
                "**🧬 Lente Activa — Detector de Memoria (Exponente de Hurst $H$):** Descarta los activos que se mueven al azar ($H \approx 0.50$) "
                "y destaca aquellos con **fuerte memoria tendencial ($H \ge 0.55$)** o **fuerte comportamiento de péndulo/reversión ($H \le 0.45$)**."
            )
            df_s0["Score Perfil (0-100)"] = df_s0["_hurst_score"]
            df_s0 = df_s0[(df_s0["Hurst (H)"] >= 0.55) | (df_s0["Hurst (H)"] <= 0.45)].sort_values(by="Score Perfil (0-100)", ascending=False)

        elif hunting_profile.startswith("💥 4."):
            st.info(
                "**💥 Lente Activa — Compresión de Volatilidad (*Squeeze*):** Ordena los activos donde la volatilidad de los últimos 10 días "
                "se ha reducido drásticamente respecto a su media de 6 meses (**Compresión < 0.75**), señalando calma antes de un movimiento brusco."
            )
            df_s0["Score Perfil (0-100)"] = df_s0["_squeeze_score"]
            df_s0 = df_s0.sort_values(by="Compresión Vol (10d/6m)", ascending=True)

        else:
            df_s0["Score Perfil (0-100)"] = df_s0["_mom_score"]
            df_s0 = df_s0.sort_values(by="Sharpe (6m)", ascending=False)

        df_s0 = df_s0.reset_index(drop=True)
        visible_cols = [c for c in df_s0.columns if not c.startswith("_")]

        st.success(f"✅ **{len(df_s0)} oportunidades tradicionales encontradas** bajo el criterio seleccionado:")
        st.dataframe(df_s0[visible_cols], use_container_width=True, hide_index=True)

        # Descarga Excel de la Pestaña 0
        excel_p0 = convert_dfs_to_excel({"Radar_Mercado_Tradicional": df_s0[visible_cols]})
        st.download_button(
            label="📥 Descargar Informe del Radar Tradicional en Excel (.xlsx)",
            data=excel_p0,
            file_name=f"Radar_Oportunidades_Tradicional_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()
        st.markdown("### 🎯 Puente Directo: Enviar un Activo del Radar para Analizar sus Retornos Cruzados")
        col_bridge1, col_bridge2 = st.columns([2, 1.2])

        s0_options = [
            f"{row['Ticker']} — {row['Nombre']}  |  [{row['Diagnóstico Cuantitativo']}]"
            for _, row in df_s0.iterrows()
        ]
        chosen_s0 = col_bridge1.selectbox(
            "Selecciona la empresa, índice o materia prima que quieres investigar a fondo:",
            options=s0_options,
            index=0
        )
        chosen_s0_ticker = chosen_s0.split(" — ")[0].strip()

        def set_target_from_screener(new_target):
            st.session_state.custom_target_str = new_target

        with col_bridge2:
            st.write("")
            st.button(
                f"🎯 Fijar `{chosen_s0_ticker}` como Activo Objetivo (X) en toda la App",
                on_click=set_target_from_screener,
                args=(chosen_s0_ticker,),
                type="secondary"
            )
        st.caption(
            f"💡 Al pulsar este botón, **`{chosen_s0_ticker}`** se colocará automáticamente como tu **Activo Objetivo (X)** en la barra lateral "
            f"para que puedas ir a la **Pestaña 1, 2 y 3** a buscar sus mejores retornos cruzados."
        )
    else:
        st.info(
            "👆 Elige tu **Lente Estadística** arriba y pulsa **🔎 Peinar el Mercado Tradicional Ahora** "
            "para descubrir las mejores acciones, índices y materias primas del momento."
        )

# =========================================================
# PESTAÑA 1: DESCUBRIDOR UNIVERSAL DE ACTIVOS
# =========================================================
with tab_discovery:
    st.subheader(f"🌍 Descubridor Universal de Candidatos Cruzados para `{target_ticker}`")
    st.markdown(
        f"Rastrea qué activos del mundo predicen mejor el futuro de **`{target_ticker}`** en modo **{training_mode}**, "
        f"descontando **`{commission_pct:.2f}%`** de comisión y exigiendo un umbral de **`{min_threshold_pct:.2f}%`**."
    )

    all_categories = list(dict.fromkeys([info[1] for info in MASTER_UNIVERSE.values()]))
    col_d1, col_d2, col_d3, col_d4 = st.columns([1.6, 1, 1, 1])

    selected_cats = col_d1.multiselect(
        "Categorías del Universo Maestro a rastrear:",
        options=all_categories,
        default=[c for c in all_categories if c != "Criptoactivos"]
    )
    top_n_to_send = col_d2.slider("Nº de mejores candidatos a seleccionar:", min_value=4, max_value=15, value=8)
    discovery_sort = col_d3.selectbox(
        "Ordenar candidatos por:",
        ["Sharpe (Test)", "Retorno Neto Trade (Test %)", "Win Rate (Test %)"],
        index=0
    )
    only_positive_test = col_d4.checkbox("Solo Test > 0%", value=True)

    if st.button(f"🚀 Rastrear Predictores Cruzados para `{target_ticker}`", type="primary"):
        with st.spinner(f"Descargando bloques seguros y evaluando predictores globales frente a `{target_ticker}`..."):
            disc_res, eval_cnt = run_universal_discovery(
                target_ticker, years_back, horizon, train_pct, selected_cats, only_positive_test,
                commission_pct, min_threshold_pct, is_rolling_mode, rolling_window_days, use_sma_filter
            )
            st.session_state.discovery_results = disc_res
            st.session_state.discovery_eval_cnt = eval_cnt
            st.session_state.discovery_target = target_ticker

    if "discovery_results" in st.session_state and st.session_state.get("discovery_target") == target_ticker:
        disc_list = st.session_state.discovery_results
        eval_cnt = st.session_state.get("discovery_eval_cnt", 0)

        if not disc_list:
            st.warning(f"Se descargaron {eval_cnt} activos pero ninguno superó el filtro. Desmarca la casilla 'Solo Test > 0%' o reduce el umbral mínimo.")
        else:
            disc_df = pd.DataFrame(disc_list).sort_values(by=discovery_sort, ascending=False).reset_index(drop=True)
            top_tickers_list = disc_df["Ticker"].head(top_n_to_send).tolist()
            top_tickers_str = ", ".join(top_tickers_list)

            st.success(
                f"✅ **¡Rastreo completado con éxito!** ({eval_cnt} activos descargados y alineados al calendario de `{target_ticker}`). "
                f"Mostrando **{len(disc_df)} candidatos** ordenados por **{discovery_sort}**:"
            )
            st.dataframe(disc_df, use_container_width=True, hide_index=True)

            excel_p1 = convert_dfs_to_excel({f"Descubridor_{target_ticker}": disc_df})
            st.download_button(
                label=f"📥 Descargar Tabla del Descubridor en Excel (.xlsx)",
                data=excel_p1,
                file_name=f"Descubridor_{target_ticker}_{horizon}d.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown(f"#### 🎯 Top {len(top_tickers_list)} Mejores Candidatos Descubiertos (Listos para enviar a la Pestaña 2)")
            st.code(top_tickers_str, language="text")

            def send_to_scanner_callback(tickers_string):
                st.session_state.custom_predictors_str = tickers_string

            st.button(
                f"⚡ Enviar este Top {len(top_tickers_list)} directamente al Universo de Candidatos (Barra Lateral / Pestaña 2)",
                on_click=send_to_scanner_callback,
                args=(top_tickers_str,),
                type="secondary"
            )
    else:
        st.info(
            f"👆 Pulsa el botón **Rastrear Predictores Cruzados para `{target_ticker}`** para descubrir automáticamente qué activos anticipan a `{target_ticker}`, "
            f"o pasa directamente a la **Pestaña 2** si ya tienes tus candidatos en la barra lateral."
        )

# =========================================================
# PESTAÑA 2: ESCÁNER DE COMBINACIONES (EL RADAR DE PAREJAS)
# =========================================================
with tab_scanner:
    st.subheader(f"🔍 Escáner de Combinaciones (Radar de Parejas y Tríos) para `{target_ticker}`")

    if not valid_candidates:
        st.warning("👈 Añade candidatos en la barra lateral izquierda o usa la Pestaña 1 (Descubridor Universal) para rellenarlos automáticamente.")
        selected_for_tab3 = []
    else:
        st.markdown(
            f"**Candidatos activos analizados ({len(valid_candidates)}):** `{', '.join(valid_candidates)}` | **Modo:** `{training_mode}`  \n"
            f"**Referencia a batir (Modelo 1 Base AR - Solo `{target_ticker}` en Test Neto):** "
            f"Retorno Neto: **`{ar_test_mean:+.2f}%`** | Win Rate: **`{ar_test_wr:.1f}%`** | Sharpe: **`{ar_test_sharpe:.2f}`** | Max DD: **`{ar_test_dd:.2f}%`**"
        )

        col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1.1, 1.1])
        max_combo_size = col_f1.selectbox("Tamaño máximo de combinación:", [1, 2, 3], index=2)
        sort_criterion = col_f2.selectbox(
            "Ordenar Ranking Top 15 por:",
            ["Sharpe (Test)", "Retorno Neto Trade (Test %)", "Win Rate (Test %)"],
            index=0
        )
        only_relative_strength = col_f3.checkbox(
            "Solo Fuerza Relativa (+/-)",
            value=True,
            help="Muestra únicamente combinaciones donde al menos un activo tiene peso positivo y otro negativo."
        )
        filter_clones_chk = col_f4.checkbox(
            "Filtrar Clones (Corr > 90%)",
            value=True,
            help="Descarta automáticamente parejas de activos que se mueven más de un 90% igual para evitar colinealidad."
        )

        scanner_rows, total_evaluated, clones_removed = run_full_scanner(
            returns_df, closes_df, target_ticker, valid_candidates, horizon, train_pct,
            max_combo_size, only_relative_strength, filter_clones_chk, ar_test_mean,
            commission_pct, min_threshold_pct, is_rolling_mode, rolling_window_days, use_sma_filter
        )

        if not scanner_rows:
            st.warning("No se encontraron combinaciones con estos filtros. Desmarca 'Solo Fuerza Relativa (+/-)' o reduce el umbral mínimo en la barra lateral.")
            selected_for_tab3 = valid_candidates[:2]
        else:
            full_scanner_df = pd.DataFrame(scanner_rows).sort_values(by=sort_criterion, ascending=False).reset_index(drop=True)
            scanner_df = full_scanner_df.head(15)

            clone_msg = f" ({clones_removed} combinaciones clónicas descartadas por correlación > 90%)" if clones_removed > 0 else ""
            st.success(
                f"✅ **{total_evaluated} combinaciones evaluadas al instante{clone_msg}.** "
                f"Aquí tienes el **Top 15 Out-of-Sample** (con Z-Score de señal y neto de comisiones) ordenado por **{sort_criterion}**:"
            )

            display_cols = [c for c in scanner_df.columns if c != "combo_list"]
            st.dataframe(scanner_df[display_cols], use_container_width=True, hide_index=True)

            excel_p2 = convert_dfs_to_excel({f"Escaner_{target_ticker}": full_scanner_df[display_cols]})
            st.download_button(
                label=f"📥 Descargar Ranking Completo del Escáner en Excel (.xlsx)",
                data=excel_p2,
                file_name=f"Escaner_Combinaciones_{target_ticker}_{horizon}d.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.divider()
            st.markdown("### 🎯 Puente Directo a la Auditoría Final (Pestaña 3)")

            combo_options = [
                f"#{i+1}: {row['Activos Predictores']}  —  [Sharpe: {row['Sharpe (Test)']} | Win Rate: {row['Win Rate (Test %)']}% | Neto Test: {row['Retorno Neto Trade (Test %)']:+.2f}% | Z-Score Hoy: {row['Z-Score Señal Hoy (σ)']:+.2f}σ]"
                for i, row in scanner_df.iterrows()
            ]
            combo_options.append("🛠️ Selección Manual Personalizada")

            chosen_option = st.selectbox(
                "Elige qué combinación del radar quieres auditar a fondo en la Pestaña 3 (por defecto ya está cargada la #1):",
                options=combo_options,
                index=0
            )

            if chosen_option == "🛠️ Selección Manual Personalizada":
                selected_for_tab3 = st.multiselect(
                    "Selecciona manualmente los activos para la Pestaña 3:",
                    options=valid_candidates,
                    default=valid_candidates[:2]
                )
            else:
                chosen_idx = int(chosen_option.split(":")[0].replace("#", "")) - 1
                selected_for_tab3 = scanner_df.iloc[chosen_idx]["combo_list"]
                st.info(
                    f"👉 Tienes seleccionada la combinación ganadora **`{' + '.join(selected_for_tab3)}`**. "
                    f"Pasa a la **Pestaña 3 (📊 Auditoría de los 3 Modelos)** para ver el combate cara a cara, Z-Score, Kelly y la conclusión operativa."
                )

# =========================================================
# PESTAÑA 3: AUDITORÍA DE LOS 3 MODELOS EN PARALELO
# =========================================================
with tab_models:
    if not selected_for_tab3:
        st.warning("👈 Selecciona al menos una combinación en la Pestaña 2 para auditar los modelos.")
        st.stop()

    dl_res = evaluate_strategy(
        returns_df, closes_df, target_ticker, selected_for_tab3, horizon, train_pct,
        commission_pct, min_threshold_pct, is_rolling_mode, rolling_window_days, use_sma_filter
    )
    ardl_features = [target_ticker] + selected_for_tab3
    ardl_res = evaluate_strategy(
        returns_df, closes_df, target_ticker, ardl_features, horizon, train_pct,
        commission_pct, min_threshold_pct, is_rolling_mode, rolling_window_days, use_sma_filter
    )

    models_summary = [
        {"id": 1, "name": f"Modelo 1: Base AR (Solo {target_ticker})", "res": ar_res},
        {"id": 2, "name": f"Modelo 2: Cruzado Puro ({' + '.join(selected_for_tab3)})", "res": dl_res},
        {"id": 3, "name": f"Modelo 3: Mixto ARDL ({target_ticker} + {' + '.join(selected_for_tab3)})", "res": ardl_res}
    ]
    best_model = max(models_summary, key=lambda x: x["res"]["test_metrics"]["mean_pct"])

    # ---------------------------------------------------------
    # BLOQUE SUPERIOR: CONSENSO, Z-SCORE, RÉGIMEN SMA 200 Y KELLY
    # ---------------------------------------------------------
    st.subheader(f"🚦 Consenso Operativo, Z-Score y Gestión de Capital para `{target_ticker}` ({horizon} días)")

    preds = [m["res"]["live_pred_pct"] for m in models_summary]
    long_votes = sum(1 for p in preds if p > min_threshold_pct)
    short_votes = sum(1 for p in preds if p < -min_threshold_pct)

    if long_votes == 3:
        consensus_banner = "🟢 CONSENSO ALCISTA UNÁNIME (3/3 LONG)"
    elif short_votes == 3:
        consensus_banner = "🔴 CONSENSO BAJISTA UNÁNIME (3/3 SHORT)"
    elif long_votes >= 2:
        consensus_banner = "🟢 MAYORÍA ALCISTA (2/3 LONG)"
    elif short_votes >= 2:
        consensus_banner = "🔴 MAYORÍA BAJISTA (2/3 SHORT)"
    else:
        consensus_banner = f"⚪ EN LIQUIDEZ / DIVIDIDO (Umbral {min_threshold_pct:.2f}%)"

    best_pred = best_model["res"]["live_pred_pct"]
    best_z = best_model["res"]["signal_zscore"]
    best_vol = best_model["res"]["target_vol_pct"]
    best_kelly = best_model["res"]["test_metrics"]["half_kelly_pct"]

    if abs(best_z) >= 2.0:
        z_badge = f"🔥 EXTREMA ({best_z:+.2f}σ)"
        z_mult = 1.25
    elif abs(best_z) >= 1.0:
        z_badge = f"⚡ FUERTE ({best_z:+.2f}σ)"
        z_mult = 1.0
    else:
        z_badge = f"〰️ MODERADA ({best_z:+.2f}σ)"
        z_mult = 0.65

    if best_pred > min_threshold_pct:
        best_dir = "🟢 LONG / COMPRAR"
        target_price = current_target_price * (1.0 + (best_pred / 100.0))
        stop_price = current_target_price * (1.0 - (best_vol / 100.0))
        suggested_pos_pct = min(50.0, best_kelly * z_mult)
    elif best_pred < -min_threshold_pct:
        best_dir = "🔴 SHORT / VENDER"
        target_price = current_target_price * (1.0 + (best_pred / 100.0))
        stop_price = current_target_price * (1.0 + (best_vol / 100.0))
        suggested_pos_pct = min(50.0, best_kelly * z_mult)
    else:
        best_dir = "⚪ NEUTRAL (En Liquidez)"
        target_price = current_target_price * (1.0 + (best_pred / 100.0))
        stop_price = current_target_price
        suggested_pos_pct = 0.0

    regime_label = f"🐂 ALCISTA (> SMA 200 en {sma200_dist_pct:+.1f}%)" if is_bull_regime else f"🐻 BAJISTA (< SMA 200 en {sma200_dist_pct:+.1f}%)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado de Consenso (3 Modelos)", consensus_banner)
    c2.metric("⭐ Modelo Ganador (Out-of-Sample)", f"Modelo {best_model['id']}")
    c3.metric(f"Señal del Ganador ({horizon}d)", f"{best_dir} ({best_pred:+.2f}%)")
    c4.metric("Z-Score de la Señal Hoy", z_badge, delta=f"Win Rate 90d: {best_model['res']['recent_metrics']['win_rate']:.1f}%")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"Precio Actual `{target_ticker}`", f"{current_target_price:,.2f}", delta=regime_label, delta_color="normal" if is_bull_regime else "inverse")
    p2.metric(f"🎯 Precio Objetivo Proyectado ({horizon}d)", f"{target_price:,.2f}", delta=f"{best_pred:+.2f}%")
    p3.metric(f"🛡️ Stop Loss Estadístico (1σ = ±{best_vol:.2f}%)", f"{stop_price:,.2f}", delta=f"Max DD Test: {best_model['res']['test_metrics']['max_dd']:.2f}%", delta_color="off")
    p4.metric("⚖️ Capital Sugerido (Half-Kelly Ajustado)", f"{suggested_pos_pct:.1f}% de Cartera", delta=f"Kelly Base Test: {best_kelly:.1f}%", delta_color="off")

    if (best_pred < -min_threshold_pct and is_bull_regime) or (best_pred > min_threshold_pct and not is_bull_regime):
        st.warning(
            f"⚠️ **Nota de Régimen Macro (SMA 200):** La señal a {horizon} días (**{best_dir}**) va a contracorriente del Régimen Mayor de `{target_ticker}` "
            f"(**{regime_label}**). Si operas contra la tendencia principal, considera reducir el tamaño de posición a la mitad o activar el filtro SMA 200 en la barra lateral."
        )

    st.markdown("#### 🏆 Tabla Comparativa Cara a Cara (Neto de Comisiones y con Z-Score en Vivo)")
    comp_rows = []
    for m in models_summary:
        r = m["res"]
        is_winner = (m["id"] == best_model["id"])
        diff_ar = r["test_metrics"]["mean_pct"] - ar_test_mean
        p_now = r["live_pred_pct"]
        z_now = r["signal_zscore"]
        if p_now > min_threshold_pct:
            sig_today = f"🟢 LONG ({p_now:+.2f}%)"
        elif p_now < -min_threshold_pct:
            sig_today = f"🔴 SHORT ({p_now:+.2f}%)"
        else:
            sig_today = f"⚪ NEUTRAL ({p_now:+.2f}%)"

        comp_rows.append({
            "Corona": "⭐ GANADOR" if is_winner else "—",
            "Modelo": m["name"],
            "Retorno Neto (Train)": f"{r['train_metrics']['mean_pct']:+.2f}%",
            "Retorno Neto (Test Out-of-Sample)": f"{r['test_metrics']['mean_pct']:+.2f}%",
            "Diferencial vs Modelo 1 (AR)": "Referencia Base" if m["id"] == 1 else f"{diff_ar:+.2f}%",
            "Win Rate (Test)": f"{r['test_metrics']['win_rate']:.1f}%",
            "Win Rate Reciente (90d)": f"{r['recent_metrics']['win_rate']:.1f}%",
            "Sharpe Anualizado (Test)": f"{r['test_metrics']['sharpe']:.2f}",
            "Max Drawdown (Test)": f"{r['test_metrics']['max_dd']:.2f}%",
            "Half-Kelly Sugerido": f"{r['test_metrics']['half_kelly_pct']:.1f}%",
            "Z-Score Señal Hoy": f"{z_now:+.2f}σ",
            f"Señal Hoy ({horizon}d)": sig_today
        })

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    def build_weights_df(res_obj):
        rows_w = []
        for f, w in zip(res_obj["features"], res_obj["weights"]):
            curr_ret = float(res_obj["latest_features"][f])
            z_val = float(res_obj["latest_feature_zscores"].get(f, 0.0))
            if abs(z_val) >= 2.0:
                z_state = f"🔥 Extremo ({z_val:+.2f}σ)"
            elif abs(z_val) >= 1.2:
                z_state = f"⚡ Estirado ({z_val:+.2f}σ)"
            else:
                z_state = f"Normal ({z_val:+.2f}σ)"

            rows_w.append({
                "Activo Predictor": f,
                "Coeficiente (Peso w)": round(float(w), 4),
                "Comportamiento": "📈 Tendencia (Momentum)" if w > 0 else "🔄 Reversión a la Media",
                f"Retorno Actual ({horizon}d %)": round(curr_ret, 2),
                "Z-Score Predictor Hoy": z_state,
                "Aportación a Hoy (%)": round(float(w) * curr_ret, 3)
            })
        return pd.DataFrame(rows_w)

    excel_p3 = convert_dfs_to_excel({
        "Comparativa_3_Modelos": comp_df,
        "Pesos_Modelo_1_AR": build_weights_df(ar_res),
        "Pesos_Modelo_2_Cruzado": build_weights_df(dl_res),
        "Pesos_Modelo_3_ARDL": build_weights_df(ardl_res)
    })
    st.download_button(
        label=f"📥 Descargar Auditoría Completa (3 Modelos + Pesos + Z-Scores) en Excel (.xlsx)",
        data=excel_p3,
        file_name=f"Auditoria_3_Modelos_{target_ticker}_{horizon}d.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=ar_res["df_model"].index, y=ar_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 1: Base AR (Solo {target_ticker})",
        line=dict(color="#9ca3af", width=2, dash="dot")
    ))
    fig_eq.add_trace(go.Scatter(
        x=dl_res["df_model"].index, y=dl_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 2: Cruzado Puro ({' + '.join(selected_for_tab3)})",
        line=dict(color="#2563eb", width=3)
    ))
    fig_eq.add_trace(go.Scatter(
        x=ardl_res["df_model"].index, y=ardl_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 3: Mixto ARDL ({target_ticker} + {' + '.join(selected_for_tab3)})",
        line=dict(color="#10b981", width=2.5)
    ))

    split_dt = dl_res["split_date"]
    fig_eq.add_vline(x=split_dt, line_width=2, line_dash="dash", line_color="#ef4444")
    fig_eq.add_annotation(
        x=split_dt,
        y=max(
            dl_res["df_model"]["Equity_Curve"].max(),
            ardl_res["df_model"]["Equity_Curve"].max(),
            ar_res["df_model"]["Equity_Curve"].max()
        ),
        text="⬅️ Entrenamiento (In-Sample) | Prueba Real (Out-of-Sample) ➡️",
        showarrow=False, yshift=15, font=dict(color="#ef4444", size=12)
    )
    fig_eq.update_layout(
        title=f"Curvas de Equidad Netas (%) — Modo: {training_mode} | Horizonte {horizon}d | Comisión {commission_pct:.2f}%",
        xaxis_title="Fecha", yaxis_title="Retorno Neto Acumulado (%)",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    def render_model_radiography(res_obj):
        col_t, col_g = st.columns([1.35, 1])
        w_df = build_weights_df(res_obj)
        with col_t:
            st.dataframe(w_df, use_container_width=True, hide_index=True)
            st.caption(
                f"**Sesgo base (Intercepto $b$):** `{res_obj['intercept']*100:+.3f}%`  |  "
                f"**Predicción Hoy:** `{res_obj['live_pred_pct']:+.2f}%` (**Z-Score: `{res_obj['signal_zscore']:+.2f}σ`**)  |  "
                f"**Tiempo en Mercado (Test):** `{res_obj['test_metrics']['trades_pct']:.0f}%` de los días"
            )
        with col_g:
            fig_bar = go.Figure(go.Bar(
                x=res_obj["features"],
                y=res_obj["weights"],
                marker_color=["#10b981" if w > 0 else "#ef4444" for w in res_obj["weights"]],
                text=[f"{w:+.3f}" for w in res_obj["weights"]],
                textposition="auto"
            ))
            fig_bar.update_layout(
                title="Pesos Aprendidos (Verde = Tendencia | Rojo = Reversión)",
                yaxis_title="Peso (w)", height=240, margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================
    # SECCIÓN 1: MODELO 1 — AUTORREGRESIVO BASE (SOLO TARGET)
    # =========================================================
    st.subheader(f"1️⃣ Modelo 1: Base Autorregresivo (Solo `{target_ticker}` consigo mismo)")
    m1_c1, m1_c2, m1_c3, m1_c4 = st.columns(4)
    m1_c1.metric(f"Predicción Hoy ({horizon}d)", f"{ar_res['live_pred_pct']:+.2f}%", delta=f"Z-Score: {ar_res['signal_zscore']:+.2f}σ", delta_color="off")
    m1_c2.metric("Retorno Neto / Trade (Test)", f"{ar_test_mean:+.2f}%")
    m1_c3.metric("Win Rate Global / 90d (Test)", f"{ar_test_wr:.1f}%  |  {ar_res['recent_metrics']['win_rate']:.1f}%")
    m1_c4.metric("Sharpe / Max DD (Test)", f"{ar_test_sharpe:.2f}  |  {ar_test_dd:.1f}%")

    render_model_radiography(ar_res)

    ar_w = ar_res["weights"][0]
    ar_style = "Tendencial (cuando sube, tiende a seguir subiendo)" if ar_w > 0 else "de Reversión a la Media (cuando cae, tiende a rebotar)"
    if ar_test_mean > 0 and ar_test_sharpe > 0.8:
        st.info(
            f"📌 **Conclusión Final del Modelo 1 (Referencia Base):** `{target_ticker}` por sí solo tiene un comportamiento **{ar_style}** "
            f"muy sólido en el mercado actual, ganando **`{ar_test_mean:+.2f}%`** neto por operación con un Sharpe de **`{ar_test_sharpe:.2f}`** y Max Drawdown de **`{ar_test_dd:.2f}%`**. "
            f"Solo debemos operar el Modelo 2 o el Modelo 3 si logran superar este `{ar_test_mean:+.2f}%`."
        )
    else:
        st.warning(
            f"📌 **Conclusión Final del Modelo 1 (Referencia Base):** Mirar únicamente el gráfico pasado de `{target_ticker}` ({ar_style}) "
            f"da un rendimiento pobre o inestable en Test (**`{ar_test_mean:+.2f}%`** por trade, Sharpe **`{ar_test_sharpe:.2f}`**). "
            f"Necesitamos apoyarnos en los Retornos Cruzados (Modelo 2 o 3) para obtener ventaja estadística."
        )

    st.divider()

    # =========================================================
    # SECCIÓN 2: MODELO 2 — RETORNOS CRUZADOS PUROS (DL)
    # =========================================================
    dl_mean = dl_res["test_metrics"]["mean_pct"]
    dl_wr = dl_res["test_metrics"]["win_rate"]
    dl_sharpe = dl_res["test_metrics"]["sharpe"]
    dl_dd = dl_res["test_metrics"]["max_dd"]
    dl_diff_ar = dl_mean - ar_test_mean

    st.subheader(f"2️⃣ Modelo 2: Retornos Cruzados Puros (`{' + '.join(selected_for_tab3)}` sin `{target_ticker}`)")
    m2_c1, m2_c2, m2_c3, m2_c4 = st.columns(4)
    m2_c1.metric(f"Predicción Hoy ({horizon}d)", f"{dl_res['live_pred_pct']:+.2f}%", delta=f"Z-Score: {dl_res['signal_zscore']:+.2f}σ", delta_color="off")
    m2_c2.metric("Retorno Neto / Trade (Test)", f"{dl_mean:+.2f}%", delta=f"{dl_diff_ar:+.2f}% vs Modelo 1 (AR)")
    m2_c3.metric("Win Rate Global / 90d (Test)", f"{dl_wr:.1f}%  |  {dl_res['recent_metrics']['win_rate']:.1f}%", delta=f"{dl_wr - ar_test_wr:+.1f}% vs AR")
    m2_c4.metric("Sharpe / Max DD (Test)", f"{dl_sharpe:.2f}  |  {dl_dd:.1f}%", delta=f"{dl_sharpe - ar_test_sharpe:+.2f} Sharpe vs AR")

    render_model_radiography(dl_res)

    has_rs_2 = any(w > 0 for w in dl_res["weights"]) and any(w < 0 for w in dl_res["weights"])
    rs_text = "Además, aprovecha un patrón de **Fuerza Relativa (+/-)** entre sus activos. " if has_rs_2 else ""

    if dl_mean > 0 and dl_diff_ar >= 0:
        st.success(
            f"✅ **Conclusión Final del Modelo 2 (APROBADO — VENTAJA REAL):** Usar **`{' + '.join(selected_for_tab3)}`** "
            f"bate al activo en solitario por **`{dl_diff_ar:+.2f}%`** neto por operación en la prueba real, alcanzando un Win Rate del **`{dl_wr:.1f}%`** "
            f"(**`{dl_res['recent_metrics']['win_rate']:.1f}%`** en los últimos 90 días), un Sharpe de **`{dl_sharpe:.2f}`**, Max Drawdown de **`{dl_dd:.2f}%`** "
            f"y un Z-Score de señal hoy de **`{dl_res['signal_zscore']:+.2f}σ`**. {rs_text}"
        )
    elif dl_mean > 0 and dl_diff_ar < 0:
        st.warning(
            f"⚠️ **Conclusión Final del Modelo 2 (RENTABLE PERO INEFICIENTE VS AR):** Aunque usar **`{' + '.join(selected_for_tab3)}`** "
            f"gana dinero en Test (**`{dl_mean:+.2f}%`** neto por operación, Win Rate **`{dl_wr:.1f}%`**), rinde **`{dl_diff_ar:.2f}%`** menos que "
            f"el Modelo 1 Base. Te recomiendo elegir una pareja más alta del Ranking en la Pestaña 2 que tenga Diferencial vs AR positivo."
        )
    else:
        st.error(
            f"❌ **Conclusión Final del Modelo 2 (DESCARTAR — SOBREAJUSTE):** Esta combinación de activos pierde dinero en la prueba reciente "
            f"(**`{dl_mean:+.2f}%`** neto por operación). La relación funcionó en el pasado pero se ha roto en el mercado actual. Prohibido operar."
        )

    st.divider()

    # =========================================================
    # SECCIÓN 3: MODELO 3 — MIXTO ARDL (TARGET + EXTERNOS)
    # =========================================================
    ardl_mean = ardl_res["test_metrics"]["mean_pct"]
    ardl_wr = ardl_res["test_metrics"]["win_rate"]
    ardl_sharpe = ardl_res["test_metrics"]["sharpe"]
    ardl_dd = ardl_res["test_metrics"]["max_dd"]
    ardl_diff_dl = ardl_mean - dl_mean
    ardl_diff_ar = ardl_mean - ar_test_mean

    st.subheader(f"3️⃣ Modelo 3: Mixto ARDL (`{target_ticker}` + `{' + '.join(selected_for_tab3)}`)")
    m3_c1, m3_c2, m3_c3, m3_c4 = st.columns(4)
    m3_c1.metric(f"Predicción Hoy ({horizon}d)", f"{ardl_res['live_pred_pct']:+.2f}%", delta=f"Z-Score: {ardl_res['signal_zscore']:+.2f}σ", delta_color="off")
    m3_c2.metric("Retorno Neto / Trade (Test)", f"{ardl_mean:+.2f}%", delta=f"{ardl_diff_dl:+.2f}% vs Modelo 2 (Cruzado)")
    m3_c3.metric("Win Rate Global / 90d (Test)", f"{ardl_wr:.1f}%  |  {ardl_res['recent_metrics']['win_rate']:.1f}%", delta=f"{ardl_wr - dl_wr:+.1f}% vs Modelo 2")
    m3_c4.metric("Sharpe / Max DD (Test)", f"{ardl_sharpe:.2f}  |  {ardl_dd:.1f}%", delta=f"{ardl_sharpe - dl_sharpe:+.2f} Sharpe vs Modelo 2")

    render_model_radiography(ardl_res)

    if ardl_mean > dl_mean and ardl_mean > ar_test_mean:
        st.success(
            f"✅ **Conclusión Final del Modelo 3 (SINERGIA PERFECTA — MEJOR OPCIÓN):** Añadir el propio `{target_ticker}` junto a "
            f"**`{' + '.join(selected_for_tab3)}`** mejora aún más al Modelo 2 en **`{ardl_diff_dl:+.2f}%`** por operación y supera al Modelo 1 Base en **`{ardl_diff_ar:+.2f}%`** "
            f"(Max Drawdown: **`{ardl_dd:.2f}%`**, Z-Score Hoy: **`{ardl_res['signal_zscore']:+.2f}σ`**). Ambos mundos se complementan sin estorbarse."
        )
    elif ardl_mean < dl_mean:
        st.warning(
            f"⚠️ **Conclusión Final del Modelo 3 (COLINEALIDAD / REDUNDANCIA DETECTADA):** Al meter a `{target_ticker}` dentro de la fórmula junto a "
            f"**`{' + '.join(selected_for_tab3)}`**, el retorno neto baja un **`{ardl_diff_dl:.2f}%`** respecto al Modelo 2 Cruzado Puro "
            f"(pasa de `{dl_mean:+.2f}%` a `{ardl_mean:+.2f}%`). Esto ocurre porque los predictores externos ya contienen la información del precio y al juntarlos se estorban. "
            f"**Quédate con el Modelo 2 (Cruzado Puro).**"
        )
    else:
        st.info(
            f"📌 **Conclusión Final del Modelo 3:** El modelo mixto mejora ligeramente al Modelo 2 (`{ardl_diff_dl:+.2f}%`), "
            f"pero sigue a `{ardl_diff_ar:+.2f}%` de distancia del Modelo 1 Base. Revisa la tabla superior para operar siempre el modelo coronado con ⭐."
        )
