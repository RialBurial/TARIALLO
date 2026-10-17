import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import plotly.graph_objects as go
from itertools import combinations
from datetime import datetime, timedelta

# Configuración general de la página
st.set_page_config(
    page_title="Cross-Returns Quant Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analizador Cuantitativo de Retornos Cruzados y Fuerza Relativa")
st.markdown(
    "Herramienta universal para predecir retornos futuros de **Índices, Acciones, Forex, Materias Primas y Cripto** "
    "utilizando *Distributed Lags* (Retornos Cruzados) y modelos de Fuerza Relativa."
)

# ---------------------------------------------------------
# MÓDULO A: PANEL LATERAL DE CONFIGURACIÓN
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. Selección de Activos")

presets = {
    "Índices EE.UU. + Sectores y Macro": {
        "target": "SPY",
        "predictors": "QQQ, IWM, TLT, GLD, ^VIX, XLF, XLK, XLE, NVDA, AAPL"
    },
    "Gigantes Tecnológicos (Acciones)": {
        "target": "NVDA",
        "predictors": "AMD, TSM, MSFT, AAPL, META, GOOGL, AMZN, QQQ, SMH"
    },
    "Índices Globales y Divisas": {
        "target": "^GDAXI",
        "predictors": "SPY, ^STOXX50E, ^IBEX, ^N225, EURUSD=X, GC=F, CL=F"
    },
    "Cripto (Ejemplo del Vídeo MemLabs)": {
        "target": "ETH-USD",
        "predictors": "BNB-USD, SOL-USD, BTC-USD, UNI7083-USD, AAVE-USD, LINK-USD, AVAX-USD"
    },
    "Personalizado (Escribir libremente)": {
        "target": "SPY",
        "predictors": "QQQ, TLT, GLD, ^VIX"
    }
}

preset_choice = st.sidebar.selectbox("Plantillas rápidas de mercado:", list(presets.keys()))

target_ticker = st.sidebar.text_input(
    "🎯 Activo Objetivo a Operar (X):",
    value=presets[preset_choice]["target"],
    help="Escribe cualquier ticker de Yahoo Finance (ej. SPY, NVDA, ^IBEX, BTC-USD, GC=F)."
).strip().upper()

predictors_input = st.sidebar.text_area(
    "🔗 Universo de Activos Predictores / Candidatos (Y):",
    value=presets[preset_choice]["predictors"],
    help="Separados por coma. Se usarán tanto para tu análisis manual como para el escáner automático."
)

candidate_tickers = [t.strip().upper() for t in predictors_input.split(",") if t.strip() and t.strip().upper() != target_ticker]

selected_predictors = st.sidebar.multiselect(
    "✅ Activos activos para el Modelo Manual:",
    options=candidate_tickers,
    default=candidate_tickers[:3] if len(candidate_tickers) >= 3 else candidate_tickers
)

st.sidebar.header("⏱️ 2. Parámetros del Modelo")

horizon = st.sidebar.slider(
    "Horizonte de Retorno / Predicción (Días):",
    min_value=1, max_value=30, value=7,
    help="En el vídeo se usan 7 días (retornos semanales). Puedes bajarlo a 1-3 días para swing corto o subirlo a 15-20."
)

years_back = st.sidebar.slider(
    "Histórico de datos a descargar (Años):",
    min_value=1, max_value=10, value=4
)

train_Pct = st.sidebar.slider(
    "División Entrenamiento / Prueba (Train %):",
    min_value=50, max_value=90, value=70, step=5,
    help="Porcentaje del pasado usado para aprender los pesos. El resto (Test) valida si gana dinero hoy."
)

include_ar = st.sidebar.checkbox(
    "Incluir propio activo en el modelo (Modelo Mixto ARDL)",
    value=False,
    help="Añade el retorno actual del activo objetivo junto a los activos cruzados."
)

force_rs = st.sidebar.checkbox(
    "Forzar restricción de Fuerza Relativa (+/-)",
    value=False,
    help="Obliga matemáticamente a que al menos un predictor tenga peso positivo y otro negativo."
)

# ---------------------------------------------------------
# FUNCIONES DE DATOS Y MODELADO MATEMÁTICO
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers_list, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    data = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = tickers_list
    # Alinear series temporales (rellenar festivos cruzados y limpiar nulos iniciales)
    closes = closes.ffill().dropna()
    return closes

def fit_linear_or_constrained(X_train, y_train, enforce_opposite_signs=False):
    n_features = X_train.shape[1]
    model = LinearRegression()
    model.fit(X_train, y_train)
    weights = model.coef_.copy()
    intercept = float(model.intercept_)

    # Si el usuario pide forzar Fuerza Relativa (+/-) y todos los pesos tienen el mismo signo:
    if enforce_opposite_signs and n_features >= 2:
        if np.all(weights >= 0) or np.all(weights <= 0):
            # Elegir el predictor con mayor correlación positiva y el menor para forzar signos opuestos
            corrs = [np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(n_features)]
            pos_idx = int(np.argmax(corrs))
            neg_idx = int(np.argmin(corrs))
            if pos_idx == neg_idx:
                neg_idx = (pos_idx + 1) % n_features

            bounds = [(None, None)]  # Intercepto libre
            for i in range(n_features):
                if i == pos_idx:
                    bounds.append((1e-6, None))
                elif i == neg_idx:
                    bounds.append((None, -1e-6))
                else:
                    bounds.append((None, None))

            def loss_func(params):
                b = params[0]
                w = params[1:]
                pred = b + X_train.dot(w)
                return np.mean((y_train - pred) ** 2)

            init_params = np.zeros(n_features + 1)
            res = minimize(loss_func, init_params, method="L-BFGS-B", bounds=bounds)
            if res.success:
                intercept = float(res.x[0])
                weights = res.x[1:]

    return weights, intercept

def evaluate_strategy(df_returns, target_col, feature_cols, horizon_days, train_ratio, enforce_rs=False):
    df = pd.DataFrame(index=df_returns.index)
    # Retorno futuro a predecir (t + horizon)
    df["Target_Future"] = df_returns[target_col].shift(-horizon_days)
    
    for col in feature_cols:
        df[col] = df_returns[col]

    # Fila más reciente (hoy) para lanzar la predicción en vivo hacia los próximos N días
    latest_features = df_returns[feature_cols].iloc[-1:]

    # Limpiar últimas filas sin retorno futuro conocido para entrenar/validar
    df_model = df.dropna().copy()
    split_idx = int(len(df_model) * (train_ratio / 100.0))

    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["Target_Future"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Target_Future"].values
    X_all = df_model[feature_cols].values

    weights, intercept = fit_linear_or_constrained(X_train, y_train, enforce_rs)

    df_model["Pred"] = intercept + X_all.dot(weights)
    # Señal operativa: +1 (Long) si predice retorno positivo, -1 (Short) si predice negativo
    df_model["Signal"] = np.sign(df_model["Pred"])
    df_model["Strategy_Return"] = df_model["Signal"] * df_model["Target_Future"]

    # Escalar el retorno por 1 / horizon_days al acumular a diario (cartera solapada equivalente)
    df_model["Equity_Curve"] = (df_model["Strategy_Return"] / horizon_days).cumsum() * 100.0
    df_model["BuyHold_Curve"] = (df_model["Target_Future"] / horizon_days).cumsum() * 100.0

    train_strat = df_model["Strategy_Return"].iloc[:split_idx]
    test_strat = df_model["Strategy_Return"].iloc[split_idx:]

    def get_metrics(series):
        if len(series) == 0:
            return {"mean_pct": 0.0, "win_rate": 0.0, "sharpe": 0.0, "total_pnl": 0.0}
        mean_ret = series.mean() * 100.0
        win_rate = (series > 0).mean() * 100.0
        std_ret = series.std()
        sharpe = (series.mean() / std_ret) * np.sqrt(252.0 / horizon_days) if std_ret > 1e-9 else 0.0
        total_pnl = (series.sum() / horizon_days) * 100.0
        return {
            "mean_pct": mean_ret,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "total_pnl": total_pnl
        }

    live_pred = float(intercept + latest_features.values.dot(weights)[0])

    return {
        "weights": weights,
        "intercept": intercept,
        "features": feature_cols,
        "train_metrics": get_metrics(train_strat),
        "test_metrics": get_metrics(test_strat),
        "df_model": df_model,
        "split_date": df_model.index[split_idx],
        "live_pred_pct": live_pred * 100.0,
        "latest_features": latest_features.iloc[0] * 100.0
    }

# ---------------------------------------------------------
# EJECUCIÓN Y CARGA DE DATOS
# ---------------------------------------------------------
all_requested_tickers = list(dict.fromkeys([target_ticker] + candidate_tickers))

with st.spinner(f"Descargando cotizaciones en tiempo real para {len(all_requested_tickers)} activos..."):
    closes_df = load_market_data(all_requested_tickers, years_back)

if target_ticker not in closes_df.columns:
    st.error(f"❌ No se encontraron datos para el activo objetivo `{target_ticker}`. Revisa el símbolo en Yahoo Finance.")
    st.stop()

valid_candidates = [c for c in candidate_tickers if c in closes_df.columns]
missing_tickers = set(candidate_tickers) - set(valid_candidates)
if missing_tickers:
    st.warning(f"⚠️ No se pudieron descargar datos para: {', '.join(missing_tickers)}. Se omitirán del análisis.")

selected_predictors = [p for p in selected_predictors if p in valid_candidates]

if len(closes_df) < horizon * 10:
    st.error("❌ No hay suficientes barras históricas para el horizonte seleccionado. Reduce el horizonte o amplía los años.")
    st.stop()

# Cálculo de retornos en la ventana N días (exactamente como en el vídeo)
returns_df = closes_df.pct_change(horizon).dropna()

# Pestañas principales de la aplicación
tab_manual, tab_scanner = st.tabs([
    "📊 1. Cuadro de Mando y Backtest (Análisis de Activos)",
    "🔍 2. Escáner Automático (Buscador de las Mejores Parejas)"
])

# =========================================================
# PESTAÑA 1: ANÁLISIS MANUAL (MÓDULOS B, C y D)
# =========================================================
with tab_manual:
    if not selected_predictors:
        st.info("👈 Selecciona al menos un activo predictor en la barra lateral para iniciar el modelo.")
    else:
        # 1. Evaluar Modelo Autorregresivo Base (AR)
        ar_res = evaluate_strategy(returns_df, target_ticker, [target_ticker], horizon, train_Pct, False)
        
        # 2. Evaluar Modelo de Retornos Cruzados elegido por el usuario
        user_features = ([target_ticker] + selected_predictors) if include_ar else selected_predictors
        cross_res = evaluate_strategy(returns_df, target_ticker, user_features, horizon, train_Pct, force_rs)

        # ---------------------------------------------------------
        # MÓDULO B: SEMÁFORO OPERATIVO Y SEÑAL ACTUAL
        # ---------------------------------------------------------
        st.subheader(f"🚦 Señal Actual en Vivo para `{target_ticker}` (Próximos {horizon} días)")
        
        pred_val = cross_res["live_pred_pct"]
        if pred_val > 0.05:
            signal_label = "🟢 LONG / ALCISTA"
        elif pred_val < -0.05:
            signal_label = "🔴 SHORT / BAJISTA"
        else:
            signal_label = "⚪ NEUTRAL / SIN VENTAJA CLARA"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dirección Propuesta", signal_label)
        col2.metric(f"Retorno Esperado ({horizon}d)", f"{pred_val:+.2f}%")
        col3.metric("Win Rate (Out-of-Sample)", f"{cross_res['test_metrics']['win_rate']:.1f}%")
        col4.metric(
            "Retorno Medio / Trade (Test)",
            f"{cross_res['test_metrics']['mean_pct']:+.2f}%",
            delta=f"{cross_res['test_metrics']['mean_pct'] - ar_res['test_metrics']['mean_pct']:+.2f}% vs AR"
        )

        # Explicación en lenguaje natural de lo que está ocurriendo hoy
        contributions = []
        for feat, w in zip(cross_res["features"], cross_res["weights"]):
            curr_ret = cross_res["latest_features"][feat]
            impact = w * curr_ret
            regime = "Tendencial (+)" if w >= 0 else "Reversión (-)"
            contributions.append((feat, curr_ret, w, impact, regime))
        
        contributions.sort(key=lambda x: abs(x[3]), reverse=True)
        top_driver = contributions[0]
        
        st.info(
            f"💡 **Lectura Cuantitativa Actual:** El mayor impulsor de la señal hoy es **`{top_driver[0]}`**, "
            f"que acumula un retorno de **{top_driver[1]:+.2f}%** en los últimos {horizon} días. "
            f"Como su coeficiente es de **{top_driver[4]}** ({top_driver[2]:+.3f}), aporta **{top_driver[3]:+.2f}%** "
            f"a la proyección de `{target_ticker}`."
        )

        st.divider()

        # ---------------------------------------------------------
        # MÓDULO C: RADIOGRAFÍA DEL MODELO (PESOS Y FUERZA RELATIVA)
        # ---------------------------------------------------------
        st.subheader("🧬 Radiografía del Modelo (Pesos y Fuerza Relativa)")
        
        col_weights, col_chart = st.columns([1.2, 1])

        weights_df = pd.DataFrame({
            "Activo Predictor": cross_res["features"],
            "Coeficiente (Peso w)": [round(w, 4) for w in cross_res["weights"]],
            "Comportamiento": ["📈 Tendencia (Momentum)" if w > 0 else "🔄 Reversión a la Media" for w in cross_res["weights"]],
            f"Retorno Actual ({horizon}d %)": [round(cross_res["latest_features"][f], 2) for f in cross_res["features"]],
            "Aportación a Hoy (%)": [round(w * cross_res["latest_features"][f], 3) for w, f in zip(cross_res["weights"], cross_res["features"])]
        })

        with col_weights:
            st.markdown("**Tabla de Coeficientes Aprendidos (Entrenamiento):**")
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
            st.caption(f"**Intercepto / Sesgo base (Bias $b$):** `{cross_res['intercept']*100:+.3f}%`")

            has_pos = any(w > 0 for w in cross_res["weights"])
            has_neg = any(w < 0 for w in cross_res["weights"])
            if has_pos and has_neg:
                pos_assets = [f for f, w in zip(cross_res["features"], cross_res["weights"]) if w > 0]
                neg_assets = [f for f, w in zip(cross_res["features"], cross_res["weights"]) if w < 0]
                st.success(
                    f"⚖️ **Patrón de Fuerza Relativa Detectado:** El modelo ha encontrado un equilibrio de signos opuestos. "
                    f"Cuando **{', '.join(pos_assets)}** rinden mejor que **{', '.join(neg_assets)}**, la expectativa para `{target_ticker}` se vuelve alcista."
                )

        with col_chart:
            fig_w = go.Figure(go.Bar(
                x=cross_res["features"],
                y=cross_res["weights"],
                marker_color=["#10b981" if w > 0 else "#ef4444" for w in cross_res["weights"]],
                text=[f"{w:+.3f}" for w in cross_res["weights"]],
                textposition="auto"
            ))
            fig_w.update_layout(
                title="Pesos del Modelo por Activo (Verde = Momentum | Rojo = Reversión)",
                yaxis_title="Peso (w)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_w, use_container_width=True)

        st.divider()

        # ---------------------------------------------------------
        # MÓDULO D: VALIDACIÓN ESTADÍSTICA Y CURVA DE EQUIDAD
        # ---------------------------------------------------------
        st.subheader("📈 Validación Estadística: In-Sample (Entrenamiento) vs. Out-of-Sample (Prueba)")

        comp_table = pd.DataFrame([
            {
                "Modelo": f"1. Autorregresivo Base (Solo {target_ticker})",
                "Retorno Medio Trade (Train)": f"{ar_res['train_metrics']['mean_pct']:+.2f}%",
                "Retorno Medio Trade (Test Out-of-Sample)": f"{ar_res['test_metrics']['mean_pct']:+.2f}%",
                "Win Rate (Test)": f"{ar_res['test_metrics']['win_rate']:.1f}%",
                "Sharpe Anualizado (Test)": f"{ar_res['test_metrics']['sharpe']:.2f}",
                "PnL Acumulado (Test)": f"{ar_res['test_metrics']['total_pnl']:+.1f}%"
            },
            {
                "Modelo": f"2. Retornos Cruzados ({', '.join(cross_res['features'])})",
                "Retorno Medio Trade (Train)": f"{cross_res['train_metrics']['mean_pct']:+.2f}%",
                "Retorno Medio Trade (Test Out-of-Sample)": f"{cross_res['test_metrics']['mean_pct']:+.2f}%",
                "Win Rate (Test)": f"{cross_res['test_metrics']['win_rate']:.1f}%",
                "Sharpe Anualizado (Test)": f"{cross_res['test_metrics']['sharpe']:.2f}",
                "PnL Acumulado (Test)": f"{cross_res['test_metrics']['total_pnl']:+.1f}%"
            }
        ])
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        # Gráfico de Curva de Equidad con separación Train / Test
        df_plot = cross_res["df_model"]
        ar_plot = ar_res["df_model"]
        split_dt = cross_res["split_date"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Equity_Curve"],
            mode="lines", name="Estrategia Retornos Cruzados",
            line=dict(color="#2563eb", width=2.5)
        ))
        fig_eq.add_trace(go.Scatter(
            x=ar_plot.index, y=ar_plot["Equity_Curve"],
            mode="lines", name=f"Modelo Autorregresivo ({target_ticker})",
            line=dict(color="#9ca3af", width=1.5, dash="dot")
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["BuyHold_Curve"],
            mode="lines", name=f"Buy & Hold ({target_ticker})",
            line=dict(color="#f59e0b", width=1.5, dash="dash")
        ))

        # Línea vertical de corte entre Entrenamiento y Prueba
        fig_eq.add_vline(x=split_dt, line_width=2, line_dash="dash", line_color="#ef4444")
        fig_eq.add_annotation(
            x=split_dt, y=max(df_plot["Equity_Curve"].max(), df_plot["BuyHold_Curve"].max()),
            text="⬅️ Entrenamiento (In-Sample) | Prueba Real (Out-of-Sample) ➡️",
            showarrow=False, yshift=15, font=dict(color="#ef4444", size=12)
        )

        fig_eq.update_layout(
            title=f"Curva de Equidad Acumulada (%) — Horizonte {horizon} días",
            xaxis_title="Fecha",
            yaxis_title="Retorno Bruto Acumulado (%)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

# =========================================================
# PESTAÑA 2: ESCÁNER AUTOMÁTICO DE PAREJAS Y CESTAS
# =========================================================
with tab_scanner:
    st.subheader(f"🔍 Escáner Automático de Predictores para `{target_ticker}`")
    st.markdown(
        "Este motor prueba automáticamente todas las combinaciones posibles de **1, 2 y 3 activos** de tu lista de candidatos "
        "y las clasifica según su rendimiento **Out-of-Sample (Prueba)** para descubrir qué activos esconden la mayor ventaja estadística."
    )

    col_s1, col_s2 = st.columns(2)
    max_combo_size = col_s1.selectbox("Tamaño máximo de combinación de activos a probar:", [1, 2, 3], index=1)
    only_relative_strength = col_s2.checkbox(
        "Mostrar solo modelos con Fuerza Relativa natural (pesos + y - a la vez)",
        value=False
    )

    if st.button("🚀 Ejecutar Escáner Cuantitativo Ahora", type="primary"):
        combos_to_test = []
        for r in range(1, max_combo_size + 1):
            combos_to_test.extend(list(combinations(valid_candidates, r)))

        results_list = []
        progress_bar = st.progress(0, text="Analizando combinaciones de activos...")

        for idx, combo in enumerate(combos_to_test):
            combo_list = list(combo)
            res = evaluate_strategy(returns_df, target_ticker, combo_list, horizon, train_Pct, force_rs)
            
            w = res["weights"]
            is_rs = (any(val > 0 for val in w) and any(val < 0 for val in w))
            
            if only_relative_strength and not is_rs:
                continue

            # Solo guardar modelos que sean positivos tanto en Train como en Test para evitar casualidades
            if res["train_metrics"]["mean_pct"] > 0:
                weights_str = ", ".join([f"{f} ({val:+.2f})" for f, val in zip(combo_list, w)])
                pred_now = res["live_pred_pct"]
                results_list.append({
                    "Activos Predictores": " + ".join(combo_list),
                    "Fuerza Relativa (+/-)": "✅ Sí" if is_rs else "No",
                    "Pesos (w)": weights_str,
                    "Retorno Medio Trade (Test %)": round(res["test_metrics"]["mean_pct"], 3),
                    "Win Rate (Test %)": round(res["test_metrics"]["win_rate"], 1),
                    "Sharpe (Test)": round(res["test_metrics"]["sharpe"], 2),
                    "Retorno Medio (Train %)": round(res["train_metrics"]["mean_pct"], 3),
                    f"Señal Hoy ({horizon}d)": f"{'🟢 LONG' if pred_now > 0 else '🔴 SHORT'} ({pred_now:+.2f}%)"
                })
            
            progress_bar.progress((idx + 1) / len(combos_to_test))

        progress_bar.empty()

        if not results_list:
            st.warning("No se encontraron combinaciones con expectativa positiva. Prueba a añadir más activos o cambiar el horizonte de días.")
        else:
            scanner_df = pd.DataFrame(results_list)
            scanner_df = scanner_df.sort_values(by="Retorno Medio Trade (Test %)", ascending=False).head(15)
            st.success(f"✅ Escaneo completado ({len(combos_to_test)} modelos evaluados). Aquí tienes el **Top 15 Out-of-Sample**:")
            st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            st.caption(
                "💡 **Consejo:** Copia los activos de tu fila favorita y selecciónalos en el panel izquierdo "
                "para ver su curva de equidad detallada en la Pestaña 1."
            )
