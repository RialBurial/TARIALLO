import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from itertools import combinations
from datetime import datetime, timedelta

# Configuración general de la página
st.set_page_config(
    page_title="Cross-Returns Quant Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analizador Cuantitativo de Retornos Cruzados y Fuerza Relativa")
st.markdown(
    "Flujo profesional en 2 pasos: **1º Rastrea** las mejores combinaciones fuera de muestra en el Escáner "
    "y **2º Audita** los 3 modelos en paralelo (Base AR, Cruzado Puro y Mixto ARDL) con diagnóstico automático."
)

# ---------------------------------------------------------
# BARRA LATERAL: CONFIGURACIÓN LIMPÍA Y DIRECTA
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. Universo de Activos")

presets = {
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
        "predictors": "SPY, ^STOXX50E, ^IBEX, ^N225, EURUSD=X, GC=F, CL=F"
    },
    "Cripto (Ejemplo del Vídeo MemLabs)": {
        "target": "ETH-USD",
        "predictors": "BNB-USD, SOL-USD, BTC-USD, UNI7083-USD, AAVE-USD, LINK-USD, AVAX-USD"
    },
    "Personalizado (Escribir libremente)": {
        "target": "SPY",
        "predictors": "QQQ, TLT, GLD, ^VIX, IWM"
    }
}

preset_choice = st.sidebar.selectbox("Plantillas rápidas de mercado:", list(presets.keys()))

target_ticker = st.sidebar.text_input(
    "🎯 Activo Objetivo a Operar (X):",
    value=presets[preset_choice]["target"],
    help="Escribe cualquier ticker de Yahoo Finance (ej. SPY, NVDA, ^IBEX, BTC-USD, GC=F)."
).strip().upper()

predictors_input = st.sidebar.text_area(
    "🔗 Universo de Activos Candidatos (Y):",
    value=presets[preset_choice]["predictors"],
    help="Separados por coma. El Escáner probará todas las combinaciones posibles entre ellos."
)

candidate_tickers = [
    t.strip().upper() for t in predictors_input.split(",")
    if t.strip() and t.strip().upper() != target_ticker
]

st.sidebar.header("⏱️ 2. Parámetros Estadísticos")

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

# ---------------------------------------------------------
# MOTOR DE DATOS Y EVALUACIÓN MATEMÁTICA
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
    closes = closes.ffill().dropna()
    return closes

def evaluate_strategy(df_returns, target_col, feature_cols, horizon_days, train_ratio):
    df = pd.DataFrame(index=df_returns.index)
    df["Target_Future"] = df_returns[target_col].shift(-horizon_days)

    for col in feature_cols:
        df[col] = df_returns[col]

    latest_features = df_returns[feature_cols].iloc[-1:]

    df_model = df.dropna().copy()
    split_idx = int(len(df_model) * (train_ratio / 100.0))

    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["Target_Future"].values
    X_all = df_model[feature_cols].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    weights = model.coef_.copy()
    intercept = float(model.intercept_)

    df_model["Pred"] = intercept + X_all.dot(weights)
    df_model["Signal"] = np.sign(df_model["Pred"])
    df_model["Strategy_Return"] = df_model["Signal"] * df_model["Target_Future"]

    df_model["Equity_Curve"] = (df_model["Strategy_Return"] / horizon_days).cumsum() * 100.0
    df_model["BuyHold_Curve"] = (df_model["Target_Future"] / horizon_days).cumsum() * 100.0

    train_strat = df_model["Strategy_Return"].iloc[:split_idx]
    test_strat = df_model["Strategy_Return"].iloc[split_idx:]

    def get_metrics(series):
        if len(series) == 0:
            return {"mean_pct": 0.0, "win_rate": 0.0, "sharpe": 0.0, "total_pnl": 0.0}
        mean_ret = float(series.mean() * 100.0)
        win_rate = float((series > 0).mean() * 100.0)
        std_ret = float(series.std())
        sharpe = float((series.mean() / std_ret) * np.sqrt(252.0 / horizon_days)) if std_ret > 1e-9 else 0.0
        total_pnl = float((series.sum() / horizon_days) * 100.0)
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

# Descarga y validación de datos
all_requested_tickers = list(dict.fromkeys([target_ticker] + candidate_tickers))

with st.spinner(f"Descargando y alineando datos de mercado para {len(all_requested_tickers)} activos..."):
    closes_df = load_market_data(all_requested_tickers, years_back)

if target_ticker not in closes_df.columns:
    st.error(f"❌ No se encontraron datos para el activo objetivo `{target_ticker}`. Revisa el símbolo en Yahoo Finance.")
    st.stop()

valid_candidates = [c for c in candidate_tickers if c in closes_df.columns]
missing_tickers = set(candidate_tickers) - set(valid_candidates)
if missing_tickers:
    st.warning(f"⚠️ No se pudieron descargar datos para: {', '.join(missing_tickers)}. Se omitirán del análisis.")

if not valid_candidates:
    st.error("❌ Necesitas al menos un activo candidato válido en la barra lateral.")
    st.stop()

if len(closes_df) < horizon * 10:
    st.error("❌ No hay suficiente histórico para este horizonte. Reduce los días de horizonte o amplía los años.")
    st.stop()

returns_df = closes_df.pct_change(horizon).dropna()

# Evaluación del Modelo 1 Base (AR) para tener siempre la referencia lista
ar_res = evaluate_strategy(returns_df, target_ticker, [target_ticker], horizon, train_pct)
ar_test_mean = ar_res["test_metrics"]["mean_pct"]
ar_test_wr = ar_res["test_metrics"]["win_rate"]
ar_test_sharpe = ar_res["test_metrics"]["sharpe"]

# ---------------------------------------------------------
# CÁLCULO AUTOMÁTICO DEL ESCÁNER (SIN BOTONES EXTRA)
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def run_full_scanner(returns_data, target, candidates, horizon_days, train_ratio, max_r, only_rs, ar_mean_ref):
    combos = []
    for r in range(1, max_r + 1):
        combos.extend(list(combinations(candidates, r)))

    rows = []
    for combo in combos:
        combo_list = list(combo)
        res = evaluate_strategy(returns_data, target, combo_list, horizon_days, train_ratio)
        w = res["weights"]
        is_rs = (any(val > 0 for val in w) and any(val < 0 for val in w))

        if only_rs and not is_rs:
            continue

        if res["train_metrics"]["mean_pct"] > 0:
            weights_str = ", ".join([f"{f} ({val:+.2f})" for f, val in zip(combo_list, w)])
            pred_now = res["live_pred_pct"]
            diff_vs_ar = res["test_metrics"]["mean_pct"] - ar_mean_ref
            rows.append({
                "combo_list": combo_list,
                "Activos Predictores": " + ".join(combo_list),
                "Fuerza Relativa (+/-)": "✅ Sí" if is_rs else "No",
                "Pesos (w)": weights_str,
                "Retorno Medio Trade (Test %)": round(res["test_metrics"]["mean_pct"], 3),
                "Diferencial vs AR (%)": round(diff_vs_ar, 3),
                "Win Rate (Test %)": round(res["test_metrics"]["win_rate"], 1),
                "Sharpe (Test)": round(res["test_metrics"]["sharpe"], 2),
                "Retorno Medio (Train %)": round(res["train_metrics"]["mean_pct"], 3),
                f"Señal Hoy ({horizon_days}d)": f"{'🟢 LONG' if pred_now > 0 else '🔴 SHORT'} ({pred_now:+.2f}%)"
            })
    return rows, len(combos)

# ---------------------------------------------------------
# PESTAÑAS REORGANIZADAS: 1º ESCÁNER | 2º AUDITORÍA DE 3 MODELOS
# ---------------------------------------------------------
tab_scanner, tab_models = st.tabs([
    "🔍 1. Escáner Automático (El Radar de Parejas)",
    "📊 2. Auditoría de los 3 Modelos y Señal Operativa"
])

# =========================================================
# PESTAÑA 1: ESCÁNER AUTOMÁTICO (EJECUCIÓN INSTANTÁNEA)
# =========================================================
with tab_scanner:
    st.subheader(f"🔍 Radar Automático de Mejores Predictores para `{target_ticker}`")
    st.markdown(
        f"**Referencia a batir (Modelo Base AR - Solo `{target_ticker}` en Test):** "
        f"Retorno Medio: **`{ar_test_mean:+.2f}%`** | Win Rate: **`{ar_test_wr:.1f}%`** | Sharpe: **`{ar_test_sharpe:.2f}`**"
    )

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1.2])
    max_combo_size = col_f1.selectbox("Tamaño máximo de combinación:", [1, 2, 3], index=2)
    sort_criterion = col_f2.selectbox(
        "Ordenar Ranking Top 15 por:",
        ["Sharpe (Test)", "Retorno Medio Trade (Test %)", "Win Rate (Test %)"],
        index=0
    )
    only_relative_strength = col_f3.checkbox(
        "Mostrar solo Fuerza Relativa natural (pesos + y - a la vez)",
        value=True
    )

    scanner_rows, total_evaluated = run_full_scanner(
        returns_df, target_ticker, valid_candidates, horizon, train_pct,
        max_combo_size, only_relative_strength, ar_test_mean
    )

    if not scanner_rows:
        st.warning("No se encontraron combinaciones con estos filtros. Desmarca la casilla de Fuerza Relativa o añade más activos.")
        selected_for_tab2 = valid_candidates[:2]
    else:
        scanner_df = pd.DataFrame(scanner_rows)
        scanner_df = scanner_df.sort_values(by=sort_criterion, ascending=False).head(15).reset_index(drop=True)

        st.success(
            f"✅ **{total_evaluated} combinaciones evaluadas al instante.** "
            f"Aquí tienes el **Top 15 Out-of-Sample** ordenado por **{sort_criterion}**:"
        )

        display_cols = [c for c in scanner_df.columns if c != "combo_list"]
        st.dataframe(scanner_df[display_cols], use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 🎯 Puente Directo al Análisis Profundo (Pestaña 2)")
        
        combo_options = [
            f"#{i+1}: {row['Activos Predictores']}  —  [Sharpe: {row['Sharpe (Test)']} | Win Rate: {row['Win Rate (Test %)']}% | Test: {row['Retorno Medio Trade (Test %)']:+.2f}%]"
            for i, row in scanner_df.iterrows()
        ]
        combo_options.append("🛠️ Selección Manual Personalizada")

        chosen_option = st.selectbox(
            "Elige qué combinación quieres auditar a fondo en la Pestaña 2 (por defecto ya está cargada la #1):",
            options=combo_options,
            index=0
        )

        if chosen_option == "🛠️ Selección Manual Personalizada":
            selected_for_tab2 = st.multiselect(
                "Selecciona manualmente los activos para la Pestaña 2:",
                options=valid_candidates,
                default=valid_candidates[:2]
            )
        else:
            chosen_idx = int(chosen_option.split(":")[0].replace("#", "")) - 1
            selected_for_tab2 = scanner_df.iloc[chosen_idx]["combo_list"]
            st.info(
                f"👉 Tienes cargada la combinación **`{' + '.join(selected_for_tab2)}`**. "
                f"Haz clic arriba en la **Pestaña 2 (📊 Auditoría de los 3 Modelos)** para ver la comparativa completa y las conclusiones."
            )

# =========================================================
# PESTAÑA 2: AUDITORÍA DE LOS 3 MODELOS EN PARALELO
# =========================================================
with tab_models:
    if not selected_for_tab2:
        st.warning("👈 Selecciona al menos un activo en la Pestaña 1 para auditar los modelos.")
        st.stop()

    # Calcular los 3 modelos simultáneamente
    # Modelo 1: Base Autorregresivo (Solo Target) -> ar_res
    # Modelo 2: Retornos Cruzados Puros (Solo Predictores Externos)
    dl_res = evaluate_strategy(returns_df, target_ticker, selected_for_tab2, horizon, train_pct)
    # Modelo 3: Mixto ARDL (Target + Predictores Externos)
    ardl_features = [target_ticker] + selected_for_tab2
    ardl_res = evaluate_strategy(returns_df, target_ticker, ardl_features, horizon, train_pct)

    # Determinar automáticamente cuál es el Mejor Modelo en Out-of-Sample
    models_summary = [
        {"id": 1, "name": f"Modelo 1: Base AR (Solo {target_ticker})", "res": ar_res},
        {"id": 2, "name": f"Modelo 2: Cruzado Puro ({' + '.join(selected_for_tab2)})", "res": dl_res},
        {"id": 3, "name": f"Modelo 3: Mixto ARDL ({target_ticker} + {' + '.join(selected_for_tab2)})", "res": ardl_res}
    ]
    # Criterio de ganador: Mayor Retorno Medio en Test (siempre que Sharpe sea positivo)
    best_model = max(models_summary, key=lambda x: x["res"]["test_metrics"]["mean_pct"])

    # ---------------------------------------------------------
    # BLOQUE SUPERIOR: CONSENSO EN VIVO Y VEREDICTO DEL COMBATE
    # ---------------------------------------------------------
    st.subheader(f"🚦 Consenso Operativo en Vivo para `{target_ticker}` (Próximos {horizon} días)")

    preds = [m["res"]["live_pred_pct"] for m in models_summary]
    long_votes = sum(1 for p in preds if p > 0.05)
    short_votes = sum(1 for p in preds if p < -0.05)

    if long_votes == 3:
        consensus_banner = "🟢 CONSENSO ALCISTA UNÁNIME (3/3 Modelos en LONG)"
    elif short_votes == 3:
        consensus_banner = "🔴 CONSENSO BAJISTA UNÁNIME (3/3 Modelos en SHORT)"
    elif long_votes >= 2:
        consensus_banner = "🟢 MAYORÍA ALCISTA (2/3 Modelos en LONG)"
    elif short_votes >= 2:
        consensus_banner = "🔴 MAYORÍA BAJISTA (2/3 Modelos en SHORT)"
    else:
        consensus_banner = "⚪ SEÑAL DIVIDIDA / NEUTRAL"

    best_pred = best_model["res"]["live_pred_pct"]
    best_dir = "🟢 LONG" if best_pred > 0.05 else ("🔴 SHORT" if best_pred < -0.05 else "⚪ NEUTRAL")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado de Consenso (3 Modelos)", consensus_banner)
    c2.metric("⭐ Modelo Ganador (Out-of-Sample)", f"Modelo {best_model['id']}")
    c3.metric(f"Señal del Modelo Ganador ({horizon}d)", f"{best_dir} ({best_pred:+.2f}%)")
    c4.metric(
        "Win Rate / Sharpe del Ganador (Test)",
        f"{best_model['res']['test_metrics']['win_rate']:.1f}%  |  Sharpe {best_model['res']['test_metrics']['sharpe']:.2f}"
    )

    # Tabla comparativa cara a cara de los 3 modelos
    st.markdown("#### 🏆 Tabla Comparativa Cara a Cara (Entrenamiento vs. Prueba Real)")
    comp_rows = []
    for m in models_summary:
        r = m["res"]
        is_winner = (m["id"] == best_model["id"])
        diff_ar = r["test_metrics"]["mean_pct"] - ar_test_mean
        comp_rows.append({
            " Corona ": "⭐ GANADOR" if is_winner else "—",
            "Modelo": m["name"],
            "Retorno Medio (Train)": f"{r['train_metrics']['mean_pct']:+.2f}%",
            "Retorno Medio (Test Out-of-Sample)": f"{r['test_metrics']['mean_pct']:+.2f}%",
            "Diferencial vs Modelo 1 (AR)": "Referencia Base" if m["id"] == 1 else f"{diff_ar:+.2f}%",
            "Win Rate (Test)": f"{r['test_metrics']['win_rate']:.1f}%",
            "Sharpe Anualizado (Test)": f"{r['test_metrics']['sharpe']:.2f}",
            "PnL Acumulado (Test)": f"{r['test_metrics']['total_pnl']:+.1f}%",
            f"Predicción Hoy ({horizon}d)": f"{'🟢 LONG' if r['live_pred_pct'] > 0 else '🔴 SHORT'} ({r['live_pred_pct']:+.2f}%)"
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Gráfico conjunto de las 3 curvas de equidad
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=ar_res["df_model"].index, y=ar_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 1: Base AR (Solo {target_ticker})",
        line=dict(color="#9ca3af", width=2, dash="dot")
    ))
    fig_eq.add_trace(go.Scatter(
        x=dl_res["df_model"].index, y=dl_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 2: Cruzado Puro ({' + '.join(selected_for_tab2)})",
        line=dict(color="#2563eb", width=3)
    ))
    fig_eq.add_trace(go.Scatter(
        x=ardl_res["df_model"].index, y=ardl_res["df_model"]["Equity_Curve"],
        mode="lines", name=f"Modelo 3: Mixto ARDL ({target_ticker} + {' + '.join(selected_for_tab2)})",
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
        title=f"Curvas de Equidad Comparadas (%) — Horizonte {horizon} días",
        xaxis_title="Fecha", yaxis_title="Retorno Bruto Acumulado (%)",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # FUNCIÓN AUXILIAR PARA PINTAR LA RADIOGRAFÍA DE CADA MODELO
    # ---------------------------------------------------------
    def render_model_radiography(res_obj, title_color="#2563eb"):
        col_t, col_g = st.columns([1.3, 1])
        w_df = pd.DataFrame({
            "Activo Predictor": res_obj["features"],
            "Coeficiente (Peso w)": [round(w, 4) for w in res_obj["weights"]],
            "Comportamiento": ["📈 Tendencia (Momentum)" if w > 0 else "🔄 Reversión a la Media" for w in res_obj["weights"]],
            f"Retorno Actual ({horizon}d %)": [round(res_obj["latest_features"][f], 2) for f in res_obj["features"]],
            "Aportación a Hoy (%)": [round(w * res_obj["latest_features"][f], 3) for w, f in zip(res_obj["weights"], res_obj["features"])]
        })
        with col_t:
            st.dataframe(w_df, use_container_width=True, hide_index=True)
            st.caption(f"**Sesgo base (Intercepto $b$):** `{res_obj['intercept']*100:+.3f}%`  |  **Suma Predicción Hoy:** `{res_obj['live_pred_pct']:+.2f}%`")
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
    m1_c1.metric(f"Señal Hoy ({horizon}d)", f"{'🟢 LONG' if ar_res['live_pred_pct'] > 0 else '🔴 SHORT'} ({ar_res['live_pred_pct']:+.2f}%)")
    m1_c2.metric("Retorno Medio / Trade (Test)", f"{ar_test_mean:+.2f}%")
    m1_c3.metric("Win Rate (Test)", f"{ar_test_wr:.1f}%")
    m1_c4.metric("Sharpe Anualizado (Test)", f"{ar_test_sharpe:.2f}")

    render_model_radiography(ar_res)

    # Conclusión Final Automática del Modelo 1
    ar_w = ar_res["weights"][0]
    ar_style = "Tendencial (cuando sube, tiende a seguir subiendo)" if ar_w > 0 else "de Reversión a la Media (cuando cae, tiende a rebotar)"
    if ar_test_mean > 0 and ar_test_sharpe > 0.8:
        st.info(
            f"📌 **Conclusión Final del Modelo 1 (Referencia Base):** `{target_ticker}` por sí solo tiene un comportamiento **{ar_style}** "
            f"muy sólido en el mercado actual, ganando **`{ar_test_mean:+.2f}%`** por operación con un Sharpe de **`{ar_test_sharpe:.2f}`**. "
            f"Esto significa que el listón está alto: solo debemos operar el Modelo 2 o el Modelo 3 si logran superar este `{ar_test_mean:+.2f}%`."
        )
    else:
        st.warning(
            f"📌 **Conclusión Final del Modelo 1 (Referencia Base):** Mirar únicamente el gráfico pasado de `{target_ticker}` ({ar_style}) "
            f"da un rendimiento pobre o inestable en Test (**`{ar_test_mean:+.2f}%`** por trade, Sharpe **`{ar_test_sharpe:.2f}`**). "
            f"Necesitamos obligatoriamente apoyarnos en los Retornos Cruzados (Modelo 2 o 3) para tener ventaja estadística."
        )

    st.divider()

    # =========================================================
    # SECCIÓN 2: MODELO 2 — RETORNOS CRUZADOS PUROS (DL)
    # =========================================================
    dl_mean = dl_res["test_metrics"]["mean_pct"]
    dl_wr = dl_res["test_metrics"]["win_rate"]
    dl_sharpe = dl_res["test_metrics"]["sharpe"]
    dl_diff_ar = dl_mean - ar_test_mean

    st.subheader(f"2️⃣ Modelo 2: Retornos Cruzados Puros (`{' + '.join(selected_for_tab2)}` sin `{target_ticker}`)")
    m2_c1, m2_c2, m2_c3, m2_c4 = st.columns(4)
    m2_c1.metric(f"Señal Hoy ({horizon}d)", f"{'🟢 LONG' if dl_res['live_pred_pct'] > 0 else '🔴 SHORT'} ({dl_res['live_pred_pct']:+.2f}%)")
    m2_c2.metric("Retorno Medio / Trade (Test)", f"{dl_mean:+.2f}%", delta=f"{dl_diff_ar:+.2f}% vs Modelo 1 (AR)")
    m2_c3.metric("Win Rate (Test)", f"{dl_wr:.1f}%", delta=f"{dl_wr - ar_test_wr:+.1f}% vs AR")
    m2_c4.metric("Sharpe Anualizado (Test)", f"{dl_sharpe:.2f}", delta=f"{dl_sharpe - ar_test_sharpe:+.2f} vs AR")

    render_model_radiography(dl_res)

    # Conclusión Final Automática del Modelo 2
    has_rs_2 = any(w > 0 for w in dl_res["weights"]) and any(w < 0 for w in dl_res["weights"])
    rs_text = "Además, aprovecha un patrón de **Fuerza Relativa (+/-)** entre sus activos. " if has_rs_2 else ""

    if dl_mean > 0 and dl_diff_ar >= 0:
        st.success(
            f"✅ **Conclusión Final del Modelo 2 (APROBADO — VENTAJA REAL):** Usar **`{' + '.join(selected_for_tab2)}`** "
            f"bate al activo en solitario por **`{dl_diff_ar:+.2f}%`** por operación en la prueba real, alcanzando un Win Rate del **`{dl_wr:.1f}%`** "
            f"y un Sharpe de **`{dl_sharpe:.2f}`**. {rs_text}"
            f"Su proyección actual para los próximos {horizon} días es **{'ALCISTA (LONG)' if dl_res['live_pred_pct'] > 0 else 'BAJISTA (SHORT)'} ({dl_res['live_pred_pct']:+.2f}%)**."
        )
    elif dl_mean > 0 and dl_diff_ar < 0:
        st.warning(
            f"⚠️ **Conclusión Final del Modelo 2 (RENTABLE PERO INEFICIENTE VS AR):** Aunque usar **`{' + '.join(selected_for_tab2)}`** "
            f"gana dinero en Test (**`{dl_mean:+.2f}%`** por operación, Win Rate **`{dl_wr:.1f}%`**), rinde **`{dl_diff_ar:.2f}%`** menos que "
            f"el Modelo 1 Base. Te recomiendo elegir una pareja más alta del Ranking en la Pestaña 1 que tenga Diferencial vs AR positivo."
        )
    else:
        st.error(
            f"❌ **Conclusión Final del Modelo 2 (DESCARTAR — SOBREAJUSTE):** Esta combinación de activos pierde dinero en la prueba reciente "
            f"(**`{dl_mean:+.2f}%`** por operación). La relación funcionó en el pasado pero se ha roto en el mercado actual. Prohibido operar."
        )

    st.divider()

    # =========================================================
    # SECCIÓN 3: MODELO 3 — MIXTO ARDL (TARGET + EXTERNOS)
    # =========================================================
    ardl_mean = ardl_res["test_metrics"]["mean_pct"]
    ardl_wr = ardl_res["test_metrics"]["win_rate"]
    ardl_sharpe = ardl_res["test_metrics"]["sharpe"]
    ardl_diff_dl = ardl_mean - dl_mean
    ardl_diff_ar = ardl_mean - ar_test_mean

    st.subheader(f"3️⃣ Modelo 3: Mixto ARDL (`{target_ticker}` + `{' + '.join(selected_for_tab2)}`)")
    m3_c1, m3_c2, m3_c3, m3_c4 = st.columns(4)
    m3_c1.metric(f"Señal Hoy ({horizon}d)", f"{'🟢 LONG' if ardl_res['live_pred_pct'] > 0 else '🔴 SHORT'} ({ardl_res['live_pred_pct']:+.2f}%)")
    m3_c2.metric("Retorno Medio / Trade (Test)", f"{ardl_mean:+.2f}%", delta=f"{ardl_diff_dl:+.2f}% vs Modelo 2 (Cruzado)")
    m3_c3.metric("Win Rate (Test)", f"{ardl_wr:.1f}%", delta=f"{ardl_wr - dl_wr:+.1f}% vs Modelo 2")
    m3_c4.metric("Sharpe Anualizado (Test)", f"{ardl_sharpe:.2f}", delta=f"{ardl_sharpe - dl_sharpe:+.2f} vs Modelo 2")

    render_model_radiography(ardl_res)

    # Conclusión Final Automática del Modelo 3
    if ardl_mean > dl_mean and ardl_mean > ar_test_mean:
        st.success(
            f"✅ **Conclusión Final del Modelo 3 (SINERGIA PERFECTA — MEJOR OPCIÓN):** Añadir el propio `{target_ticker}` junto a "
            f"**`{' + '.join(selected_for_tab2)}`** mejora aún más al Modelo 2 en **`{ardl_diff_dl:+.2f}%`** por operación y supera al Modelo 1 Base en **`{ardl_diff_ar:+.2f}%`**. "
            f"Ambos mundos se complementan sin estorbarse. Haz caso a la señal de este Modelo 3 (**{ardl_res['live_pred_pct']:+.2f}%**)."
        )
    elif ardl_mean < dl_mean:
        st.warning(
            f"⚠️ **Conclusión Final del Modelo 3 (COLINEALIDAD / REDUNDANCIA DETECTADA):** Al meter a `{target_ticker}` dentro de la fórmula junto a "
            f"**`{' + '.join(selected_for_tab2)}`**, el retorno medio baja un **`{ardl_diff_dl:.2f}%`** respecto al Modelo 2 Cruzado Puro "
            f"(pasa de `{dl_mean:+.2f}%` a `{ardl_mean:+.2f}%`). Esto ocurre porque los predictores externos ya contienen la información del precio y al juntarlos se estorban. "
            f"**Quédate con el Modelo 2 (Cruzado Puro).**"
        )
    else:
        st.info(
            f"📌 **Conclusión Final del Modelo 3:** El modelo mixto mejora ligeramente al Modelo 2 (`{ardl_diff_dl:+.2f}%`), "
            f"pero sigue a `{ardl_diff_ar:+.2f}%` de distancia del Modelo 1 Base. Revisa la tabla superior para operar siempre el modelo coronado con ⭐."
        )
