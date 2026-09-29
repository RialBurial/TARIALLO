import streamlit as st
import yfinance as yf
import pandas as pd
import math
import io
import requests
import plotly.graph_objects as go
from datetime import datetime
import os
import ta

# Intentamos importar la librería para generar PDFs.
try:
    from fpdf import FPDF
    fpdf_disponible = True
except ModuleNotFoundError:
    fpdf_disponible = False

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Radar Fundamental Pro", page_icon="📈", layout="wide")

# ==========================================
# INYECCIÓN DE ESTILOS CSS (DISEÑO PREMIUM NEGRO/ROJO)
# ==========================================
st.markdown("""
<style>
    header[data-testid="stHeader"] { background-color: transparent !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    .metric-card-principal {
        background-color: #1a1c23; border-left: 5px solid #E50914;
        padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        text-align: center; margin-bottom: 20px;
    }
    .metric-card-secundaria {
        background-color: #1a1c23; border-left: 3px solid #555555;
        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;
    }
    .metric-title {
        color: #a0a0a0; font-size: 14px; font-weight: bold; text-transform: uppercase;
        margin-bottom: 5px; letter-spacing: 1px;
    }
    .metric-value-xl {
        color: #E50914; font-size: 42px; font-weight: 900; margin: 5px 0; letter-spacing: 2px;
    }
    .metric-value-md { color: #ffffff; font-size: 26px; font-weight: bold; margin: 5px 0; }
    .metric-subtitle { color: #00CC96; font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

fecha_hoy = datetime.now().strftime('%Y-%m-%d')
col_precio_header = f'Precio [{fecha_hoy}]'
col_pfcf_header = 'P/FCF (Opt <22)'

ARCHIVO_AUTOGUARDADO = "Radar_Autoguardado.xlsx"

# ESTADOS DE MEMORIA
if 'portfolio_df' not in st.session_state:
    try:
        df_guardado = pd.read_excel(ARCHIVO_AUTOGUARDADO)
        st.session_state.portfolio_df = df_guardado
    except Exception:
        st.session_state.portfolio_df = pd.DataFrame()

if 'tech_df' not in st.session_state:
    st.session_state.tech_df = pd.DataFrame()

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = "llave_inicial"

# ==========================================
# FUNCIONES AUXILIARES (Análisis Fundamental)
# ==========================================
def calcular_cagr(valor_final, valor_inicial, periodos=3):
    if pd.isna(valor_final) or pd.isna(valor_inicial): return 0.0
    if valor_inicial <= 0: 
        if valor_final > 0: return 0.20 
        else: return 0.0 
    return ((valor_final / valor_inicial) ** (1 / periodos)) - 1

def calcular_nota_y_etiquetas(ratios):
    nota = 0
    tags = []
    
    if ratios.get('prueba_acida', 0) > 1: nota += 1
    if 0 <= ratios.get('deuda_patrimonio', 99) < 1: nota += 1
    if ratios.get('cobertura_intereses', 0) > 4: nota += 1
    
    if ratios.get('roic', 0) > 12: nota += 1
    if ratios.get('margen_neto', 0) > 10: nota += 1
    
    if ratios.get('margen_fcf', 0) > 10: nota += 1
    if ratios.get('utilidad_neta', 0) > 0 and ratios.get('fcf', 0) > 0:
        if (ratios['fcf'] / ratios['utilidad_neta']) > 0.8: nota += 1
        
    if ratios.get('cagr_ventas', 0) > 5: nota += 1
    if ratios.get('cagr_beneficios', 0) > 5: nota += 1
    
    if 0 < ratios.get('price_fcf', 999) < 22: nota += 1
    
    if ratios.get('deuda_patrimonio', 0) < 0.3 and ratios.get('prueba_acida', 0) > 1.5: 
        tags.append("Fortaleza Financiera")
    elif ratios.get('deuda_patrimonio', 0) > 2 or ratios.get('prueba_acida', 0) < 0.8: 
        tags.append("RIESGO LIQUIDEZ/DEUDA")
        
    if ratios.get('roic', 0) > 15 and ratios.get('margen_fcf', 0) > 15: 
        tags.append("Foso Económico (Moat)")
        
    if 0 < ratios.get('price_fcf', 0) < 12: 
        tags.append("Ganga por FCF")
    elif ratios.get('price_fcf', 0) > 30: 
        tags.append("Sobrevalorada")
        
    return nota, " | ".join(tags) if tags else "Estándar"

def calcular_dcf_rapido(fcf, growth_rate, shares, discount_rate):
    if fcf <= 0 or shares <= 0: return 0
    terminal_growth = 0.025 
    if growth_rate > 0.25: growth_rate = 0.25 
    if growth_rate < 0: growth_rate = 0.02
        
    val_presente = 0
    fcf_proyectado = fcf
    for i in range(1, 6):
        fcf_proyectado *= (1 + growth_rate)
        val_presente += fcf_proyectado / ((1 + discount_rate)**i)
        
    terminal_value = (fcf_proyectado * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    val_presente += terminal_value / ((1 + discount_rate)**5)
    return val_presente / shares

# --- FUNCIÓN BLINDADA Y HÍBRIDA ---
def obtener_datos_empresa(ticker, api_key=None):
    stock = yf.Ticker(ticker)
    try:
        balance = stock.balance_sheet
        financials = stock.financials
        cashflow = stock.cashflow
        if balance.empty or financials.empty: return None

        info = stock.info if stock.info else {}

        try:
            precio_accion = float(stock.fast_info.last_price)
            shares = float(stock.fast_info.shares)
        except:
            hist = stock.history(period="5d")
            precio_accion = float(hist['Close'].iloc[-1]) if not hist.empty else 0
            shares = info.get('sharesOutstanding', 1)

        if info.get('currency') == 'GBp': 
            precio_accion = precio_accion / 100

        nombre_empresa = info.get('longName', info.get('shortName', ticker)) if info.get('shortName') else ticker
        sector = info.get('sector', 'Desconocido')
        beta = info.get('beta', 1.0)
        
        activo_corriente = balance.loc['Current Assets'].iloc[0] if 'Current Assets' in balance.index else 0
        pasivo_corriente = balance.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance.index else 0
        inventarios = balance.loc['Inventory'].iloc[0] if 'Inventory' in balance.index else 0
        activo_total = balance.loc['Total Assets'].iloc[0]
        
        if 'Total Debt' in balance.index: deuda_total = balance.loc['Total Debt'].iloc[0]
        elif 'Total Liabilities Net Minority Interest' in balance.index: deuda_total = balance.loc['Total Liabilities Net Minority Interest'].iloc[0]
        elif 'Total Liabilities' in balance.index: deuda_total = balance.loc['Total Liabilities'].iloc[0]
        else: deuda_total = 0
            
        patrimonio = balance.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance.index else (activo_total - deuda_total)
        
        ventas = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        utilidad_neta = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
        ebit = financials.loc['Ebit'].iloc[0] if 'Ebit' in financials.index else utilidad_neta
        gastos_financieros = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 1
        
        if len(financials.columns) >= 4:
            cagr_ventas = calcular_cagr(ventas, financials.loc['Total Revenue'].iloc[3], 3) * 100
            cagr_beneficio = calcular_cagr(utilidad_neta, financials.loc['Net Income'].iloc[3], 3) * 100
        else: 
            cagr_ventas, cagr_beneficio = 0, 0

        fcf = info.get('freeCashflow', None)
        if fcf is None and not cashflow.empty:
            op_cash = cashflow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cashflow.index else 0
            capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
            fcf = op_cash + capex
            
        price_fcf = ((precio_accion * shares) / fcf) if fcf and fcf > 0 else 0
        if fcf and fcf < 0: price_fcf = -1

        div_pagados = 0
        if 'Cash Dividends Paid' in cashflow.index: div_pagados = abs(cashflow.loc['Cash Dividends Paid'].iloc[0])
        elif 'Dividends Paid' in cashflow.index: div_pagados = abs(cashflow.loc['Dividends Paid'].iloc[0])
            
        market_cap = precio_accion * shares
        div_yield = (div_pagados / market_cap) * 100 if market_cap > 0 else 0
        payout_ratio = (div_pagados / utilidad_neta) * 100 if utilidad_neta > 0 else 0

        eps = utilidad_neta / shares if shares > 1 else 0
        valor_libro_accion = patrimonio / shares if shares > 1 else 0
        valor_graham = math.sqrt(22.5 * eps * valor_libro_accion) if (eps > 0 and valor_libro_accion > 0) else 0

        est_growth = info.get('earningsGrowth', 0.05) if info.get('earningsGrowth') else 0.05
        risk_free_rate = 0.042
        market_premium = 0.055 
        discount_rate = risk_free_rate + (beta * market_premium)
        discount_rate = max(0.07, min(discount_rate, 0.15)) 
        
        valor_dcf = 0
        if api_key:
            try:
                api_key_clean = api_key.strip()
                url_dcf = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{ticker}?apikey={api_key_clean}"
                res_dcf = requests.get(url_dcf).json()
                if isinstance(res_dcf, list) and len(res_dcf) > 0 and 'dcf' in res_dcf[0]:
                    valor_dcf = float(res_dcf[0]['dcf'])
                else:
                    valor_dcf = calcular_dcf_rapido(fcf, est_growth, shares, discount_rate)
            except:
                valor_dcf = calcular_dcf_rapido(fcf, est_growth, shares, discount_rate)
        else:
            valor_dcf = calcular_dcf_rapido(fcf, est_growth, shares, discount_rate)

        valor_liquidacion = (activo_corriente - deuda_total) / shares if shares > 1 else 0

        tasa_impositiva = 0.25 
        capital_invertido = deuda_total + patrimonio
        roic = ((ebit * (1 - tasa_impositiva)) / capital_invertido) * 100 if capital_invertido > 0 else 0

        ratios_raw = {
            'prueba_acida': (activo_corriente - inventarios) / pasivo_corriente if pasivo_corriente else 0,
            'deuda_patrimonio': deuda_total / patrimonio if patrimonio > 0 else 0,
            'cobertura_intereses': ebit / abs(gastos_financieros) if gastos_financieros != 0 else 0,
            'roic': roic,
            'margen_neto': (utilidad_neta / ventas) * 100 if ventas else 0,
            'margen_fcf': (fcf / ventas) * 100 if ventas and fcf else 0,
            'utilidad_neta': utilidad_neta,
            'fcf': fcf,
            'cagr_ventas': cagr_ventas,
            'cagr_beneficios': cagr_beneficio,
            'price_fcf': price_fcf
        }
        
        nota, etiquetas = calcular_nota_y_etiquetas(ratios_raw)

        return {
            'Ticker': ticker, 'Nombre': nombre_empresa, 'Sector': sector, col_precio_header: precio_accion,
            'Valor Intrínseco (DCF)': round(valor_dcf, 2),
            'WACC %': round(discount_rate * 100, 2),
            'Beta': round(beta, 2),
            'Valor de Liquidación': round(valor_liquidacion, 2),
            'Valor Graham': round(valor_graham, 2), 'Div. Yield %': round(div_yield, 2),
            'Payout %': round(payout_ratio, 2), col_pfcf_header: round(price_fcf, 2),
            'Prueba Ácida (>1)': round(ratios_raw['prueba_acida'], 2),
            'Deuda/Patr. (<1)': round(ratios_raw['deuda_patrimonio'], 2),
            'Cob. Intereses (>4)': round(ratios_raw['cobertura_intereses'], 2),
            'ROIC % (>12%)': round(ratios_raw['roic'], 2),
            'Margen Neto % (>10%)': round(ratios_raw['margen_neto'], 2),
            'Margen FCF % (>10%)': round(ratios_raw['margen_fcf'], 2),
            'FCF/Beneficio (>0.8)': round(ratios_raw['fcf'] / ratios_raw['utilidad_neta'], 2) if ratios_raw['utilidad_neta'] > 0 else 0,
            'CAGR Ventas 3y % (>5%)': round(ratios_raw['cagr_ventas'], 2),
            'CAGR Benef. 3y % (>5%)': round(ratios_raw['cagr_beneficios'], 2),
            'FCF (Millones)': round(fcf / 1_000_000, 2) if fcf else 0,
            'Beneficio Neto (Millones)': round(utilidad_neta / 1_000_000, 2),
            'Nota (/10)': nota, 'Etiquetas': etiquetas
        }
    except Exception as e: 
        return None

def aplicar_simbolos_sectoriales(df):
    df_mod = df.copy()
    ratios_relativos = {
        'ROIC % (>12%)': True, 'Margen FCF % (>10%)': True,
        col_pfcf_header: False, 'Deuda/Patr. (<1)': False
    }
    for ratio in ratios_relativos.keys():
        if ratio in df_mod.columns: df_mod[ratio] = df_mod[ratio].astype(object)
    
    for sector in df_mod['Sector'].unique():
        if sector == 'Desconocido': continue
        filtro = df_mod['Sector'] == sector
        df_sector = df_mod[filtro]
        if len(df_sector) < 2: continue
            
        for ratio, mayor_es_mejor in ratios_relativos.items():
            valores_num = pd.to_numeric(df_sector[ratio], errors='coerce')
            if ratio == col_pfcf_header: valores_validos = valores_num[valores_num > 0]
            else: valores_validos = valores_num.dropna()
                
            if valores_validos.empty: continue
            
            media_sector = valores_validos.mean()
            mejor_valor = valores_validos.max() if mayor_es_mejor else valores_validos.min()
            margen_sup = media_sector * 1.15
            margen_inf = media_sector * 0.85
            
            for idx in df_sector.index:
                valor = df_mod.loc[idx, ratio]
                try: valor_num = float(valor)
                except: continue
                
                if pd.isna(valor_num) or (valor_num == 0 and ratio == col_pfcf_header): continue
                    
                simbolo = ""
                if valor_num == mejor_valor: simbolo = "★ TOP"
                elif mayor_es_mejor:
                    if valor_num > margen_sup: simbolo = "▲"
                    elif valor_num < margen_inf: simbolo = "▼"
                    else: simbolo = "▬"
                else: 
                    if valor_num < margen_inf: simbolo = "▲"
                    elif valor_num > margen_sup: simbolo = "▼"
                    else: simbolo = "▬"
                    
                df_mod.loc[idx, ratio] = f"{valor_num} {simbolo}".strip()
    return df_mod

def aplicar_estilos(df):
    verde = 'background-color: #0b3d0b; color: #85e085'
    rojo = 'background-color: #4a0000; color: #ff6666'
    azul_claro = 'background-color: #002244; color: #66b3ff; font-weight: bold'
    
    def limpiar_num(val):
        try: return float(str(val).split()[0])
        except: return 0.0

    df_styler = pd.DataFrame('', index=df.index, columns=df.columns)
    for i in df.index:
        df_styler.loc[i, 'Nota (/10)'] = azul_claro
        if "RIESGO" in str(df.loc[i, 'Etiquetas']): df_styler.loc[i, 'Etiquetas'] = rojo
        elif "Foso" in str(df.loc[i, 'Etiquetas']) or "Ganga" in str(df.loc[i, 'Etiquetas']) or "Fortaleza" in str(df.loc[i, 'Etiquetas']): df_styler.loc[i, 'Etiquetas'] = verde
        
        if df.loc[i, 'Valor Intrínseco (DCF)'] > df.loc[i, col_precio_header]: df_styler.loc[i, 'Valor Intrínseco (DCF)'] = verde
        elif df.loc[i, 'Valor Intrínseco (DCF)'] < df.loc[i, col_precio_header] and df.loc[i, 'Valor Intrínseco (DCF)'] > 0: df_styler.loc[i, 'Valor Intrínseco (DCF)'] = rojo

        pfcf_val = limpiar_num(df.loc[i, col_pfcf_header])
        if 0 < pfcf_val <= 22: df_styler.loc[i, col_pfcf_header] = verde
        elif pfcf_val < 0 or pfcf_val > 30: df_styler.loc[i, col_pfcf_header] = rojo

        if df.loc[i, 'Prueba Ácida (>1)'] >= 1: df_styler.loc[i, 'Prueba Ácida (>1)'] = verde
        elif df.loc[i, 'Prueba Ácida (>1)'] < 0.8: df_styler.loc[i, 'Prueba Ácida (>1)'] = rojo
        
        deuda_val = limpiar_num(df.loc[i, 'Deuda/Patr. (<1)'])
        if deuda_val < 1: df_styler.loc[i, 'Deuda/Patr. (<1)'] = verde
        elif deuda_val > 2: df_styler.loc[i, 'Deuda/Patr. (<1)'] = rojo

        if df.loc[i, 'Cob. Intereses (>4)'] > 4: df_styler.loc[i, 'Cob. Intereses (>4)'] = verde
        elif df.loc[i, 'Cob. Intereses (>4)'] < 2: df_styler.loc[i, 'Cob. Intereses (>4)'] = rojo

        roic_val = limpiar_num(df.loc[i, 'ROIC % (>12%)'])
        if roic_val >= 12: df_styler.loc[i, 'ROIC % (>12%)'] = verde
        elif roic_val < 5: df_styler.loc[i, 'ROIC % (>12%)'] = rojo

        if df.loc[i, 'Margen Neto % (>10%)'] >= 10: df_styler.loc[i, 'Margen Neto % (>10%)'] = verde
        elif df.loc[i, 'Margen Neto % (>10%)'] < 5: df_styler.loc[i, 'Margen Neto % (>10%)'] = rojo

        margen_fcf_val = limpiar_num(df.loc[i, 'Margen FCF % (>10%)'])
        if margen_fcf_val >= 10: df_styler.loc[i, 'Margen FCF % (>10%)'] = verde
        elif margen_fcf_val < 5: df_styler.loc[i, 'Margen FCF % (>10%)'] = rojo

        if df.loc[i, 'FCF/Beneficio (>0.8)'] >= 0.8: df_styler.loc[i, 'FCF/Beneficio (>0.8)'] = verde
        elif df.loc[i, 'FCF/Beneficio (>0.8)'] < 0.5: df_styler.loc[i, 'FCF/Beneficio (>0.8)'] = rojo

        if df.loc[i, 'CAGR Ventas 3y % (>5%)'] >= 5: df_styler.loc[i, 'CAGR Ventas 3y % (>5%)'] = verde
        elif df.loc[i, 'CAGR Ventas 3y % (>5%)'] < 0: df_styler.loc[i, 'CAGR Ventas 3y % (>5%)'] = rojo

        if df.loc[i, 'CAGR Benef. 3y % (>5%)'] >= 5: df_styler.loc[i, 'CAGR Benef. 3y % (>5%)'] = verde
        elif df.loc[i, 'CAGR Benef. 3y % (>5%)'] < 0: df_styler.loc[i, 'CAGR Benef. 3y % (>5%)'] = rojo

    return df_styler

def generar_pdf_tearsheet(datos, col_precio):
    pdf = FPDF()
    pdf.add_page()
    def format_text(texto):
        t = str(texto).replace("★ TOP", " [LÍDER]").replace("▲", " [SUPERIOR]").replace("▬", " [MEDIA]").replace("▼", " [INFERIOR]")
        return t.strip()

    pdf.set_fill_color(15, 37, 55)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", style="B", size=20)
    pdf.set_xy(15, 12)
    pdf.cell(0, 10, f"TEAR SHEET: {datos['Ticker']}", ln=True)
    pdf.set_font("Helvetica", style="I", size=11)
    pdf.set_xy(15, 22)
    pdf.cell(0, 10, f"Radiografia Fundamental V3 | {datos['Nombre']} | Sector: {datos['Sector']}")
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 45)
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.set_fill_color(230, 240, 250)
    pdf.cell(180, 8, "  1. RESUMEN DE VALORACION Y PRECIO", fill=True, ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(60, 6, "Precio Actual:", ln=False)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(120, 6, f"{datos[col_precio]}", ln=True)
    
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(60, 6, "Valor Intrinseco (DCF):", ln=False)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(120, 6, f"{datos['Valor Intrínseco (DCF)']} (WACC: {datos.get('WACC %', 'N/A')}%)", ln=True)

    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(60, 6, "Nota Scoring Global:", ln=False)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(120, 6, f"{datos['Nota (/10)']}/10  -  Perfil: {format_text(datos['Etiquetas'])}", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.set_fill_color(230, 240, 250)
    pdf.cell(180, 8, "  2. RADIOGRAFIA DE SALUD FINANCIERA (V3)", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", style="B", size=10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 60, 90)
    pdf.cell(100, 8, " Metrica Analizada (Rango Optimo)", border=1, fill=True)
    pdf.cell(80, 8, " Resultado y Posicion Sectorial", border=1, fill=True, ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=10)
    
    ratios_tabla = [
        ("Prueba Acida / Liquidez (> 1)", datos['Prueba Ácida (>1)']),
        ("Deuda sobre Patrimonio Neto (< 1)", datos['Deuda/Patr. (<1)']),
        ("Cobertura de Intereses Deuda (> 4)", datos['Cob. Intereses (>4)']),
        ("ROIC - Retorno Cap. Invertido % (> 12%)", datos['ROIC % (>12%)']),
        ("Margen Neto de Beneficio % (> 10%)", datos['Margen Neto % (>10%)']),
        ("Margen Free Cash Flow % (> 10%)", datos['Margen FCF % (>10%)']),
        ("Veracidad Contable FCF/Neto (> 0.8)", datos['FCF/Beneficio (>0.8)']),
        ("Crecimiento Anual Ventas 3y (> 5%)", datos['CAGR Ventas 3y % (>5%)']),
        ("Crecimiento Anual Beneficio 3y (> 5%)", datos['CAGR Benef. 3y % (>5%)']),
        ("Multiplo Valoracion - P/FCF (< 22)", datos[col_pfcf_header]),
    ]
    
    alternar = False
    for nombre, valor in ratios_tabla:
        if alternar: pdf.set_fill_color(245, 248, 252)
        else: pdf.set_fill_color(255, 255, 255)
        pdf.cell(100, 7, f" {nombre}", border=1, fill=True)
        pdf.set_font("Helvetica", style="B" if "LÍDER" in str(format_text(valor)) else "", size=10)
        pdf.cell(80, 7, f" {format_text(valor)}", border=1, fill=True, ln=True)
        pdf.set_font("Helvetica", size=10)
        alternar = not alternar

    pdf.set_y(-25)
    pdf.set_font("Helvetica", style="I", size=8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 4, f"Documento emitido por Radar Fundamental Pro el {datetime.now().strftime('%d/%m/%Y')}.", align="C", ln=True)
    
    try:
        return bytes(pdf.output()) 
    except TypeError:
        return pdf.output(dest='S').encode('latin-1') 

# ==========================================
# FUNCIONES: TÉCNICO Y FMP SCREENER
# ==========================================
@st.cache_data(show_spinner=False)
def obtener_datos_tecnicos(tickers_list):
    resultados = []
    for ticker in tickers_list:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty: continue
                
            close_prices = df['Close'].squeeze()
            high_prices = df['High'].squeeze()
            low_prices = df['Low'].squeeze()
            
            precio_actual = float(close_prices.iloc[-1])
            
            rsi = float(ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1])
            sma200 = float(ta.trend.SMAIndicator(close_prices, window=200).sma_indicator().iloc[-1])
            desviacion_sma200 = ((precio_actual - sma200) / sma200) * 100
            
            max_52s = float(high_prices.tail(252).max())
            min_52s = float(low_prices.tail(252).min())
            
            caida_desde_max = ((precio_actual - max_52s) / max_52s) * 100
            distancia_al_min = ((precio_actual - min_52s) / min_52s) * 100
            
            resultados.append({
                "Ticker": ticker,
                "Precio ($)": round(precio_actual, 2),
                "RSI": round(rsi, 2),
                "Caída Max 52S (%)": round(caida_desde_max, 2),
                "Dist. Mínimo 52S (%)": round(distancia_al_min, 2),
                "Desviación SMA 200 (%)": round(desviacion_sma200, 2)
            })
        except Exception:
            pass
    return pd.DataFrame(resultados)

# --- NUEVO SISTEMA ANTI-ERRORES PARA FMP ---
def obtener_tickers_fmp(api_key, mercado, sector, mcap, limite):
    base_url = "https://financialmodelingprep.com/api/v3/stock-screener"
    
    api_key_clean = api_key.strip()
    
    params = {
        "apikey": api_key_clean,
        "limit": limite,
        "isActivelyTrading": "true"
    }
    
    if mercado == "Wall Street (NYSE, NASDAQ)": params["exchange"] = "NYSE,NASDAQ"
    elif mercado == "Europa (EURONEXT, XETRA, LSE)": params["exchange"] = "EURONEXT,XETRA,LSE"
        
    if sector != "Todos": params["sector"] = sector
        
    if mcap == "> 2 Billones ($)": params["marketCapMoreThan"] = 2000000000
    elif mcap == "> 10 Billones ($)": params["marketCapMoreThan"] = 10000000000
        
    try:
        res = requests.get(base_url, params=params)
        data = res.json()
        
        if isinstance(data, dict) and "Error Message" in data:
            st.error(f"🛑 FMP dice: {data['Error Message']}")
            return []
            
        return [item['symbol'] for item in data] if isinstance(data, list) else []
    except Exception as e:
        st.error(f"🛑 Error de conexión con FMP: {e}")
        return []

def obtener_tickers_indice(indice):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        def extraer_columna_ticker(tabla):
            for col in tabla.columns:
                col_name = str(col).lower()
                if 'ticker' in col_name or 'symbol' in col_name:
                    lista_sucia = tabla[col].dropna().astype(str).tolist()
                    return [t.split('[')[0].strip() for t in lista_sucia]
            return None

        if indice == "S&P 500":
            html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            res = extraer_columna_ticker(tablas[0])
            if res: return res
        elif indice == "NASDAQ 100":
            urls = ['https://en.wikipedia.org/wiki/Nasdaq-100', 'https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies']
            for url in urls:
                try:
                    html = requests.get(url, headers=headers).text
                    tablas = pd.read_html(io.StringIO(html))
                    for t in tablas:
                        res = extraer_columna_ticker(t)
                        if res and len(res) > 50: return res
                except: continue
        elif indice == "DOW JONES 30":
            urls = ['https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 'https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies']
            for url in urls:
                try:
                    html = requests.get(url, headers=headers).text
                    tablas = pd.read_html(io.StringIO(html))
                    for t in tablas:
                        res = extraer_columna_ticker(t)
                        if res and len(res) > 20: return res
                except: continue
        elif indice == "IBEX 35 (España)":
            html = requests.get('https://en.wikipedia.org/wiki/IBEX_35', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            for t in tablas:
                res = extraer_columna_ticker(t)
                if res and len(res) > 20: return [f"{str(x).replace('.MC', '')}.MC" for x in res]
        elif indice == "DAX 40 (Alemania)":
            html = requests.get('https://en.wikipedia.org/wiki/DAX', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            for t in tablas:
                res = extraer_columna_ticker(t)
                if res and len(res) > 20: return [f"{str(x).replace('.', '-')}.DE" for x in res]
        elif indice == "CAC 40 (Francia)":
            html = requests.get('https://en.wikipedia.org/wiki/CAC_40', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            for t in tablas:
                res = extraer_columna_ticker(t)
                if res and len(res) > 20: return [f"{str(x).replace('.', '-')}.PA" for x in res]
        elif indice == "FTSE 100 (Reino Unido)":
            html = requests.get('https://en.wikipedia.org/wiki/FTSE_100_Index', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            for t in tablas:
                res = extraer_columna_ticker(t)
                if res and len(res) > 50: return [f"{str(x).replace('.', '-')}.L" for x in res]
        elif "SECTOR:" in indice:
            html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text
            tablas = pd.read_html(io.StringIO(html))
            df_sp = tablas[0]
            sectores_gics = {
                "SECTOR: Tecnología": "Information Technology", "SECTOR: Salud": "Health Care",
                "SECTOR: Finanzas": "Financials", "SECTOR: Consumo Cíclico": "Consumer Discretionary",
                "SECTOR: Consumo Defensivo": "Consumer Staples", "SECTOR: Energía": "Energy"
            }
            for col in df_sp.columns:
                col_name = str(col).lower()
                if 'ticker' in col_name or 'symbol' in col_name:
                    return df_sp[df_sp['GICS Sector'] == sectores_gics.get(indice)][col].tolist()
    except: return []
    return []

def escanear_lista_tickers(lista, solo_top_7, fmp_api_key=None):
    st.session_state.tech_df = pd.DataFrame() 
    resultados_nuevos = []
    progreso_texto = st.empty()
    barra_progreso = st.progress(0)
    total = len(lista)
    
    for idx, ticker in enumerate(lista):
        sufijos_europeos = [".MC", ".DE", ".PA", ".L"]
        ticker_yf = str(ticker).replace('.', '-') if "." in str(ticker) and not any(str(ticker).endswith(s) for s in sufijos_europeos) else ticker
        
        if ticker_yf in st.session_state.portfolio_df.get('Ticker', pd.Series()).values: continue
            
        progreso_texto.write(f"⏳ Analizando {idx+1}/{total}: **{ticker_yf}**...")
        barra_progreso.progress((idx + 1) / total)
        
        datos = obtener_datos_empresa(ticker_yf, fmp_api_key)
        if datos:
            if solo_top_7 and datos['Nota (/10)'] < 7: continue
            resultados_nuevos.append(datos)
            
    progreso_texto.empty()
    barra_progreso.empty()
    
    if resultados_nuevos:
        nuevo_df = pd.DataFrame(resultados_nuevos)
        st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, nuevo_df], ignore_index=True)
        try: st.session_state.portfolio_df.to_excel(ARCHIVO_AUTOGUARDADO, index=False)
        except: pass
        st.success(f"💥 ¡Escaneo completado! Añadidas {len(resultados_nuevos)} empresas.")
    else:
        st.warning("El escaneo terminó. Ninguna empresa superó el filtro.")

# ==========================================
# INTERFAZ DE USUARIO (UI) - PANEL LATERAL
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #E50914;'>RADAR PRO V3</h2>", unsafe_allow_html=True)
    
    st.header("🔑 Conexión FMP")
    api_key = st.text_input("API Key (FMP):", type="password", help="Tu clave gratuita de financialmodelingprep.com")
    
    st.divider()
    
    with st.expander("📡 Screener Institucional (FMP)", expanded=True):
        if api_key:
            fmp_mercado = st.selectbox("Mercado:", ["Wall Street (NYSE, NASDAQ)", "Europa (EURONEXT, XETRA, LSE)", "Cualquiera"])
            fmp_sector = st.selectbox("Sector:", ["Todos", "Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Energy", "Industrials"])
            fmp_mcap = st.selectbox("Tamaño Empresa:", ["> 2 Billones ($)", "> 10 Billones ($)", "Cualquiera"])
            fmp_limite = st.slider("Máximo de resultados a extraer:", 10, 100, 40)
            fmp_filtro_activo = st.checkbox("Filtrar solo Nota >= 7", value=True, key="fmp_filtro")

            if st.button("🚀 Escanear con FMP", type="primary", use_container_width=True):
                tickers_fmp = obtener_tickers_fmp(api_key, fmp_mercado, fmp_sector, fmp_mcap, fmp_limite)
                if tickers_fmp: 
                    escanear_lista_tickers(tickers_fmp, fmp_filtro_activo, api_key)
        else:
            st.warning("Introduce tu API Key arriba para desbloquear el screener avanzado.")

    with st.expander("⚙️ Índices Clásicos (Wikipedia)"):
        filtro_activo = st.checkbox("Filtrar solo Nota >= 7", value=True, key="wiki_filtro")
        opciones_radar = ["S&P 500", "NASDAQ 100", "DOW JONES 30", "IBEX 35 (España)", "DAX 40 (Alemania)", "CAC 40 (Francia)"]
        indice_seleccionado = st.selectbox("Índice:", opciones_radar)
        if st.button(f"🔍 Escanear Índice", use_container_width=True):
            tickers = obtener_tickers_indice(indice_seleccionado)
            if tickers: escanear_lista_tickers(tickers, filtro_activo, api_key)
            
    with st.expander("🔎 Búsqueda Manual"):
        ticker_input = st.text_input("Tickers (ej. AAPL, SAN.MC):").upper().strip()
        if st.button("Añadir Empresa", use_container_width=True):
            if ticker_input:
                lista_tickers = [t.strip() for t in ticker_input.replace(',', ' ').split() if t.strip()]
                escanear_lista_tickers(lista_tickers, solo_top_7=False, fmp_api_key=api_key)
            else:
                st.error("Introduce al menos un ticker.")
                
    with st.expander("🎯 Filtro Final TradingView"):
        tv_input = st.text_area("Pega tu lista final:", help="Ejemplo: NYSE:DECK, NASDAQ:NFLX, BME:SAN")
        if st.button("Filtrar Radar", use_container_width=True):
            if tv_input and not st.session_state.portfolio_df.empty:
                clean_tv_tickers = [t.split(':')[-1].strip().upper() for t in tv_input.split(',') if t.strip()]
                
                def match_ticker(yf_ticker, tv_list):
                    yf_exact = str(yf_ticker).upper()
                    yf_base_dot = yf_exact.split('.')[0]
                    yf_base_dash = yf_exact.split('-')[0]
                    
                    for tv_t in tv_list:
                        tv_exact = tv_t.upper()
                        if yf_exact == tv_exact or yf_base_dot == tv_exact or yf_base_dash == tv_exact:
                            return True
                    return False

                filtro = st.session_state.portfolio_df['Ticker'].apply(lambda x: match_ticker(x, clean_tv_tickers))
                st.session_state.portfolio_df = st.session_state.portfolio_df[filtro].reset_index(drop=True)
                
                st.session_state.tech_df = pd.DataFrame() 
                try: st.session_state.portfolio_df.to_excel(ARCHIVO_AUTOGUARDADO, index=False)
                except: pass
                st.rerun()
            elif st.session_state.portfolio_df.empty:
                st.warning("El radar fundamental está vacío.")
                
    with st.expander("🛠️ Gestión de Datos"):
        archivo_subido = st.file_uploader("📤 Cargar Portafolio (Excel)", type=['xlsx'], key=st.session_state.uploader_key)
        if archivo_subido is not None:
            try:
                df_cargado = pd.read_excel(archivo_subido)
                if 'Ticker' in df_cargado.columns:
                    df_cargado.rename(columns=lambda x: col_precio_header if str(x).startswith('Precio [') else x, inplace=True)
                    st.session_state.portfolio_df = df_cargado
                    st.success("✅ ¡Cargado con éxito!")
                else:
                    st.error("❌ Formato incorrecto.")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("🗑️ Vaciar Radar", type="primary", use_container_width=True):
            st.session_state.portfolio_df = pd.DataFrame()
            st.session_state.tech_df = pd.DataFrame()
            try:
                if os.path.exists(ARCHIVO_AUTOGUARDADO): os.remove(ARCHIVO_AUTOGUARDADO)
            except Exception: pass
            st.session_state.uploader_key = str(datetime.now())
            st.rerun()

# ==========================================
# INTERFAZ DE USUARIO (UI) - PANTALLA PRINCIPAL
# ==========================================
if not st.session_state.portfolio_df.empty:
    df_mostrar = st.session_state.portfolio_df.sort_values(by='Nota (/10)', ascending=False).reset_index(drop=True)
    df_mostrar = aplicar_simbolos_sectoriales(df_mostrar)
    
    mejor_empresa = df_mostrar.iloc[0]['Ticker']
    mejor_nota = df_mostrar.iloc[0]['Nota (/10)']
    
    st.markdown(f"""
    <div class="metric-card-principal">
        <div class="metric-title">🏆 TOP PICK / Mejor Puntuada</div>
        <div class="metric-value-xl">{mejor_empresa}</div>
        <div class="metric-subtitle">Puntuación de Salud Fundamental: {mejor_nota}/10</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
        <div class="metric-card-secundaria">
            <div class="metric-title">📅 Fecha del Análisis</div>
            <div class="metric-value-md">{fecha_hoy}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div class="metric-card-secundaria">
            <div class="metric-title">🏢 Empresas Analizadas (Filtro Fundamental)</div>
            <div class="metric-value-md">{len(df_mostrar)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Tabla de Resultados", "🔬 Análisis Profundo (Tear Sheet)", "📈 Screener Técnico (Timing)"])
    
    with tab1:
        df_estilizado = df_mostrar.style.apply(aplicar_estilos, axis=None)
        
        st.dataframe(
            df_estilizado, 
            use_container_width=True, height=450, 
            column_config={
                "Ticker": st.column_config.TextColumn(pinned=True),
                "Nombre": st.column_config.TextColumn(pinned=True),
                "Valor Intrínseco (DCF)": st.column_config.NumberColumn(help="Calculado vía FMP API si la llave está activa."),
                "Beta": st.column_config.NumberColumn(help="Volatilidad respecto al mercado. >1 es más volátil."),
                "WACC %": st.column_config.NumberColumn(help="Tasa de descuento dinámica calculada por CAPM."),
                "Prueba Ácida (>1)": st.column_config.Column(help="Liquidez sin inventarios."),
                "Deuda/Patr. (<1)": st.column_config.Column(help="Apalancamiento financiero real."),
                "Cob. Intereses (>4)": st.column_config.Column(help="Capacidad de pago de deuda."),
                "ROIC % (>12%)": st.column_config.Column(help="Retorno sobre Capital Invertido."),
                "Margen Neto % (>10%)": st.column_config.Column(help="Poder de fijación de precios."),
                "Margen FCF % (>10%)": st.column_config.Column(help="Conversión de ventas en caja libre."),
                "FCF/Beneficio (>0.8)": st.column_config.Column(help="Veracidad contable del beneficio."),
                "CAGR Ventas 3y % (>5%)": st.column_config.Column(help="Crec. compuesto de ventas."),
                "CAGR Benef. 3y % (>5%)": st.column_config.Column(help="Crec. compuesto de beneficios."),
                col_pfcf_header: st.column_config.Column(help="Valoración por múltiplos de FCF."),
                "Nota (/10)": st.column_config.Column(help="Score V3 Integral."),
                "Etiquetas": st.column_config.Column(help="Clasificación automática del algoritmo.")
            }
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
            df_estilizado.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Descargar Tabla en Excel", 
            data=buffer_excel.getvalue(), 
            file_name=f"Radar_V3_{fecha_hoy}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    with tab2:
        opciones_detalles = [f"{row['Ticker']} - {row['Nombre']}" for idx, row in df_mostrar.iterrows()]
        seleccion = st.selectbox("Selecciona empresa:", opciones_detalles)
        ticker_seleccionado = seleccion.split(" - ")[0]
        datos_seleccionados = df_mostrar[df_mostrar['Ticker'] == ticker_seleccionado].iloc[0]
        
        col_graf_1, col_graf_2 = st.columns([1, 1.5])
        
        with col_graf_1:
            st.subheader("⚖️ Equilibrio Fundamental V3")
            st.plotly_chart(plot_radar(datos_seleccionados), use_container_width=True)
            
            dcf_val = datos_seleccionados['Valor Intrínseco (DCF)']
            precio_act = datos_seleccionados[col_precio_header]
            wacc = datos_seleccionados.get('WACC %', 9.0)
            
            if dcf_val > 0:
                margen_seguridad = ((dcf_val - precio_act) / dcf_val) * 100
                color_delta = "normal" if margen_seguridad > 0 else "inverse"
                st.metric(label=f"Valor Justo FMP (WACC {wacc}%)", value=f"{dcf_val:.2f}", delta=f"{margen_seguridad:.1f}% Margen", delta_color=color_delta)
            else: st.metric(label="Valor Justo", value="N/A")
            
            if fpdf_disponible:
                pdf_data = generar_pdf_tearsheet(datos_seleccionados, col_precio_header)
                st.download_button(label="📄 Informe PDF (V3)", data=pdf_data, file_name=f"{ticker_seleccionado}_V3.pdf", mime="application/pdf")
            
        with col_graf_2:
            st.subheader("📊 Tendencia (5 Años)")
            fig_historico = plot_historico(ticker_seleccionado)
            if fig_historico: st.plotly_chart(fig_historico, use_container_width=True)

    with tab3:
        st.subheader("⏱️ Timing de Compra (Análisis Técnico)")
        st.markdown("Busca el mejor momento de entrada para las empresas que **ya han superado tu filtro fundamental**.")
        
        tickers_fundamentales = df_mostrar['Ticker'].tolist()
        
        if st.button("🚀 Calcular Indicadores Técnicos para las empresas actuales", type="primary"):
            with st.spinner("Descargando históricos (2 años) y calculando RSI y Medias Móviles..."):
                st.session_state.tech_df = obtener_datos_tecnicos(tickers_fundamentales)
        
        if not st.session_state.tech_df.empty:
            df_tech = st.session_state.tech_df.copy()
            
            empresas_menos20 = len(df_tech[df_tech['Caída Max 52S (%)'] <= -20])
            empresas_menos30 = len(df_tech[df_tech['Caída Max 52S (%)'] <= -30])
            
            col_kpi1, col_kpi2 = st.columns(2)
            with col_kpi1: st.info(f"📉 **{empresas_menos20}** Valores a más de -20% de máximos")
            with col_kpi2: st.error(f"🚨 **{empresas_menos30}** Valores a más de -30% de máximos")
            
            st.divider()
            
            st.subheader("🛠️ Ajusta tus Criterios de Entrada")
            f_col1, f_col2, f_col3 = st.columns(3)
            
            with f_col1:
                filtro_caida = st.selectbox("Caída desde Máximos (52S):", ["Mostrar Todos", "Solo caídas > 20%", "Solo caídas > 30%"])
            with f_col2:
                max_rsi = st.slider("RSI Máximo (Buscar sobreventa):", 0, 100, 100)
            with f_col3:
                filtro_sma = st.checkbox("Precios por DEBAJO de la SMA 200 (Tendencia bajista corto plazo)")

            if filtro_caida == "Solo caídas > 20%": df_tech = df_tech[df_tech['Caída Max 52S (%)'] <= -20]
            elif filtro_caida == "Solo caídas > 30%": df_tech = df_tech[df_tech['Caída Max 52S (%)'] <= -30]
                
            df_tech = df_tech[df_tech['RSI'] <= max_rsi]
            
            if filtro_sma: df_tech = df_tech[df_tech['Desviación SMA 200 (%)'] < 0]
                
            def aplicar_colores_tech(columna):
                colores = []
                for valor in columna:
                    if columna.name == 'RSI':
                        color = 'background-color: #0b3d0b; color: #85e085' if valor < 30 else 'background-color: #4a0000; color: #ff6666' if valor > 70 else ''
                    elif columna.name == 'Caída Max 52S (%)':
                        color = 'background-color: #0b3d0b; color: #85e085' if valor <= -30 else 'color: #85e085' if valor <= -20 else 'background-color: #4a0000; color: #ff6666' if valor >= 0 else ''
                    elif columna.name == 'Dist. Mínimo 52S (%)':
                        color = 'background-color: #0b3d0b; color: #85e085' if valor <= 5 else '' 
                    elif columna.name == 'Desviación SMA 200 (%)':
                        color = 'background-color: #0b3d0b; color: #85e085' if valor < 0 else 'background-color: #4a0000; color: #ff6666'
                    else:
                        color = ''
                    colores.append(color)
                return colores

            st.write(f"**Resultados Técnicos:** {len(df_tech)} valores cumplen las condiciones.")
            
            if not df_tech.empty:
                df_tech_estilizado = df_tech.style.apply(aplicar_colores_tech, axis=0)
                st.dataframe(df_tech_estilizado, use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                buffer_excel_tech = io.BytesIO()
                with pd.ExcelWriter(buffer_excel_tech, engine='openpyxl') as writer:
                    df_tech_estilizado.to_excel(writer, index=False)
                
                tickers_aprobados = df_tech['Ticker'].tolist()
                df_fundamental_filtrado = df_mostrar[df_mostrar['Ticker'].isin(tickers_aprobados)].reset_index(drop=True)
                df_fundamental_estilizado = df_fundamental_filtrado.style.apply(aplicar_estilos, axis=None)
                
                buffer_excel_fund = io.BytesIO()
                with pd.ExcelWriter(buffer_excel_fund, engine='openpyxl') as writer:
                    df_fundamental_estilizado.to_excel(writer, index=False)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label="📥 Descargar Resultados Técnicos", 
                        data=buffer_excel_tech.getvalue(), 
                        file_name=f"Radar_Tecnico_{fecha_hoy}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col_btn2:
                    st.download_button(
                        label="📥 Descargar Fundamental de las Elegidas", 
                        data=buffer_excel_fund.getvalue(), 
                        file_name=f"Radar_Fundamental_Filtrado_{fecha_hoy}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Ninguna empresa cumple con estos filtros técnicos tan estrictos.")

else: 
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="color: #E50914; font-size: 60px;">RADAR PRO V3</h1>
        <p style="color: #888888; font-size: 20px;">Tu radar está completamente vacío. Busca empresas manualmente o escanea un índice.</p>
    </div>
    """, unsafe_allow_html=True)
