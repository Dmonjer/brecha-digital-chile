"""
Script de preparación e integración de datos para el proyecto
"Análisis de la Brecha Digital y Calidad de Red en Chile"

Integra: SUBTEL fija + SUBTEL antenas móviles (comunal) + SUBTEL móvil nacional + INE + CASEN
Filtrado a macrozona Centro-Norte: Coquimbo (4), Valparaíso (5), Metropolitana (13)

Ejecutar con: py -3 src/preparar_datos.py
"""
import sys
import io
import os
import unicodedata
import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

BASE_DIR      = Path(__file__).parent.parent
SUBTEL_PATH   = BASE_DIR / "data" / "raw" / "subtel" / "series_internet_fija_subtel.xlsx"
ANTENAS_PATH  = BASE_DIR / "data" / "raw" / "subtel" / "estaciones_base_antenas.xlsx"
MOVIL_PATH    = BASE_DIR / "data" / "raw" / "subtel" / "series_internet_movil_subtel.xlsx"
DATA_PROC     = BASE_DIR / "data" / "processed"
OUT_PATH      = DATA_PROC / "dataset_comunas_chile.csv"

# Macrozona Centro-Norte: Regiones 4 (Coquimbo), 5 (Valparaíso), 13 (Metropolitana)
REGIONES_MACROZONA     = [4, 5, 13]
REGIONES_MACROZONA_STR = ['IV', 'V', 'RM']

DATA_PROC.mkdir(parents=True, exist_ok=True)

# Diccionario de sinonimia de nombres de comunas
SYNONYMS = {
    'paiguano':  'paihuano',
    'aysen':     'aisen',
    'tiltil':    'til til',
    'llayllay':  'llay llay',
    'treguaco':  'trehuaco',
}


def norm(s):
    """Normaliza texto eliminando acentos, caracteres especiales y espacios sobrantes."""
    if not isinstance(s, str):
        return ''
    s = s.strip().lower()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def extraer_subtel(path):
    """Extrae conexiones fijas residenciales por comuna de SUBTEL."""
    print("Extrayendo datos SUBTEL fija por comuna...")
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['7.11.1.CO_FIJAS_RES_COMUNA']

    subtel_dict = {}
    for r in range(10, ws.max_row + 1):
        com = ws.cell(r, 54).value
        if com:
            com_str = str(com).strip()
            if not com_str.startswith('Total') and com_str not in ['Sin clasificacion', 'None']:
                # Columnas 115-126: 12 meses de 2024
                c_2024 = [ws.cell(r, c).value for c in range(115, 127) if isinstance(ws.cell(r, c).value, (int, float))]
                prom_2024 = sum(c_2024) / len(c_2024) if c_2024 else None
                # Columnas 103-114: 2023
                c_2023 = [ws.cell(r, c).value for c in range(103, 115) if isinstance(ws.cell(r, c).value, (int, float))]
                prom_2023 = sum(c_2023) / len(c_2023) if c_2023 else None
                # Columna 144: dato más reciente
                reciente = ws.cell(r, 144).value
                reciente = float(reciente) if isinstance(reciente, (int, float)) else None

                key = norm(com_str)
                subtel_dict[key] = {
                    'conexiones_recientes': reciente,
                    'conexiones_prom2024': prom_2024,
                    'conexiones_prom2023': prom_2023
                }

    print(f"  {len(subtel_dict)} comunas extraídas de SUBTEL fija")
    return subtel_dict


def extraer_antenas_movil(path):
    """
    Extrae antenas (estaciones base) de telefonía e internet móvil por comuna
    desde el catastro oficial de SUBTEL (estaciones_base_antenas.xlsx).
    Permite calcular infraestructura móvil real a nivel comunal.
    """
    print("Extrayendo datos de infraestructura móvil (antenas/estaciones base) por comuna...")
    df_ant = pd.read_excel(path, sheet_name=0)

    # Identificar columna de región
    col_reg = [c for c in df_ant.columns if 'reg' in norm(c)][0]
    df_macro = df_ant[df_ant[col_reg].isin(REGIONES_MACROZONA_STR)].copy()
    df_macro['comuna_norm'] = df_macro['Comuna'].apply(norm)

    # Resumen por comuna: conteo de antenas y operadores únicos
    antenas_dict = {}
    grouped = df_macro.groupby('comuna_norm')
    for com, group in grouped:
        antenas_dict[com] = {
            'antenas_movil': len(group),
            'operadores_movil': group['Empresa'].nunique()
        }

    print(f"  {len(antenas_dict)} comunas con infraestructura móvil registradas en macrozona")
    print(f"  Total antenas en macrozona: {len(df_macro):,}")
    return antenas_dict


def extraer_movil_nacional(path):
    """
    Extrae estadísticas nacionales de internet móvil desde SUBTEL (Hoja 8.1.CO_TEC_MOVIL).
    Sirve como referencia macroeconómica/contexto nacional.
    """
    print("Extrayendo estadísticas nacionales de internet móvil...")
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['8.1.CO_TEC_MOVIL']
    rows = list(ws.iter_rows(values_only=True))

    year = None
    data_2024 = []
    for row in rows[7:]:
        if row[1] is not None:
            year = row[1]
        if year == 2024 and row[2] is not None:
            data_2024.append({
                'mes': row[2],
                'total_2g': row[3],
                'total_3g': row[4],
                'total_4g': row[5],
                'total_5g': row[6],
                'total_movil': row[7],
            })

    if not data_2024:
        print("  Advertencia: no se encontraron datos 2024 en hoja móvil")
        return {}

    total_movil_prom = np.mean([r['total_movil'] for r in data_2024 if r['total_movil']])
    total_4g_prom    = np.mean([r['total_4g']    for r in data_2024 if r['total_4g']])
    total_5g_prom    = np.mean([r['total_5g']    for r in data_2024 if r['total_5g']])

    resultado = {
        'movil_prom2024_nacional':    float(total_movil_prom),
        'movil_4g_prom2024_nacional': float(total_4g_prom),
        'movil_5g_prom2024_nacional': float(total_5g_prom) if total_5g_prom else 0.0,
    }
    print(f"  Promedio nacional móvil 2024: {total_movil_prom:,.0f} accesos")
    return resultado


def actualizar_dataset(subtel_dict, antenas_dict=None, movil_nacional=None):
    """
    Integra datos SUBTEL fija, infraestructura móvil comunal (antenas),
    contexto nacional y filtra a Macrozona Centro-Norte (Coquimbo, Valparaíso, RM).
    """
    print("\nIntegrando con dataset base (INE + CASEN)...")
    df = pd.read_csv(OUT_PATH)

    # Deduplicar por cut_comuna
    df = df.drop_duplicates(subset=['cut_comuna'], keep='first').sort_values('cut_comuna').reset_index(drop=True)

    # --- 1. Conexiones fijas por comuna ---
    recientes_list, prom2024_list, prom2023_list = [], [], []
    for idx, row in df.iterrows():
        k = norm(row['nombre_comuna'])
        if k in SYNONYMS:
            k = SYNONYMS[k]
        data = subtel_dict.get(k, {})
        recientes_list.append(data.get('conexiones_recientes', None))
        prom2024_list.append(data.get('conexiones_prom2024', None))
        prom2023_list.append(data.get('conexiones_prom2023', None))

    df['conexiones_recientes'] = recientes_list
    df['conexiones_prom2024']  = prom2024_list
    df['conexiones_prom2023']  = prom2023_list
    df['conexiones_x1000hab']  = (df['conexiones_prom2024'] / df['poblacion_2024']) * 1000

    # --- 2. Infraestructura móvil real por comuna (Antenas) ---
    if antenas_dict:
        antenas_list, operadores_list = [], []
        for idx, row in df.iterrows():
            k = norm(row['nombre_comuna'])
            if k in SYNONYMS:
                k = SYNONYMS[k]
            data = antenas_dict.get(k, {'antenas_movil': 0, 'operadores_movil': 0})
            antenas_list.append(data['antenas_movil'])
            operadores_list.append(data['operadores_movil'])

        df['antenas_movil']     = antenas_list
        df['operadores_movil']  = operadores_list
        df['antenas_x1000hab']  = (df['antenas_movil'] / df['poblacion_2024']) * 1000

        # Ratio móvil/fija real comunal: densidad de antenas móviles vs conexiones fijas por habitante
        # Mide la preponderancia relativa de la infraestructura celular respecto a la red fija residencial
        df['ratio_movil_fija'] = np.where(
            df['conexiones_x1000hab'] > 0,
            ((df['antenas_x1000hab'] / df['conexiones_x1000hab']) * 100).round(4),
            np.nan
        )
        print("  Infraestructura móvil por comuna integrada exitosamente (antenas, densidad, operadores, ratio real)")
    else:
        print("  Advertencia: sin datos de antenas móviles comunales.")

    # --- 3. Filtrar Macrozona Centro-Norte ---
    n_antes = len(df)
    df = df[df['num_region'].isin(REGIONES_MACROZONA)].reset_index(drop=True)
    print(f"  Filtro macrozona: {n_antes} -> {len(df)} comunas (regiones {REGIONES_MACROZONA})")

    # Limpieza de columnas obsoletas con varianza cero si existían
    cols_drop_cero_var = ['movil_x1000hab_nacional', 'movil_4g_x1000hab', 'movil_5g_x1000hab']
    df = df.drop(columns=[c for c in cols_drop_cero_var if c in df.columns], errors='ignore')

    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"  Guardado: {OUT_PATH} con {len(df)} comunas de Macrozona Centro-Norte")
    print(f"  Nulos en conexiones_prom2024: {df['conexiones_prom2024'].isnull().sum()}")
    print(f"  Nulos en antenas_movil: {df['antenas_movil'].isnull().sum()}")
    return df


def main():
    print("=" * 60)
    print("  PREPARACIÓN DE DATOS - BRECHA DIGITAL CHILE")
    print("  Macrozona Centro-Norte: Coquimbo (4), Valparaíso (5), RM (13)")
    print("=" * 60)

    if not os.path.exists(SUBTEL_PATH):
        print(f"Error: No se encontró {SUBTEL_PATH}.")
        return

    subtel_dict = extraer_subtel(SUBTEL_PATH)

    antenas_dict = None
    if os.path.exists(ANTENAS_PATH):
        antenas_dict = extraer_antenas_movil(ANTENAS_PATH)
    else:
        print(f"Advertencia: Archivo de antenas no encontrado en {ANTENAS_PATH}.")

    movil_nacional = None
    if os.path.exists(MOVIL_PATH):
        movil_nacional = extraer_movil_nacional(MOVIL_PATH)

    df = actualizar_dataset(subtel_dict, antenas_dict, movil_nacional)

    print("\nResumen por región en dataset procesado:")
    print(df.groupby('num_region')[['nombre_comuna', 'antenas_movil', 'conexiones_prom2024']].agg({
        'nombre_comuna': 'count',
        'antenas_movil': 'sum',
        'conexiones_prom2024': 'sum'
    }).rename(columns={'nombre_comuna': 'n_comunas'}).to_string())


if __name__ == "__main__":
    main()
