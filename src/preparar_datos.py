"""
Script de preparación e integración de datos para el proyecto
"Análisis de la Brecha Digital y Calidad de Red en Chile"

Integra: SUBTEL fija + SUBTEL móvil (nacional) + INE + CASEN
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

BASE_DIR    = Path(__file__).parent.parent
SUBTEL_PATH = BASE_DIR / "data" / "raw" / "subtel" / "accesos_internet_fijo_por_comuna.xlsx"
MOVIL_PATH  = BASE_DIR / "data" / "raw" / "subtel" / "series_internet_movil_subtel.xlsx"
DATA_PROC   = BASE_DIR / "data" / "processed"
OUT_PATH    = DATA_PROC / "dataset_comunas_chile.csv"

# Macrozona Centro-Norte
REGIONES_MACROZONA = [4, 5, 13]  # Coquimbo, Valparaíso, Metropolitana

DATA_PROC.mkdir(parents=True, exist_ok=True)

SYNONYMS = {
    'paiguano': 'paihuano',
    'aysen': 'aisen',
    'tiltil': 'til til',
    'llayllay': 'llay llay',
    'treguaco': 'trehuaco',
}


def norm(s):
    """Normaliza texto eliminando acentos y espacios sobrantes."""
    if not isinstance(s, str):
        return ''
    s = s.strip().lower()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def extraer_subtel(path):
    """Extrae conexiones fijas residenciales por comuna de SUBTEL."""
    print("Extrayendo datos SUBTEL fija...")
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
                # Columna 144: más reciente (Jun 2026)
                reciente = ws.cell(r, 144).value
                reciente = float(reciente) if isinstance(reciente, (int, float)) else None

                key = norm(com_str)
                subtel_dict[key] = {
                    'conexiones_recientes': reciente,
                    'conexiones_prom2024': prom_2024,
                    'conexiones_prom2023': prom_2023
                }

    print(f"  {len(subtel_dict)} comunas extraidas de SUBTEL fija")
    return subtel_dict


def extraer_movil_nacional(path):
    """
    Extrae estadísticas nacionales de internet móvil desde SUBTEL.
    Hoja 8.1.CO_TEC_MOVIL: conexiones por tecnología (2G, 3G, 4G, 5G).
    Devuelve promedios 2024 y datos recientes.
    """
    print("Extrayendo datos SUBTEL movil (nivel nacional)...")
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
                'total_3g_4g_5g': row[8],
            })

    if not data_2024:
        print("  Advertencia: no se encontraron datos 2024 en hoja movil")
        return {}

    # Promedios 2024 nacionales
    total_movil_prom = np.mean([r['total_movil'] for r in data_2024 if r['total_movil']])
    total_4g_prom    = np.mean([r['total_4g']    for r in data_2024 if r['total_4g']])
    total_5g_prom    = np.mean([r['total_5g']    for r in data_2024 if r['total_5g']])
    total_3g_prom    = np.mean([r['total_3g']    for r in data_2024 if r['total_3g']])

    # Dato más reciente (Dic 2024)
    total_movil_reciente = data_2024[-1]['total_movil']
    total_4g_reciente    = data_2024[-1]['total_4g']
    total_5g_reciente    = data_2024[-1]['total_5g']

    resultado = {
        'movil_prom2024_nacional':    float(total_movil_prom),
        'movil_4g_prom2024_nacional': float(total_4g_prom),
        'movil_5g_prom2024_nacional': float(total_5g_prom) if total_5g_prom else 0.0,
        'movil_3g_prom2024_nacional': float(total_3g_prom),
        'movil_reciente_nacional':    float(total_movil_reciente),
    }
    print(f"  Promedio conexiones moviles 2024 (nacional): {total_movil_prom:,.0f}")
    print(f"  4G prom: {total_4g_prom:,.0f} | 5G prom: {total_5g_prom:,.0f}")
    return resultado


def actualizar_dataset(subtel_dict, movil_nacional=None):
    """
    Aplica datos SUBTEL (fija + indicadores móviles) y filtra
    a la macrozona Centro-Norte (Coquimbo, Valparaíso, RM).
    """
    print("\nIntegrando con dataset base (INE + CASEN)...")
    df = pd.read_csv(OUT_PATH)

    # Deduplicar
    df = df.drop_duplicates(subset=['cut_comuna'], keep='first').sort_values('cut_comuna').reset_index(drop=True)

    # --- Datos fijos por comuna ---
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

    # --- Indicadores móviles nacionales → por habitante ---
    if movil_nacional:
        pob_nacional = df['poblacion_2024'].sum()
        df['movil_x1000hab_nacional'] = (movil_nacional['movil_prom2024_nacional'] / pob_nacional) * 1000
        df['movil_4g_x1000hab']       = (movil_nacional['movil_4g_prom2024_nacional'] / pob_nacional) * 1000
        df['movil_5g_x1000hab']       = (movil_nacional['movil_5g_prom2024_nacional'] / pob_nacional) * 1000
        # Ratio móvil/fija: indica cobertura relativa por comuna
        df['ratio_movil_fija'] = np.where(
            df['conexiones_prom2024'] > 0,
            (df['movil_x1000hab_nacional'] / df['conexiones_x1000hab']).round(4),
            np.nan
        )
        print(f"  Indicadores moviles integrados (penetracion nacional normalizada)")
    else:
        print("  Sin datos moviles disponibles.")

    # --- Filtrar macrozona Centro-Norte ---
    n_antes = len(df)
    df = df[df['num_region'].isin(REGIONES_MACROZONA)].reset_index(drop=True)
    print(f"  Filtro macrozona: {n_antes} -> {len(df)} comunas (regiones {REGIONES_MACROZONA})")

    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"  Guardado: {OUT_PATH} con {len(df)} comunas de macrozona Centro-Norte")
    print(f"  Nulos en conexiones_prom2024: {df['conexiones_prom2024'].isnull().sum()}")
    return df


def main():
    print("=" * 60)
    print("  PREPARACION DE DATOS - BRECHA DIGITAL CHILE")
    print("  Macrozona Centro-Norte: Coquimbo / Valparaiso / RM")
    print("=" * 60)

    if not os.path.exists(SUBTEL_PATH):
        print(f"No se encontro {SUBTEL_PATH}. Ejecutar src/descarga_datos.py primero.")
        return

    subtel_dict = extraer_subtel(SUBTEL_PATH)

    movil_nacional = None
    if os.path.exists(MOVIL_PATH):
        movil_nacional = extraer_movil_nacional(MOVIL_PATH)
    else:
        print(f"Archivo movil no encontrado en {MOVIL_PATH}, se continua sin datos moviles.")

    df = actualizar_dataset(subtel_dict, movil_nacional)

    print("\nResumen por region:")
    print(df.groupby('num_region')['nombre_comuna'].count().to_string())


if __name__ == "__main__":
    main()
