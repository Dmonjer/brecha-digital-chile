"""
Script de preparación e integración de datos para el proyecto
"Análisis de la Brecha Digital y Calidad de Red en Chile"

Integra: SUBTEL + INE + CASEN → dataset_comunas_chile.csv

Ejecutar con: python src/preparar_datos.py
"""
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
SUBTEL_PATH = BASE_DIR / "data" / "raw" / "subtel" / "series_internet_fija_subtel.xlsx"
INE_PATH    = BASE_DIR / "data" / "raw" / "ine" / "ine_proyecciones_poblacion_comunas.xlsx"
INE_UR_PATH = BASE_DIR / "data" / "raw" / "ine" / "ine_proyecciones_urbano_rural.xlsx"
CASEN_PATH  = BASE_DIR / "data" / "raw" / "casen" / "casen_2022_pobreza_ingresos_comunas.xlsx"
CASEN_MD    = BASE_DIR / "data" / "raw" / "casen" / "casen_2022_pobreza_multidimensional_comunas.xlsx"
OUT_PATH    = BASE_DIR / "data" / "processed" / "dataset_comunas_chile.csv"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGAR Y LIMPIAR DATOS SUBTEL
# ============================================================
def cargar_subtel(path):
    """
    Extrae conexiones fijas residenciales por comuna del Excel de SUBTEL.
    Toma el último mes disponible como la cifra más reciente.
    """
    print("📡 Cargando datos SUBTEL (hoja: 7.11.1.CO_FIJAS_RES_COMUNA)...")
    
    try:
        # Leer hoja de conexiones residenciales por comuna
        df_raw = pd.read_excel(path, sheet_name="7.11.1.CO_FIJAS_RES_COMUNA", header=None)
    except Exception:
        # Fallback a hoja total
        df_raw = pd.read_excel(path, sheet_name="7.11.CO_FIJAS_COMUNA", header=None)
    
    # Buscar fila de encabezados (fila con "Región" y "Comuna")
    header_row = None
    for i, row in df_raw.iterrows():
        row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
        if "Región" in row_str or "Region" in row_str or "regi" in row_str.lower():
            if "Comuna" in row_str or "omuna" in row_str:
                header_row = i
                break
    
    if header_row is None:
        # Buscar fila donde aparezcan datos de comunas
        for i, row in df_raw.iterrows():
            row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
            if "Alto Hospicio" in row_str or "Arica" in row_str:
                header_row = i - 1
                break
    
    print(f"   Fila de encabezados encontrada: {header_row}")
    
    # Re-leer con header correcto
    df = pd.read_excel(path, sheet_name="7.11.1.CO_FIJAS_RES_COMUNA", 
                       header=None, skiprows=header_row+1)
    
    # Identificar columnas de región y comuna (primeras no-NaN)
    # Buscar datos reales: primera columna con textos de comunas
    data_rows = []
    for i, row in df.iterrows():
        vals = [str(v) for v in row.values if pd.notna(v) and str(v) not in ['nan', '']]
        if len(vals) >= 3:
            # Verificar que el tercero sea numérico (conexiones)
            try:
                float(vals[2].replace(',', '.'))
                data_rows.append(i)
            except:
                pass
    
    if not data_rows:
        print("   ⚠️ No se encontraron filas de datos. Usando método alternativo...")
        # Método alternativo: leer directamente
        df2 = df_raw.dropna(how='all').reset_index(drop=True)
        return _extraer_subtel_alternativo(df2)
    
    # Extraer datos de comunas
    print(f"   Filas con datos encontradas: {len(data_rows)}")
    
    comunas_data = []
    for i in data_rows:
        row = df.iloc[i]
        vals = [v for v in row.values if pd.notna(v) and str(v) not in ['nan', '']]
        if len(vals) >= 3:
            # Primeros dos son región y comuna, el resto son series temporales
            try:
                region = str(vals[0]).strip()
                comuna = str(vals[1]).strip()
                # Tomar el último valor numérico (mes más reciente)
                nums = []
                for v in vals[2:]:
                    try:
                        nums.append(float(str(v).replace(',', '.')))
                    except:
                        pass
                if nums and (region.lstrip('-').lstrip().isdigit() or any(c.isdigit() for c in region)):
                    conexiones_recientes = nums[-1] if nums else np.nan
                    comunas_data.append({
                        'num_region': region,
                        'nombre_comuna': comuna,
                        'conexiones_fijas_residenciales': conexiones_recientes,
                        'max_conexiones_historico': max(nums) if nums else np.nan
                    })
            except Exception as e:
                pass
    
    if comunas_data:
        df_subtel = pd.DataFrame(comunas_data)
        df_subtel = df_subtel[df_subtel['nombre_comuna'].str.len() > 2]
        print(f"   ✅ {len(df_subtel)} comunas extraídas de SUBTEL")
        return df_subtel
    else:
        return _extraer_subtel_alternativo(df_raw)


def _extraer_subtel_alternativo(df_raw):
    """Método alternativo: busca directamente la columna de comunas conocidas."""
    print("   Usando método alternativo de extracción SUBTEL...")
    
    df_raw_str = df_raw.astype(str)
    
    comunas_data = []
    for i, row in df_raw_str.iterrows():
        row_vals = list(row.values)
        # Buscar filas donde aparezcan nombres de comunas chilenas conocidas
        row_str = " ".join([v for v in row_vals if v not in ['nan', 'NaN', '']])
        if any(c in row_str for c in ['Alto Hospicio', 'Arica', 'Iquique', 'Santiago', 'Providencia']):
            # Encontrar posición del nombre de la comuna
            for j, val in enumerate(row_vals):
                if any(c in str(val) for c in ['Alto Hospicio', 'Arica', 'Iquique', 'Santiago']):
                    nums = []
                    for v in row_vals[j+1:]:
                        try:
                            n = float(str(v).replace(',', '.'))
                            if n > 0:
                                nums.append(n)
                        except:
                            pass
                    if nums:
                        # Obtener región (columna anterior)
                        region = str(row_vals[j-1]) if j > 0 else 'N/A'
                        comunas_data.append({
                            'num_region': region.strip(),
                            'nombre_comuna': str(val).strip(),
                            'conexiones_fijas_residenciales': nums[-1],
                            'max_conexiones_historico': max(nums)
                        })
                    break
    
    df_subtel = pd.DataFrame(comunas_data) if comunas_data else pd.DataFrame()
    print(f"   ✅ {len(df_subtel)} comunas extraídas (método alternativo)")
    return df_subtel


# ============================================================
# 2. CARGAR DATOS INE
# ============================================================
def cargar_ine(path_comunas, path_urbano_rural):
    """Extrae proyecciones de población 2024 por comuna del INE."""
    print("\n📊 Cargando datos INE (proyecciones de población)...")
    
    # Leer el archivo de proyecciones por comunas
    xl = pd.ExcelFile(path_comunas)
    print(f"   Hojas disponibles: {xl.sheet_names[:8]}")
    
    # Buscar hoja con datos de total población
    df_ine = None
    for hoja in xl.sheet_names:
        if 'total' in hoja.lower() or 'pob' in hoja.lower() or 'ambos' in hoja.lower():
            try:
                df_temp = pd.read_excel(path_comunas, sheet_name=hoja, header=None)
                # Verificar si tiene columnas con año 2024
                df_str = df_temp.astype(str)
                if '2024' in df_str.to_string() or '2023' in df_str.to_string():
                    df_ine = df_temp
                    print(f"   Usando hoja: {hoja}")
                    break
            except:
                continue
    
    if df_ine is None:
        # Usar primera hoja con datos
        df_ine = pd.read_excel(path_comunas, sheet_name=xl.sheet_names[0], header=None)
    
    print(f"   Shape: {df_ine.shape}")
    print(f"   Primeras filas:")
    print(df_ine.head(6).to_string())
    
    # Identificar estructura del archivo
    # Típicamente: Región | Nombre Región | CUT | Nombre Comuna | Año 2002 | ... | Año 2035
    comunas_data = []
    
    # Buscar fila de encabezados
    header_idx = None
    for i, row in df_ine.iterrows():
        row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
        if ('2024' in row_str or '2023' in row_str) and ('2022' in row_str or '2025' in row_str):
            header_idx = i
            break
    
    if header_idx is not None:
        headers = list(df_ine.iloc[header_idx].values)
        print(f"   Encabezados encontrados en fila {header_idx}: {headers[:10]}...")
        
        # Buscar índice de año 2024
        col_2024 = None
        for j, h in enumerate(headers):
            if str(h).strip() == '2024' or str(h).strip() == '2024.0':
                col_2024 = j
                break
        
        if col_2024 is None:
            # Buscar 2023
            for j, h in enumerate(headers):
                if str(h).strip() in ['2023', '2023.0']:
                    col_2024 = j
                    break
        
        print(f"   Columna de año objetivo: {col_2024} (valor: {headers[col_2024] if col_2024 else 'N/A'})")
        
        # Extraer datos
        for i, row in df_ine.iloc[header_idx+1:].iterrows():
            vals = list(row.values)
            # Buscar código de región y nombre de comuna
            try:
                # Típicamente: [num_region, nombre_region, CUT, nombre_comuna, pob_2002, ..., pob_2024, ...]
                non_nan = [v for v in vals if pd.notna(v)]
                if len(non_nan) >= 5:
                    num_reg = str(non_nan[0]).strip()
                    # Verificar que sea un número de región
                    if num_reg.replace('.0','').isdigit():
                        try:
                            cut = int(float(str(non_nan[2]).replace(',','')))
                            nombre_com = str(non_nan[3]).strip()
                            pob = float(str(vals[col_2024]).replace(',','')) if col_2024 else np.nan
                            
                            if cut > 1000 and cut < 99999 and len(nombre_com) > 2:
                                comunas_data.append({
                                    'cut_comuna': cut,
                                    'nombre_comuna': nombre_com,
                                    'num_region': num_reg.replace('.0',''),
                                    'poblacion_2024': pob
                                })
                        except:
                            pass
            except:
                pass
    
    if comunas_data:
        df_out = pd.DataFrame(comunas_data).drop_duplicates('cut_comuna')
        print(f"   ✅ {len(df_out)} comunas extraídas del INE")
        return df_out
    else:
        print("   ⚠️ No se pudo extraer datos estructurados. Retornando datos crudos para inspección manual.")
        return df_ine


# ============================================================
# 3. CARGAR DATOS CASEN
# ============================================================
def cargar_casen(path_ingresos, path_multidim):
    """Extrae tasas de pobreza por comuna de CASEN 2022."""
    print("\n👥 Cargando datos CASEN 2022...")
    
    dfs = []
    
    for path, tipo in [(path_ingresos, 'pobreza_ingresos'), (path_multidim, 'pobreza_multidimensional')]:
        if not os.path.exists(path):
            print(f"   ⚠️ Archivo no encontrado: {path}")
            continue
        
        try:
            xl = pd.ExcelFile(path)
            print(f"   Hojas en {os.path.basename(path)}: {xl.sheet_names}")
            
            # Leer primera hoja de datos
            df_raw = pd.read_excel(path, header=None)
            print(f"   Shape: {df_raw.shape}")
            print(f"   Primeras 8 filas:")
            print(df_raw.head(8).to_string())
            
            # Buscar encabezados
            header_idx = None
            for i, row in df_raw.iterrows():
                row_str = " ".join([str(v) for v in row.values if pd.notna(v)]).lower()
                if 'comuna' in row_str and ('pobreza' in row_str or 'tasa' in row_str or 'código' in row_str or 'codigo' in row_str):
                    header_idx = i
                    break
            
            if header_idx is not None:
                df = pd.read_excel(path, header=header_idx)
                df.columns = [str(c).strip() for c in df.columns]
                print(f"   Columnas ({tipo}): {list(df.columns)}")
                df['_tipo'] = tipo
                dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error cargando {os.path.basename(path)}: {e}")
    
    if dfs:
        return dfs
    else:
        return None


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
def main():
    print("=" * 60)
    print("  PREPARACIÓN DE DATOS – BRECHA DIGITAL CHILE")
    print("=" * 60)
    
    # 1. Cargar SUBTEL
    df_subtel = cargar_subtel(SUBTEL_PATH)
    if df_subtel is not None and len(df_subtel) > 0:
        print(f"\n✅ SUBTEL: {len(df_subtel)} filas, columnas: {list(df_subtel.columns)}")
        print(df_subtel.head(5).to_string())
    
    # 2. Cargar INE
    df_ine = cargar_ine(INE_PATH, INE_UR_PATH)
    if isinstance(df_ine, pd.DataFrame) and len(df_ine) > 0:
        print(f"\n✅ INE: {len(df_ine)} filas")
    
    # 3. Cargar CASEN
    casen_dfs = cargar_casen(CASEN_PATH, CASEN_MD)
    if casen_dfs:
        print(f"\n✅ CASEN: {len(casen_dfs)} datasets cargados")
    
    print("\n" + "="*60)
    print("Inspección completa. Verificar las columnas disponibles")
    print("para ajustar el merge en el notebook.")
    print("="*60)


if __name__ == "__main__":
    main()
