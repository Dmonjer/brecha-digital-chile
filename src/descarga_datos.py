"""
Script de descarga automática de datos oficiales para el proyecto
"Análisis de la Brecha Digital y Calidad de Red en Chile"

Fuentes:
  - SUBTEL: Subsecretaría de Telecomunicaciones
  - INE:    Instituto Nacional de Estadísticas
  - CASEN:  Encuesta de Caracterización Socioeconómica

Uso:
    python src/descarga_datos.py
"""

import os
import sys
import requests
import time
from pathlib import Path

# ============================================================
# RUTAS
# ============================================================
BASE_DIR    = Path(__file__).parent.parent
DATA_SUBTEL = BASE_DIR / "data" / "raw" / "subtel"
DATA_INE    = BASE_DIR / "data" / "raw" / "ine"
DATA_CASEN  = BASE_DIR / "data" / "raw" / "casen"

for d in [DATA_SUBTEL, DATA_INE, DATA_CASEN]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# URLS DE DESCARGA – DATOS OFICIALES
# ============================================================
# SUBTEL – Datos de accesos a internet
# Fuente: https://datos.gob.cl / https://www.subtel.gob.cl
ARCHIVOS_SUBTEL = [
    {
        "url": "https://datos.gob.cl/dataset/d6e1fa26-2e05-4b28-9440-4b014e6a8bd0/resource/2c3b3ed5-c898-4d70-8b25-0186d6524e8f/download/accesos_internet_fijo_por_comuna.xlsx",
        "nombre": "accesos_internet_fijo_por_comuna.xlsx",
        "descripcion": "Accesos a internet fijo por comuna - SUBTEL"
    },
    {
        "url": "https://datos.gob.cl/dataset/d6e1fa26-2e05-4b28-9440-4b014e6a8bd0/resource/cobertura_movil_por_comuna.xlsx",
        "nombre": "cobertura_movil_por_comuna.xlsx",
        "descripcion": "Cobertura móvil 4G/5G por comuna - SUBTEL"
    },
]

# INE – Datos del Censo 2017
ARCHIVOS_INE = [
    {
        "url": "https://www.ine.gob.cl/docs/default-source/censo-de-poblacion-y-vivienda/bbdd/2017/csv/personas.csv",
        "nombre": "censo_2017_personas.csv",
        "descripcion": "Microdatos Censo 2017 - Personas"
    },
]

# CASEN 2022 – Ministerio de Desarrollo Social
ARCHIVOS_CASEN = [
    {
        "url": "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2022/Casen_2022_STATA.zip",
        "nombre": "casen_2022.zip",
        "descripcion": "Microdatos CASEN 2022"
    },
]


# ============================================================
# FUNCIÓN DE DESCARGA
# ============================================================
def descargar_archivo(url, destino, descripcion="", reintentos=3, timeout=60):
    """Descarga un archivo desde una URL con reintentos.
    
    Args:
        url: URL del archivo a descargar
        destino: Path de destino
        descripcion: Descripción del archivo
        reintentos: Número máximo de intentos
        timeout: Tiempo máximo de espera en segundos
    
    Returns:
        bool: True si la descarga fue exitosa
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for intento in range(1, reintentos + 1):
        try:
            print(f"  📥 Descargando: {descripcion or url}")
            print(f"     Intento {intento}/{reintentos}...")
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            total = int(response.headers.get("content-length", 0))
            descargado = 0
            
            with open(destino, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        descargado += len(chunk)
            
            size_mb = os.path.getsize(destino) / (1024 * 1024)
            print(f"  ✅ Descargado: {destino.name} ({size_mb:.1f} MB)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Error en intento {intento}: {e}")
            if intento < reintentos:
                print(f"     Reintentando en 5 segundos...")
                time.sleep(5)
            else:
                print(f"  ❌ Falló la descarga de: {descripcion}")
                if os.path.exists(destino):
                    os.remove(destino)
                return False


# ============================================================
# DESCARGA PRINCIPAL
# ============================================================
def main():
    print("=" * 60)
    print("  DESCARGA DE DATOS – BRECHA DIGITAL CHILE")
    print("=" * 60)
    
    exitosas = 0
    fallidas = 0
    
    # SUBTEL
    print("\n📡 Descargando datos SUBTEL...")
    for archivo in ARCHIVOS_SUBTEL:
        destino = DATA_SUBTEL / archivo["nombre"]
        if destino.exists():
            print(f"  ⏭️  Ya existe: {archivo['nombre']}")
            exitosas += 1
        else:
            ok = descargar_archivo(archivo["url"], destino, archivo["descripcion"])
            exitosas += 1 if ok else 0
            fallidas += 0 if ok else 1
    
    # INE
    print("\n📊 Descargando datos INE...")
    for archivo in ARCHIVOS_INE:
        destino = DATA_INE / archivo["nombre"]
        if destino.exists():
            print(f"  ⏭️  Ya existe: {archivo['nombre']}")
            exitosas += 1
        else:
            ok = descargar_archivo(archivo["url"], destino, archivo["descripcion"])
            exitosas += 1 if ok else 0
            fallidas += 0 if ok else 1
    
    # CASEN
    print("\n👥 Descargando datos CASEN...")
    for archivo in ARCHIVOS_CASEN:
        destino = DATA_CASEN / archivo["nombre"]
        if destino.exists():
            print(f"  ⏭️  Ya existe: {archivo['nombre']}")
            exitosas += 1
        else:
            ok = descargar_archivo(archivo["url"], destino, archivo["descripcion"])
            exitosas += 1 if ok else 0
            fallidas += 0 if ok else 1
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"  RESUMEN: {exitosas} exitosas | {fallidas} fallidas")
    print("=" * 60)
    
    if fallidas > 0:
        print("\n⚠️  INSTRUCCIONES MANUALES para descargas fallidas:")
        print("  Ver README.md sección 'Descarga Manual de Datos'")
    
    return exitosas, fallidas


if __name__ == "__main__":
    main()
