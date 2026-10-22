"""
Script para buscar y descargar datos de SUBTEL probando múltiples URLs alternativas.
Ejecutar con: python src/buscar_subtel.py
"""
import requests
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
SUBTEL_DIR = BASE_DIR / "data" / "raw" / "subtel"
SUBTEL_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-CL,es;q=0.9"
}

# URLs de páginas a explorar
PAGINAS_SUBTEL = [
    "https://www.subtel.gob.cl/estudios-y-estadisticas/internet/",
    "https://www.subtel.gob.cl/estadisticas/internet/",
    "https://datos.gob.cl/organization/subsecretaria-de-telecomunicaciones",
    "https://datos.gob.cl/dataset?q=subtel+internet&sort=score+desc%2C+metadata_modified+desc",
    "https://datos.gob.cl/dataset?q=accesos+internet+comunas"
]

def buscar_links_descarga(url):
    """Busca links de descarga de archivos Excel/CSV en una página."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.(xlsx|xls|csv|zip)$", href, re.IGNORECASE):
                # Hacer URL absoluta si es relativa
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("//"):
                    full_url = "https:" + href
                elif href.startswith("/"):
                    from urllib.parse import urlparse
                    base = urlparse(url)
                    full_url = f"{base.scheme}://{base.netloc}{href}"
                else:
                    full_url = url.rstrip("/") + "/" + href
                
                text = a.get_text(strip=True) or href.split("/")[-1]
                links.append({"url": full_url, "texto": text})
        
        return links
    except Exception as e:
        print(f"  ⚠️  Error al acceder {url}: {e}")
        return []

def descargar(url, destino, nombre="archivo"):
    """Intenta descargar un archivo desde una URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        
        # Verificar que es un archivo real (no HTML)
        ct = r.headers.get("content-type", "")
        if "html" in ct and r.headers.get("content-length", "999999") == "0":
            return False
        
        size = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
        
        if size < 1000:  # Archivo vacío o error HTML disfrazado
            os.remove(destino)
            return False
        
        print(f"    ✅ {nombre} ({size/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        if os.path.exists(destino):
            os.remove(destino)
        return False

def main():
    print("=" * 60)
    print("  BÚSQUEDA AUTOMÁTICA DE DATOS SUBTEL")
    print("=" * 60)
    
    todos_links = []
    
    for url in PAGINAS_SUBTEL:
        print(f"\n🔍 Explorando: {url}")
        links = buscar_links_descarga(url)
        print(f"   → {len(links)} archivos descargables encontrados")
        for l in links:
            print(f"     • {l['texto']}: {l['url']}")
        todos_links.extend(links)
    
    # Deduplicar
    seen = set()
    links_unicos = []
    for l in todos_links:
        if l["url"] not in seen:
            seen.add(l["url"])
            links_unicos.append(l)
    
    print(f"\n📋 Total de links únicos encontrados: {len(links_unicos)}")
    
    # Intentar descargar los que parezcan relevantes
    keywords = ["internet", "fija", "fijo", "movil", "móvil", "comunas", "accesos", "cobertura"]
    
    print("\n📥 Descargando archivos relevantes...")
    for link in links_unicos:
        texto_lower = link["texto"].lower()
        url_lower = link["url"].lower()
        
        if any(kw in texto_lower or kw in url_lower for kw in keywords):
            nombre = link["url"].split("/")[-1]
            destino = SUBTEL_DIR / nombre
            
            if destino.exists():
                print(f"  ⏭️  Ya existe: {nombre}")
                continue
            
            print(f"  📥 {link['texto']}: {nombre}")
            descargar(link["url"], destino, nombre)
    
    # Listar resultados finales
    archivos = list(SUBTEL_DIR.iterdir())
    print(f"\n📁 Archivos en data/raw/subtel/ ({len(archivos)}):")
    for f in archivos:
        print(f"   • {f.name} ({f.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
