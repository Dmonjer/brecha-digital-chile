"""
Script para buscar y descargar datos de SUBTEL probando multiples URLs.
Ejecutar con: py -3 src/buscar_subtel.py
"""
import sys
import io
import requests
import os
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
SUBTEL_DIR = BASE_DIR / "data" / "raw" / "subtel"
SUBTEL_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "es-CL,es;q=0.9"
}

PAGINAS_SUBTEL = [
    "https://www.subtel.gob.cl/estudios-y-estadisticas/internet/",
    "https://www.subtel.gob.cl/estudios-y-estadisticas/telefonia/",
]


def buscar_links_descarga(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.(xlsx|xls|csv|zip)$", href, re.IGNORECASE):
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
        print(f"  Error al acceder {url}: {e}")
        return []


def descargar(url, destino, nombre="archivo"):
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        size = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
        if size < 1000:
            os.remove(destino)
            return False
        print(f"  OK: {nombre} ({size/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Error descargando {nombre}: {e}")
        if os.path.exists(destino):
            os.remove(destino)
        return False


def main():
    print("=" * 60)
    print("  BUSQUEDA DE DATOS SUBTEL")
    print("=" * 60)

    todos_links = []

    for url in PAGINAS_SUBTEL:
        print(f"\nExplorando: {url}")
        links = buscar_links_descarga(url)
        print(f"  -> {len(links)} archivos encontrados")
        for l in links:
            print(f"     {l['url']}")
        todos_links.extend(links)

    # Deduplicar
    seen = set()
    links_unicos = []
    for l in todos_links:
        if l["url"] not in seen:
            seen.add(l["url"])
            links_unicos.append(l)

    print(f"\nTotal links unicos: {len(links_unicos)}")

    # Descargar relevantes
    keywords = ["internet", "fija", "fijo", "movil", "moviles", "comunas",
                "accesos", "cobertura", "telefonia", "celular", "series"]

    print("\nDescargando archivos relevantes...")
    for link in links_unicos:
        texto_lower = unicodedata.normalize('NFD', link["texto"].lower()).encode('ascii','ignore').decode()
        url_lower = link["url"].lower()
        if any(kw in texto_lower or kw in url_lower for kw in keywords):
            nombre = link["url"].split("/")[-1]
            destino = SUBTEL_DIR / nombre
            if destino.exists():
                print(f"  Ya existe: {nombre}")
                continue
            print(f"  Descargando: {nombre}")
            descargar(link["url"], destino, nombre)

    archivos = list(SUBTEL_DIR.iterdir())
    print(f"\nArchivos en data/raw/subtel/ ({len(archivos)}):")
    for f in archivos:
        print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
