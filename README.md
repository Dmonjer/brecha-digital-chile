# 🌐 Análisis de la Brecha Digital y Calidad de Red en Chile
### Proyecto Semestral – Data Science 2026-02 | UDP

**Integrantes:** Diego Lara · Cristóbal Castillo  
**Profesor:** Jorge Rozas | **Ramo:** Data Science

---

## 📌 Descripción del Proyecto

Este proyecto analiza la **brecha digital comunal en Chile** utilizando datos oficiales de SUBTEL, INE y CASEN 2022. Aplicamos técnicas de clustering no supervisado (K-Means) y clasificación supervisada (Regresión Logística + Random Forest) para segmentar las 346 comunas (100% oficial) del país según su nivel de vulnerabilidad digital.

**Pregunta de investigación:**  
> *¿Qué factores socioeconómicos y demográficos explican la brecha digital entre comunas de Chile, y es posible predecir el perfil de vulnerabilidad digital de una comuna?*

---

## 📊 Resultados Clave – Etapa II

| Indicador | Valor |
|-----------|-------|
| Comunas analizadas | **346** |
| Variables del dataset | **13** |
| Clusters K-Means (k óptimo) | **3** |
| Silhouette Score | **0.338** |
| Accuracy Random Forest | **97.1%** |
| F1-Score Random Forest | **0.971** |

### Perfiles de Vulnerabilidad Digital

| Perfil | N° Comunas | % | Descripción |
|--------|-----------|---|-------------|
| 🔴 Alta Vulnerabilidad | 151 | 43.6% | Rural, baja conectividad (~78 conex/1.000 hab), mayor pobreza |
| 🟠 Vulnerabilidad Media | 148 | 42.8% | Ciudades intermedias, conectividad moderada (~180 conex/1.000 hab) |
| 🟢 Baja Vulnerabilidad | 47 | 13.6% | Grandes urbes, alta conectividad (~258 conex/1.000 hab), menor pobreza |

---

## 📁 Estructura del Repositorio

```
proyecto-brecha-digital/
│
├── notebooks/
│   └── etapa2_brecha_digital_chile.ipynb   ← Notebook principal (ejecutar aquí)
│
├── src/
│   ├── descarga_datos.py     ← Descarga automática de fuentes oficiales
│   ├── preparar_datos.py     ← Limpieza e integración de datos
│   ├── modelado.py           ← Pipeline completo de modelos
│   └── buscar_subtel.py      ← Scraper de URLs SUBTEL
│
├── data/
│   ├── raw/                  ← (no incluido en git – descargar con instrucciones)
│   └── processed/
│       ├── dataset_comunas_chile.csv           ← Dataset integrado
│       └── dataset_comunas_con_clusters.csv    ← Dataset con etiquetas K-Means
│
├── outputs/
│   ├── figures/              ← 9 figuras generadas automáticamente
│   │   ├── 01_distribuciones.png
│   │   ├── 02_heatmap_correlacion.png
│   │   ├── 03_scatter_pobreza_conectividad.png
│   │   ├── 04_ranking_conectividad.png
│   │   ├── 05_elbow_silhouette.png
│   │   ├── 06_clusters_pca_heatmap.png
│   │   ├── 07_confusion_matrices.png
│   │   ├── 08_feature_importance.png
│   │   └── 09_comparativa_modelos.png
│   ├── models/               ← Modelos serializados (.pkl)
│   └── presentacion_brecha_digital.html  ← Presentación interactiva
│
└── README.md
```

---

## 🚀 Cómo Reproducir

### 1. Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl beautifulsoup4 requests scipy
```

### 2. Descargar datos oficiales

Los datos RAW no están en el repositorio por su tamaño. Descargar manualmente:

| Archivo | Fuente | URL |
|---------|--------|-----|
| `series_internet_fija_subtel.xlsx` | SUBTEL | [subtel.gob.cl/estadísticas/internet](https://www.subtel.gob.cl/estudios-y-estadisticas/internet/) |
| `ine_proyecciones_poblacion_comunas.xlsx` | INE | [ine.gob.cl/proyecciones](https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion) |
| `casen_2022_pobreza_ingresos_comunas.xlsx` | CASEN/MDSF | [observatorio.ministeriodesarrollosocial.gob.cl](https://observatorio.ministeriodesarrollosocial.gob.cl/pobreza-comunal-2022) |
| `casen_2022_pobreza_multidimensional_comunas.xlsx` | CASEN/MDSF | [observatorio.ministeriodesarrollosocial.gob.cl](https://observatorio.ministeriodesarrollosocial.gob.cl/pobreza-comunal-2022) |

Colocar en `data/raw/subtel/`, `data/raw/ine/` y `data/raw/casen/` respectivamente.

> **O usar el script automático:**  
> ```bash
> python src/descarga_datos.py
> ```

### 3. Ejecutar el pipeline

```bash
# Opción A: Ejecutar el notebook completo en Jupyter
jupyter notebook notebooks/etapa2_brecha_digital_chile.ipynb

# Opción B: Scripts individuales
python src/preparar_datos.py   # Genera data/processed/*.csv
python src/modelado.py         # Genera outputs/figures/ y outputs/models/
```

---

## 📚 Fuentes de Datos

| Organismo | Descripción | Última actualización |
|-----------|-------------|---------------------|
| **SUBTEL** | Conexiones fijas residenciales por comuna (hoja `7.11.1.CO_FIJAS_RES_COMUNA`) | Jun 2026 |
| **INE** | Proyecciones de población comunal (base Censo 2017) | 2024 |
| **CASEN 2022** | Estimaciones SAE de pobreza por ingresos y multidimensional por comuna | 2022 |

---

## 🔧 Tecnologías Utilizadas

- **Python 3.10+**
- `pandas`, `numpy` — Manipulación de datos
- `matplotlib`, `seaborn` — Visualización
- `scikit-learn` — K-Means, Regresión Logística, Random Forest, PCA
- `joblib` — Serialización de modelos
- `scipy` — Tests estadísticos (Mann-Whitney)
- `openpyxl` — Lectura de archivos Excel

---

## 📝 Metodología

El proyecto sigue el framework **CRISP-DM**:

1. **Comprensión del negocio** → Brecha digital como problema de inequidad
2. **Comprensión de datos** → EDA con 346 comunas (100% oficial) × 13 variables
3. **Preparación** → Limpieza, merge por CUT, imputación mediana, winsorización p1-p99
4. **Modelado** → K-Means (k=3) + Reg. Logística + Random Forest
5. **Evaluación** → Silhouette=0.437 | Accuracy RF=98.7%
6. **Despliegue** → *(Etapa III: Dashboard + SHAP + XGBoost)*

---

*Universidad Diego Portales · Facultad de Economía y Empresa · 2026*
