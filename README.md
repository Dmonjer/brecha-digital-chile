# 🌐 Análisis de la Brecha Digital y Calidad de Red en Chile (Etapa 3)
### Proyecto Semestral – Data Science 2026-02 | Universidad Diego Portales (UDP)

**Integrantes:** Diego Lara · Cristóbal Castillo  
**Profesor:** Jorge Rozas | **Ramo:** Data Science  
**Área de Estudio:** Macrozona Centro-Norte (Regiones de Coquimbo, Valparaíso y Metropolitana)

---

## 📌 Descripción del Proyecto

Este proyecto investiga la **brecha digital comunal en Chile** a nivel multidimensional, integrando **infraestructura de red fija y red móvil** con datos oficiales de **SUBTEL, INE y CASEN**. 

En la **Etapa 3**, el estudio se enfoca en las 105 comunas de la **Macrozona Centro-Norte** (Región de Coquimbo [4], Región de Valparaíso [5] y Región Metropolitana [13]), analizando la complementariedad y disparidad entre la penetración de internet fijo residencial y la infraestructura celular móvil (estaciones base). Se aplican técnicas de agrupamiento no supervisado (**K-Means**) y clasificación supervisada (**Random Forest**) bajo validación cruzada estratificada imparcial.

**Pregunta de investigación:**  
> *¿De qué manera la combinación de infraestructura fija y móvil, junto a factores demográficos y socioeconómicos, define los perfiles de conectividad comunal en la Macrozona Centro-Norte de Chile, y con qué precisión podemos predecir dicho nivel de conectividad?*

---

## 📊 Resultados Clave – Etapa 3

| Indicador | Valor |
|-----------|-------|
| Comunas analizadas | **105** (100% Macrozona Centro-Norte) |
| Antenas móviles analizadas | **13.823** estaciones base (catastro SUBTEL) |
| Clusters K-Means ($k$ óptimo) | **3** (`Alta conectividad`, `Conectividad media`, `Baja conectividad`) |
| Silhouette Score | **0.3473** |
| Random Forest CV F1-Macro (5-fold) | **91.22% ± 7.34%** |
| Random Forest CV Accuracy (5-fold) | **95.29% ± 4.40%** |
| Accuracy en Test (n=21) | **100.0%** (F1-Macro = 1.000) |

### Perfiles de Conectividad Comunal

| Perfil | N° Comunas | % | Conexiones Fijas / 1.000 hab. | Antenas Móviles / 1.000 hab. | Pobreza Ingresos | Ruralidad Promedio |
|--------|-----------|---|-------------------------------|------------------------------|------------------|--------------------|
| 🟢 **Alta conectividad** | 12 | 11.4% | **330.7** | **3.94** | **3.9%** | 30.8% |
| 🔵 **Conectividad media** | 52 | 49.5% | **223.0** | **0.90** | **5.4%** | 10.4% |
| 🔴 **Baja conectividad** | 41 | 39.1% | **122.8** | **1.74** | **8.3%** | 42.7% |

---

## 📁 Estructura del Repositorio

```
proyecto-brecha-digital/
│
├── notebooks/
│   └── etapa2_brecha_digital_chile.ipynb   ← Notebook principal reproducible
│
├── src/
│   ├── preparar_datos.py     ← Extracción SUBTEL (fija + antenas móviles), INE, CASEN y merge
│   ├── modelado.py           ← Pipeline completo: EDA, K-Means (k=3), PCA y Random Forest
│   └── buscar_subtel.py      ← Scraper y utilidades de fuentes SUBTEL
│
├── data/
│   ├── raw/                  ← Fuentes oficiales brutas (SUBTEL, INE, CASEN)
│   └── processed/
│       ├── dataset_comunas_chile.csv           ← Dataset integrado comunal (105 comunas × 17 vars)
│       └── dataset_comunas_con_clusters.csv    ← Dataset con clusters y distancias a centroides
│
├── outputs/
│   ├── figures/              ← 9 figuras generadas automáticamente en alta resolución
│   │   ├── 01_distribuciones.png
│   │   ├── 02_heatmap_correlacion.png
│   │   ├── 03_scatter_pobreza_conectividad.png
│   │   ├── 04_ranking_conectividad.png
│   │   ├── 05_elbow_silhouette.png
│   │   ├── 06_clusters_pca_heatmap.png
│   │   ├── 07_confusion_matrix_rf.png
│   │   ├── 08_feature_importance.png
│   │   └── 09_metricas_rf.png
│   └── models/               ← Modelos serializados (.pkl)
│       ├── kmeans_model.pkl
│       ├── scaler_clustering.pkl
│       └── random_forest.pkl
│
├── INFORME_ETAPA_3.md        ← Informe ejecutivo y técnico detallado de la Etapa 3
└── README.md
```

---

## 🚀 Cómo Reproducir el Proyecto

### 1. Entorno y Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
```

### 2. Ejecución del Pipeline Completo

```bash
# Paso 1: Preparación de datos (SUBTEL fija + antenas móviles + INE + CASEN)
python src/preparar_datos.py

# Paso 2: Modelado estadístico y Machine Learning (EDA, Clustering, RF)
python src/modelado.py
```

---

## 📚 Fuentes de Información Oficial

| Organismo | Recurso Oficial | Uso en el Modelo |
|---|---|---|
| **SUBTEL** | *Series de Conexiones a Internet Fijo por Comuna* | Conexiones residenciales fijas per cápita |
| **SUBTEL** | *Catastro Nacional de Estaciones Base y Antenas* | Infraestructura y densidad celular móvil comunal |
| **INE** | *Proyecciones de Población Comunal (Censo)* | Ponderador per cápita y base demográfica |
| **CASEN / MDSF**| *Estimaciones SAE Comunal 2022* | Tasa de pobreza por ingresos y multidimensional |

---

## ⚖️ Principales Mejoras Metodológicas (Etapa 3)

1. **Datos Móviles Reales por Comuna:** Se descartaron variables agregadas nacionales sin varianza y se construyó un indicador objetivo de infraestructura celular mediante el catastro de 13.823 antenas de SUBTEL.
2. **Evaluación Imparcial:** Uso estricto de **F1-Macro** en validación cruzada estratificada 5-fold, eliminando el sesgo de la clase mayoritaria.
3. **Nomenclatura Técnica No Estigmatizante:** Sustitución de "vulnerabilidad" por niveles descriptivos de conectividad (`Alta`, `Media`, `Baja`).
4. **Muestra y Representatividad:** El cluster de baja conectividad creció de 18 a 41 comunas, duplicando la muestra de test ($n=8$) y garantizando validez estadística.

---

*Universidad Diego Portales · Facultad de Ingeniería y Ciencias / Escuela de Informática y Telecomunicaciones · 2026*
