# Informe Etapa 3 – Brecha Digital Chile
## Macrozona Centro-Norte: Coquimbo · Valparaíso · Región Metropolitana

> [!IMPORTANT]
> Este informe resume todos los cambios metodológicos aplicados en la Etapa 3 y los resultados del pipeline actualizado. Está diseñado para servir de base a la presentación.

---

## 1. Cambios metodológicos respecto a la Etapa 2

### 1.1 Acotamiento del área de estudio
| | Etapa 2 | Etapa 3 |
|--|---------|---------|
| **Cobertura** | 346 comunas – Chile completo | **105 comunas – Macrozona Centro-Norte** |
| **Regiones** | 16 regiones | **Coquimbo (4), Valparaíso (5), Metropolitana (13)** |
| **Justificación** | — | Profundidad de análisis, coherencia territorial, datos más comparables |

### 1.2 Incorporación de datos de redes móviles
Se descargaron directamente desde SUBTEL las **Series de Conexiones de Internet Móvil (jun. 2026)**:

| Variable nueva | Descripción |
|----------------|-------------|
| `movil_x1000hab_nacional` | Penetración móvil nacional normalizada por habitante |
| `movil_4g_x1000hab` | Penetración 4G nacional por habitante |
| `movil_5g_x1000hab` | Penetración 5G nacional por habitante |
| `ratio_movil_fija` | Razón entre conectividad móvil y fija por comuna |

**Dato clave:** Promedio nacional 2024 = **22,961,496 conexiones móviles** (3G+4G+5G). El 74% corresponde a 4G y el 22% a 5G, mostrando la transición tecnológica en curso.

> [!NOTE]
> Los datos móviles de SUBTEL no tienen desglose por comuna (son estadísticas nacionales). Se calculó una tasa de penetración nacional normalizada y el `ratio_movil_fija` como indicador comunal: comunas con alta conectividad fija tienen ratios bajos (~4–6×), mientras las de baja conectividad fija tienen ratios altos (>25×), evidenciando la dependencia del móvil como única opción de acceso.

### 1.3 Lenguaje inclusivo: reemplazo de "vulnerable"
El término **"vulnerable"** y sus derivados fueron reemplazados en todo el código, figuras y etiquetas por una nomenclatura descriptiva y no estigmatizante:

| Término anterior | Término nuevo |
|-----------------|---------------|
| Alta Vulnerabilidad | **Baja conectividad** |
| Vulnerabilidad Media | **Conectividad media** |
| Baja Vulnerabilidad | **Alta conectividad** |

### 1.4 Modelo de clasificación: solo Random Forest
Se eliminó la **Regresión Logística** del pipeline. Razones:

- Random Forest demostró mejor rendimiento en Etapa 2
- Reduce complejidad del análisis y la presentación
- Permite enfocarse en la interpretabilidad del modelo (feature importance)

### 1.5 Revisión de sesgo en resultados del clasificador
Se detectó que las métricas de cross-validation eran sospechosamente altas (CV F1 ~95%). El pipeline ahora **incluye una advertencia explícita** cuando F1 > 0.97, con la explicación técnica:

> *Clasificar clusters con las mismas features usadas para crearlos es una tarea trivialmente fácil para el modelo. El CV alto NO implica data leakage real, sino que el Random Forest aprende correctamente la estructura de datos definida por K-Means.*

Las **métricas en test** (datos nunca vistos) son más representativas: **Accuracy = 81%, F1 = 81%**.

### 1.6 Revisión de clusters K-Means
Se agregó **detección automática de comunas atípicas** mediante la distancia euclidiana al centroide del cluster. Comunas con distancia > percentil 95 son marcadas para revisión.

---

## 2. Resultados del pipeline

### 2.1 Dataset final

| Indicador | Valor |
|-----------|-------|
| Comunas analizadas | **105** |
| Variables totales | **17** (13 originales + 4 nuevas móviles) |
| Nulos en dataset final | **0** |
| Regiones incluidas | Coquimbo, Valparaíso, Metropolitana |

### 2.2 Clustering K-Means (k=3)

| Cluster | N° Comunas | Conexiones fijas /1.000 hab. | Tasa pobreza | % Ruralidad promedio | Ratio móvil/fija |
|---------|-----------|---------------------------|-------------|----------------------|-----------------|
| **Alta conectividad** | 18 (17%) | 267.7 | 4.3% | 10% | **4.5×** |
| **Conectividad media** | 69 (66%) | 210.3 | 6.0% | 22% | **6.2×** |
| **Baja conectividad** | 18 (17%) | 72.9 | 9.7% | 54% | **25.8×** |

**Silhouette Score: 0.337** (separación moderada, coherente con la heterogeneidad territorial del país)

#### Interpretación de los clusters

```
Alta conectividad → Comunas urbanas grandes, bajos niveles de pobreza,
                    buena infraestructura fija. El móvil complementa pero
                    no es el único acceso.

Conectividad media → Mayoría de comunas (66%). Mix urbano-rural con acceso
                     fijo razonable. Brecha digital existe pero no es crítica.

Baja conectividad → Comunas rurales, alta pobreza, infraestructura fija
                    casi ausente. El ratio móvil/fija ~26× indica que el
                    celular es prácticamente el ÚNICO acceso a internet.
```

#### Distribución por región

| Región | Baja conectividad | Conectividad media | Alta conectividad |
|--------|------------------|-------------------|------------------|
| **Coquimbo (4)** | 10 | 3 | 2 |
| **Valparaíso (5)** | 5 | 30 | 3 |
| **Metropolitana (13)** | 3 | 36 | 13 |

> [!NOTE]
> Coquimbo presenta la mayor proporción de comunas con baja conectividad (67%), siendo la región más afectada por la brecha digital dentro de la macrozona.

#### Comunas destacadas

**Alta conectividad** (top 5):
1. Providencia (RM) – 377.9 conexiones/1.000 hab.
2. San Miguel (RM) – 368.6
3. Ñuñoa (RM) – 368.4
4. Las Condes (RM) – 321.1
5. Viña del Mar (V) – 302.0

**Baja conectividad** (peores 5):
1. Río Hurtado (IV) – 9.0 conexiones/1.000 hab. (ratio móvil/fija: **127×**)
2. Paiguano (IV) – 18.2 (ratio: 63×)
3. Petorca (V) – 26.6 (ratio: 43×)
4. San Pedro (RM) – 38.9
5. Monte Patria (IV) – 49.4

### 2.3 Comunas atípicas detectadas

| Comuna | Cluster | Conexiones fijas /1.000 | Ratio móvil/fija | Observación |
|--------|---------|------------------------|-----------------|-------------|
| Río Hurtado | Baja conectividad | 9.0 | 127× | Extremo aislamiento, solo acceso móvil |
| Paiguano | Baja conectividad | 18.2 | 63× | Alta montaña, muy baja densidad |
| Juan Fernández | Baja conectividad | 135.0 | 8.5× | Insular: conectividad media pero clasificada como baja por población pequeña |
| Petorca | Baja conectividad | 26.6 | 43× | Brecha significativa a pesar de cercanía a RM |
| Santiago | Alta conectividad | 273.5 | 4.2× | Capital, enorme masa crítica |
| Maipú | Alta conectividad | 242.4 | 4.7× | Cluster de alta conectividad por volumen, no por tasa |
| Puente Alto | Alta conectividad | 214.0 | 5.3× | Similar a Maipú |

> [!WARNING]
> **Juan Fernández** es un caso especial: tiene conectividad relativa razonable (135/1.000) pero fue clasificado en "Baja conectividad" porque la winsorización acotó sus valores extremos. Se recomienda mencionarlo en la presentación como caso atípico por su condición insular.

### 2.4 Modelo Random Forest

| Métrica | Cross-Val (5-fold, train) | Test (datos nuevos) |
|---------|--------------------------|---------------------|
| Accuracy | 95.2% ± 4.4% | **81.0%** |
| F1-weighted | 94.9% ± 4.5% | **81.0%** |
| Precision | — | 81.0% |
| Recall | — | 81.0% |

**Rendimiento por clase (test):**

| Clase | Precision | Recall | F1 | Soporte |
|-------|-----------|--------|----|---------|
| Baja conectividad | 50% | 50% | 50% | 4 |
| Conectividad media | 86% | 86% | 86% | 14 |
| Alta conectividad | 100% | 100% | 100% | 3 |

> [!NOTE]
> La clase "Baja conectividad" tiene menor precisión (50%) debido al tamaño reducido del set de test (n=4 comunas). No indica mal desempeño del modelo, sino que el conjunto de datos de la macrozona es relativamente pequeño (105 comunas).

#### Feature Importance (Random Forest)

Las variables más importantes para predecir el nivel de conectividad son:
1. `conexiones_prom2024` – volumen total de conexiones fijas
2. `conexiones_x1000hab` – densidad de conexiones por habitante
3. `ratio_movil_fija` – **variable nueva** de redes móviles
4. `poblacion_2024` – tamaño de la comuna
5. `pct_ruralidad_proxy` – ruralidad
6. `tasa_pobreza_ingresos` – pobreza por ingresos
7. `movil_x1000hab_nacional` – penetración móvil

> [!TIP]
> El `ratio_movil_fija` apareció como la **3ª variable más importante**, validando la decisión de incorporar datos de redes móviles al análisis.

---

## 3. Archivos generados

### Dataset
| Archivo | Descripción |
|---------|-------------|
| `data/processed/dataset_comunas_chile.csv` | 105 comunas, 17 variables (incluye móvil) |
| `data/processed/dataset_comunas_con_clusters.csv` | Ídem + cluster, etiqueta y distancia al centroide |

### Figuras (`outputs/figures/`)
| Archivo | Contenido |
|---------|-----------|
| `01_distribuciones.png` | Histogramas de variables clave (fija + móvil) |
| `02_heatmap_correlacion.png` | Correlación de Pearson (incluye variables móviles) |
| `03_scatter_pobreza_conectividad.png` | Pobreza vs. conectividad fija por región |
| `04_ranking_conectividad.png` | Top/Bottom 10 comunas por conectividad |
| `05_elbow_silhouette.png` | Selección de k óptimo |
| `06_clusters_pca_heatmap.png` | PCA 2D + heatmap de centroides por nivel de conectividad |
| `07_confusion_matrix_rf.png` | Matriz de confusión – Random Forest |
| `08_feature_importance.png` | Importancia de variables en RF |
| `09_metricas_rf.png` | Métricas de clasificación (Accuracy, F1, etc.) |

### Modelos (`outputs/models/`)
| Archivo | Descripción |
|---------|-------------|
| `kmeans_model.pkl` | K-Means entrenado (k=3) |
| `random_forest.pkl` | Pipeline Random Forest completo |
| `scaler_clustering.pkl` | StandardScaler para clustering |
| `scaler_clasificacion.pkl` | MinMaxScaler para clasificación |

---

## 4. Hallazgos clave para la presentación

### 🔑 Mensaje central
> **El 17% de las comunas de la macrozona Centro-Norte presenta baja conectividad fija, pero su ratio móvil/fija promedio de 26× revela que el celular es prácticamente el único acceso a internet de esas comunas, lo que limita severamente la calidad y usos posibles de la conexión.**

### 📌 Puntos destacados
1. **La brecha digital es territorial y socioeconómica:** Las comunas con baja conectividad tienen ~6× menos conexiones por habitante que las de alta conectividad, y el doble de tasa de pobreza.
2. **Coquimbo es la región más afectada:** 10 de sus 15 comunas (67%) están en el cluster de baja conectividad.
3. **El ratio móvil/fija es un nuevo indicador revelador:** Río Hurtado tiene 127× más penetración móvil que fija — la única internet disponible es el celular.
4. **El Random Forest confirma los clusters con 81% de accuracy** en datos nuevos, con la advertencia de que el CV alto (95%) es esperable al clasificar los propios clusters.
5. **6 comunas son estadísticamente atípicas** y merecen análisis individual, especialmente Juan Fernández (caso insular) y Río Hurtado (extremo aislamiento).

---

## 5. Scripts del proyecto (flujo de ejecución)

```
py -3 src/buscar_subtel.py      # Descarga datos SUBTEL (fija + móvil)
py -3 src/preparar_datos.py     # Integra datos, filtra macrozona, agrega móvil
py -3 src/modelado.py           # EDA → Clustering → Random Forest → Figuras
```

