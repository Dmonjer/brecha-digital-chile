# INFORME TÉCNICO ETAPA 3: MODELADO Y ANÁLISIS DE LA BRECHA DIGITAL Y CALIDAD DE RED EN CHILE
## MACROZONA CENTRO-NORTE: REGIONES DE COQUIMBO, VALPARAÍSO Y METROPOLITANA

---

**Institución:** Universidad Diego Portales (UDP)  
**Facultad:** Facultad de Ingeniería y Ciencias / Escuela de Informática y Telecomunicaciones  
**Asignatura:** Data Science (2026-02)  
**Profesor Guía:** Jorge Rozas  
**Estudiantes:** Diego Lara · Cristóbal Castillo  
**Fecha de Entrega:** Noviembre 2026  
**Repositorio GitHub:** [brecha-digital-chile](https://github.com/Dmonjer/brecha-digital-chile)

---

## RESUMEN EJECUTIVO (ABSTRACT)

El presente informe expone los resultados definitivos de la **Etapa 3** del proyecto de investigación sobre la brecha digital comunal en Chile. A diferencia de las aproximaciones tradicionales basadas exclusivamente en suscripciones de banda ancha fija residencial, esta investigación incorpora de manera empírica la **infraestructura de red móvil celular** como vector fundamental de acceso, utilizando el catastro nacional oficial de **13.823 estaciones base y antenas de SUBTEL**. 

El análisis se focalizó en las **105 comunas de la Macrozona Centro-Norte** (Región de Coquimbo [IV], Región de Valparaíso [V] y Región Metropolitana [RM]), territorio que concentra a más de 11,3 millones de habitantes (60% del país) y exhibe tanto el polo financiero de mayor desarrollo como valles rurales en severo aislamiento. 

A través de un pipeline automatizado y metodológicamente auditado, se aplicó agrupamiento no supervisado mediante **K-Means ($k=3$)**, alcanzando un **Silhouette Score de 0.3473**, el cual categoriza a las comunas en tres niveles objetivos: **Alta conectividad** ($n=12$), **Conectividad media** ($n=52$) y **Baja conectividad** ($n=41$). Posteriormente, se entrenó un clasificador supervisado **Random Forest** bajo validación cruzada estratificada de 5 pliegues (*Stratified 5-Fold CV*), logrando un **F1-Macro de 91.22% ± 7.34%** y un **Accuracy de 95.29% ± 4.40%**, alcanzando una generalización del 100% en el conjunto de prueba independiente ($n=21$). El análisis de importancia de características (*Feature Importance*) demostró un equilibrio sustantivo entre la red fija (**15.4%**) y la infraestructura celular (**14.0%**), ratificando que la brecha digital contemporánea en Chile es un fenómeno multidimensional donde la telefonía móvil actúa como infraestructura crítica de subsistencia en zonas rurales y periféricas.

---

## 1. INTRODUCCIÓN Y PLANTEAMIENTO DEL PROBLEMA

El acceso a internet de alta velocidad y calidad constituye un pilar habilitante para el ejercicio de derechos fundamentales: educación digital, teletrabajo, salud remota, trámites del Estado y desarrollo económico. No obstante, en Chile coexisten realidades contrapuestas: mientras los sectores acomodados acceden a fibra óptica simétrica gigabit y cobertura multi-operador 5G, decenas de comunas rurales e insulares experimentan una marcada exclusión digital.

### Pregunta de Investigación
> *¿De qué manera la interacción entre la penetración de internet fijo residencial y la infraestructura de telecomunicaciones móviles, condicionada por factores demográficos y socioeconómicos, estructura la brecha digital en la Macrozona Centro-Norte de Chile, y con qué grado de precisión es posible predecir el perfil de conectividad integral de cada comuna?*

### Objetivos del Proyecto
1. **Integrar fuentes oficiales multidimensionales:** Unificar los registros de conexiones fijas residenciales de SUBTEL, el catastro de estaciones base de telefonía e internet móvil, las proyecciones de población del INE y las estimaciones comunales de pobreza de la encuesta CASEN 2022.
2. **Segmentar el territorio objetivamente:** Aplicar algoritmos de Machine Learning no supervisado para agrupar las comunas según su disponibilidad de conectividad fija y celular sin recurrir a juicios de valor preestablecidos.
3. **Entrenar y evaluar un clasificador predictivo robusto:** Desarrollar un modelo supervisado (Random Forest) con métricas inmunes al desbalance de clases (F1-Macro) y sin fuga de datos (*data leakage*).
4. **Identificar anomalías geográficas y brechas territoriales:** Detectar comunas con comportamientos atípicos (aislamiento insular, dispersión en alta montaña) para formular recomendaciones pertinentes de política pública.

---

## 2. DELIMITACIÓN TERRITORIAL: MACROZONA CENTRO-NORTE

En etapas preliminares del proyecto se consideró un alcance exploratorio sobre las 346 comunas de Chile. No obstante, para la Etapa 3 se adoptó la decisión metodológica de acotar el estudio a la **Macrozona Centro-Norte**, integrada por las regiones de:
- **Región de Coquimbo (4):** 15 comunas.
- **Región de Valparaíso (5):** 38 comunas.
- **Región Metropolitana de Santiago (13):** 52 comunas.
- **Total:** **105 comunas**.

### Justificación Metodológica:
1. **Continuidad y Coherencia Territorial:** Representa una franja geográfica y productiva interconectada que abarca desde la zona semiárida y minera del Norte Chico hasta el valle central agrícola y la gran metrópolis de Santiago.
2. **Representatividad Poblacional:** Alberga a 11.325.689 personas proyectadas al 2024, concentrando más del 58% de la población nacional.
3. **Contraste Socioeconómico y Geográfico:** Permite estudiar simultáneamente comunas con menos de 1% de pobreza por ingresos (Las Condes, Vitacura) junto a comunas con más de 14% de pobreza y alta ruralidad (Punitaqui, San Pedro, Canela), así como territorios insulares extremos (Archipiélago de Juan Fernández, Isla de Pascua).
4. **Consistencia de Datos:** Evita sesgos extremos introducidos por la disgregación de zonas patagónicas o desérticas extremas, permitiendo un modelado mucho más calibrado y comparable.

---

## 3. AUDITORÍA TÉCNICA Y RESOLUCIÓN DE VULNERABILIDADES

Previo a la consolidación de los resultados de esta etapa, se ejecutó una auditoría exhaustiva sobre el código y los artefactos del proyecto. Se identificaron vulnerabilidades metodológicas y de implementación que fueron resueltas íntegramente:

| Vulnerabilidad Auditada | Naturaleza | Impacto Previo | Solución Implementada en Etapa 3 |
|---|---|---|---|
| **Ruta errónea de SUBTEL** | Bug de código | `preparar_datos.py` arrojaba `FileNotFoundError` al buscar un archivo inexistente. | Se corrigió la ruta apuntando a `data/raw/subtel/series_internet_fija_subtel.xlsx`. El pipeline ahora es 100% reproducible. |
| **Variables móviles constantes ($\sigma=0$)** | Falla metodológica | Se asignó una cifra nacional a todas las comunas; StandardScaler producía columnas nulas o irrelevantes. | Se extrajo el catastro de **13.823 antenas móviles** por comuna (`estaciones_base_antenas.xlsx`), creando variables con varianza comunal real. |
| **Ratio móvil/fija espurio** | Error conceptual | Se calculaba $k / \text{fija}$, siendo un artefacto matemático inverso de la red fija. | Se computó el ratio comunal auténtico: $\frac{\text{antenas\_x1000hab}}{\text{conexiones\_x1000hab}} \times 100$, midiendo la dependencia relativa de la red celular. |
| **Sesgo en métrica de evaluación** | Sesgo métrico | Se reportaba `f1_weighted`, inflado artificialmente por la clase mayoritaria (Conectividad media). | Se adoptó **`f1_macro`** como métrica primaria de optimización en validación cruzada 5-fold, evaluando a cada clase con igual ponderación. |
| **Muestra exigua en test set** | Fragilidad estadística | La clase minoritaria tenía solo $n=4$ comunas en el test set, generando métricas no generalizables. | Con la integración móvil real, el cluster de baja conectividad creció a 41 comunas, duplicando la muestra de test a **$n=8$ comunas**. |
| **Terminología estigmatizante** | Rigor conceptual | Se empleaba el término "vulnerable", sensible y susceptible a malinterpretaciones. | Se sustituyó universalmente por **Alta conectividad, Conectividad media y Baja conectividad**. |
| **Multiplicidad de modelos** | Complejidad innecesaria | La Regresión Logística presentaba supuestos lineales violados por distribuciones asimétricas. | Se eliminó la Regresión Logística, focalizando el trabajo en **Random Forest con Pipeline y MinMaxScaler**. |

---

## 4. FUENTES DE DATOS Y PIPELINE DE INTEGRACIÓN (ETL)

El dataset final se conformó a partir de cuatro fuentes primarias del Estado de Chile:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FUENTES OFICIALES RAW                           │
├─────────────────────┬──────────────────┬───────────────────────────────┤
│ Organismo Emisor    │ Archivo Fuente   │ Variables Extraídas           │
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ SUBTEL              │ series_fija.xlsx │ Conexiones fijas 2023-2024    │
│ SUBTEL              │ antenas.xlsx     │ 13.823 estaciones base móvil  │
│ INE                 │ poblacion.xlsx   │ Población comunal 2023-2024   │
│ CASEN / MDSF        │ casen_2022.xlsx  │ Pobreza por ingresos y multi. │
└─────────────────────┴──────────────────┴───────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 INGENIERÍA DE DATOS (preparar_datos.py)                 │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Normalización de cadenas y resolución de sinonimias comunales       │
│ 2. Fusión por Código Único Territorial (CUT) oficial                   │
│ 3. Cálculo de tasas normalizadas por habitante (x1.000 hab.)           │
│ 4. Filtrado geográfico estricto a regiones 4, 5 y 13 (105 comunas)     │
│ 5. Control de valores nulos (0 nulos en dataset procesado)             │
└────────────────────────────────────────────────────────────────────────┘
```

### Variables del Dataset Procesado (`dataset_comunas_chile.csv`)
1. `num_region`: Código numérico de región (4, 5, 13).
2. `cut_comuna`: Código Único Territorial oficial del INE.
3. `nombre_comuna`: Nombre oficial de la comuna.
4. `poblacion_2024`: Población total proyectada al 2024 (INE).
5. `poblacion_2023`: Población proyectada al 2023 (INE).
6. `conexiones_prom2024`: Promedio mensual de conexiones fijas residenciales en 2024 (SUBTEL).
7. `conexiones_prom2023`: Promedio mensual de conexiones fijas residenciales en 2023 (SUBTEL).
8. `conexiones_recientes`: Último dato mensual reportado (SUBTEL).
9. `conexiones_x1000hab`: Tasa de penetración fija: $(\text{conexiones\_prom2024} / \text{población\_2024}) \times 1.000$.
10. `antenas_movil`: Cantidad total de antenas/estaciones base de telefonía celular en la comuna (SUBTEL).
11. `antenas_x1000hab`: Densidad física de infraestructura móvil: $(\text{antenas\_movil} / \text{población\_2024}) \times 1.000$.
12. `operadores_movil`: Conteo de empresas proveedoras con infraestructura propia en la comuna.
13. `ratio_movil_fija`: Relación porcentual entre densidad de antenas y penetración fija: $(\text{antenas\_x1000hab} / \text{conexiones\_x1000hab}) \times 100$.
14. `tasa_pobreza_ingresos`: Porcentaje de personas bajo la línea de pobreza por ingresos (CASEN 2022).
15. `tasa_pobreza_multidimensional`: Porcentaje de personas con carencias multidimensionales (CASEN 2022).
16. `es_rural`: Indicador binario de ruralidad según clasificación INE.
17. `pct_ruralidad_proxy`: Estimación acotada en tramos del porcentaje de población rural comunal.

---

## 5. ANÁLISIS EXPLORATORIO DE DATOS (EDA)

El análisis exploratorio reveló asimetrías estructurales en la distribución de la infraestructura de telecomunicaciones dentro de la macrozona:

### 5.1 Estadísticas Descriptivas de las Variables Clave ($N=105$)

| Variable | Media | Desv. Est. ($\sigma$) | Mínimo | Mediana | Máximo |
|---|---|---|---|---|---|
| **Conexiones fijas / 1.000 hab.** | 196.57 | 90.07 | 9.01 | 200.95 | 431.89 |
| **Antenas móviles / 1.000 hab.** | 1.59 | 1.31 | 0.37 | 1.24 | 6.62 |
| **Ratio móvil / fija** | 1.44 | 3.13 | 0.17 | 0.62 | 23.18 |
| **Tasa pobreza por ingresos** | 5.9% | 2.9% | 0.9% | 5.5% | 15.9% |
| **Población 2024** | 107.864 | 126.759 | 1.104 | 68.939 | 667.904 |

### 5.2 Hallazgos de Correlación (Matriz de Pearson)
- **Conexiones fijas vs. Pobreza por ingresos ($r = -0.57$):** Fuerte correlación negativa. A mayor tasa de pobreza comunal, menor penetración de banda ancha fija residencial.
- **Conexiones fijas vs. Antenas por habitante ($r = +0.19$):** Correlación lineal débil y ortogonal. La infraestructura móvil celular no duplica la información de la red fija, sino que aporta información complementaria sobre el despliegue territorial.
- **Ratio móvil/fija vs. Ruralidad ($r = +0.62$):** En comunas rurales, la relación entre antenas celulares y conexiones fijas es máxima. Debido a la baja o nula instalación de tendidos de fibra óptica, el celular se consolida como el principal acceso a internet.

---

## 6. AGRUPAMIENTO NO SUPERVISADO (K-MEANS)

### 6.1 Preparación de la Matriz de Características
Para garantizar que el algoritmo no fuera distorsionado por escalas de magnitud dispares o valores atípicos severos, se aplicó:
1. **Winsorización conservadora:** Acotamiento al percentil 2 ($p_2$) y percentil 98 ($p_{98}$).
2. **Estandarización:** Escalado $Z$-Score (`StandardScaler`) con media 0 y varianza unitaria 1:
$$z = \frac{x - \mu}{\sigma}$$
Las variables seleccionadas para el espacio de agrupamiento fueron:
`conexiones_x1000hab`, `antenas_x1000hab`, `tasa_pobreza_ingresos`, `poblacion_2024` y `pct_ruralidad_proxy`.

### 6.2 Determinación del Número Óptimo de Clusters ($k$)
Se evaluaron iterativamente valores de $k \in [2, 6]$ empleando 30 inicializaciones aleatorias con el método K-Means++:

```
k=2: Inercia = 342.7  |  Silhouette Score = 0.3306
k=3: Inercia = 267.6  |  Silhouette Score = 0.3473  <-- ÓPTIMO
k=4: Inercia = 222.1  |  Silhouette Score = 0.2963
k=5: Inercia = 183.2  |  Silhouette Score = 0.3076
k=6: Inercia = 158.8  |  Silhouette Score = 0.3053
```

El valor de **$k=3$ maximiza el Silhouette Score global (0.3473)** y genera una reducción angular sustantiva en la inercia intracluster (método del codo), separando el espectro de conectividad en tres estratos cualitativa y cuantitativamente diferenciados.

### 6.3 Caracterización de los Perfiles de Conectividad

La asignación semántica de las etiquetas se fundamentó en un **Score Integral de Conectividad**:
$$\text{Score} = Z(\text{conexiones\_x1000hab}) + Z(\text{antenas\_x1000hab}) - Z(\text{tasa\_pobreza\_ingresos})$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PERFILES DEFINITIVOS DE CONECTIVIDAD                 │
├──────────────────────────┬──────────────┬──────────────┬───────────────┤
│ Indicador Promedio       │ Alta (n=12)  │ Media (n=52) │ Baja (n=41)   │
├──────────────────────────┼──────────────┼──────────────┼───────────────┤
│ Conexiones fijas / 1.000 │ 336.56       │ 223.03       │ 122.05        │
│ Antenas móvil / 1.000    │ 3.99         │ 0.90         │ 1.75          │
│ Tasa pobreza ingresos    │ 3.9%         │ 5.4%         │ 8.3%          │
│ % Ruralidad promedio     │ 30.8%        │ 10.4%        │ 42.7%         │
│ Población media          │ ~71.000 hab. │ ~181.000 hab.│ ~21.000 hab.  │
│ Score Integral           │ +3.00        │ -0.64        │ -2.37         │
└──────────────────────────┴──────────────┴──────────────┴───────────────┘
```

#### Descripción Detallada de los Clusters:
- 🟢 **Alta conectividad ($n=12$, 11.4%):** Representa el estrato de vanguardia digital. Presenta una penetración fija residencial sobresaliente (336.6 conexiones por 1.000 hab.) y una densa red celular multioperador (3.99 antenas por 1.000 hab.). Agrupa a los municipios de mayores ingresos del Gran Santiago (Las Condes, Providencia, Vitacura, La Reina) y a los principales balnearios y centros costeros de la Región de Valparaíso (Algarrobo, Concón, Santo Domingo, Zapallar, Puchuncaví).
- 🔵 **Conectividad media ($n=52$, 49.5%):** El estrato más numeroso de la macrozona. Reúne a las grandes comunas residenciales y centros urbanos consolidados (Santiago Centro, Maipú, Puente Alto, La Florida, Viña del Mar, Valparaíso, Quillota, La Serena, Coquimbo). La penetración fija es robusta (223.0 conexiones por 1.000 hab.), mientras que la densidad de antenas por habitante es relativamente baja (0.90) debido a la edificación en altura y concentración urbana, donde una sola antena cubre a miles de habitantes simultáneamente.
- 🔴 **Baja conectividad ($n=41$, 39.1%):** Estrato caracterizado por la precariedad de infraestructura fija (122.1 conexiones por 1.000 hab., casi un tercio del nivel alto) y los mayores índices de pobreza socioeconómica (8.3%). Abarca el 80% de las comunas de Coquimbo (Río Hurtado, Paiguano, Combarbalá), comunas del interior de Valparaíso (Petorca, Cabildo) y la periferia rural metropolitana (San Pedro, Alhué, María Pinto). En este estrato, el ratio móvil/fija es crítico: la población depende de la señal móvil para su vida cotidiana.

### 6.4 Distribución Geográfica Territorial por Región

```
Distribución Comunal por Región y Nivel de Conectividad:
┌─────────────────────┬──────────────────┬─────────────────┬─────────────────┬────────┐
│ Región              │ Alta Conectividad│ Conect. Media   │ Baja Conectivid.│ Total  │
├─────────────────────┼──────────────────┼─────────────────┼─────────────────┼────────┤
│ Coquimbo (4)        │ 0 (0.0%)         │ 3 (20.0%)       │ 12 (80.0%)      │ 15     │
│ Valparaíso (5)      │ 8 (21.1%)        │ 11 (28.9%)      │ 19 (50.0%)      │ 38     │
│ Metropolitana (13)  │ 4 (7.7%)         │ 38 (73.1%)      │ 10 (19.2%)      │ 52     │
├─────────────────────┼──────────────────┼─────────────────┼─────────────────┼────────┤
│ Total Macrozona     │ 12 (11.4%)       │ 52 (49.5%)      │ 41 (39.1%)      │ 105    │
└─────────────────────┴──────────────────┴─────────────────┴─────────────────┴────────┘
```

> [!NOTE]
> **Severidad de la Brecha en Coquimbo:** La Región de Coquimbo carece de comunas en el cluster de Alta conectividad y tiene al 80% de sus comunas clasificadas en Baja conectividad. Solo la conurbación La Serena-Coquimbo y la capital provincial de Ovalle alcanzan la Conectividad media.

### 6.5 Detección y Análisis de Casos Atípicos (Outliers)
Utilizando la distancia euclídea de cada comuna a su centroide asignado ($d = ||x_i - \mu_c||$), se fijó el umbral en el percentil 95 ($d > 2.74$). Se identificaron 6 comunas con dinámicas atípicas:

1. **La Higuera (IV Región, $d=4.09$):** Máxima distancia al centroide. Tiene una densidad excepcional de antenas móviles (5.92 por 1.000 hab.) pero una penetración fija ínfima (57.8 por 1.000 hab.). La geografía desértica y la Ruta 5 Norte demandan torres de transmisión vial que no abastecen a los poblados interiores.
2. **Santiago Centro (RM, $d=3.38$):** Posee el mayor número absoluto de antenas del país (1.460 estaciones base) y 273.5 conexiones fijas por 1.000 hab., influenciada por su rol como centro financiero, comercial y de servicios con una enorme población flotante.
3. **Maipú y Puente Alto (RM, $d=3.08$ y $d=2.99$):** Ambas comunas superan el medio millón de habitantes. Su volumen masivo de conexiones amortigua los ratios per cápita, situándose en una posición periférica dentro del cluster de conectividad media.
4. **Juan Fernández (V Región, $d=3.04$):** Territorio insular extremo en el Océano Pacífico. Registra 135.0 conexiones fijas por 1.000 hab., pero sus costos de transporte satelital, su reducido tamaño poblacional (~1.100 habitantes) y su vulnerabilidad logística la ubican como un caso singular de aislamiento estructural.
5. **Las Condes (RM, $d=2.76$):** Concentra 1.173 antenas celulares, 321.1 conexiones fijas por 1.000 hab. y una pobreza inferior al 1%, definiendo el límite superior de la alta conectividad urbana.

---

## 7. MODELADO PREDICTIVO SUPERVISADO (RANDOM FOREST)

### 7.1 Configuración del Pipeline y Protocolo de Entrenamiento
Se formuló un problema de clasificación multiclase donde la variable objetivo es el nivel de conectividad asignado por el clustering ($y \in \{0: \text{Alta}, 1: \text{Baja}, 2: \text{Media}\}$).

Para evitar cualquier tipo de **fuga de datos (data leakage)**, se implementó un pipeline estricto en Scikit-Learn:
- **Escalador:** `MinMaxScaler` ajustado únicamente sobre el conjunto de entrenamiento de cada pliegue.
- **Clasificador:** `RandomForestClassifier` con 300 árboles de decisión (`n_estimators=300`), criterio de división Gini, `min_samples_split=4`, `min_samples_leaf=2`, `max_features='sqrt'` y semilla pseudoaleatoria fija (`random_state=42`).
- **Partición:** 80% entrenamiento ($n=84$) y 20% prueba ($n=21$), estratificado por la variable objetivo.

### 7.2 Validación Cruzada Estratificada (5-Fold Stratified CV en Train)

El uso de **F1-Macro** como métrica de control garantiza que el desempeño sobre las clases minoritarias (Alta y Baja conectividad) no sea enmascarado por la clase mayoritaria (Conectividad media):

$$\text{F1-Macro} = \frac{1}{|C|} \sum_{c \in C} \text{F1}_c$$

```
Resultados de Validación Cruzada (5 Pliegues en Train, n=84):
- CV Accuracy:    95.29% ± 4.40%
- CV F1-Macro:    91.22% ± 7.34%  (Métrica primaria de evaluación imparcial)
- CV F1-Weighted: 94.89% ± 4.64%
```

> [!NOTE]
> El resultado de **91.22% en F1-Macro** certifica que el bosque aleatorio es capaz de reproducir de forma robusta las fronteras geométricas del espacio multidimensional sin sobreajustarse a la clase predominante.

### 7.3 Evaluación en Datos de Prueba Independientes ($n=21$)

Al aplicar el modelo entrenado sobre el conjunto de prueba retenido, se obtuvieron los siguientes resultados:

```
Reporte de Clasificación en Test (Datos Nunca Vistos):
───────────────────────────────────────────────────────────────────
Nivel de Conectividad    Precision    Recall    F1-Score    Soporte
───────────────────────────────────────────────────────────────────
Alta conectividad         1.0000      1.0000     1.0000        3
Baja conectividad         1.0000      1.0000     1.0000        8
Conectividad media        1.0000      1.0000     1.0000       10
───────────────────────────────────────────────────────────────────
Accuracy Global                                  1.0000       21
F1-Macro Promedio         1.0000      1.0000     1.0000       21
F1-Weighted Promedio      1.0000      1.0000     1.0000       21
───────────────────────────────────────────────────────────────────
```

```
Matriz de Confusión en Test:
                     Pred: Alta    Pred: Baja    Pred: Media
Real: Alta Conect.       3             0              0
Real: Baja Conect.       0             8              0
Real: Media Conect.      0             0             10
```

El incremento del soporte en Baja conectividad ($n=8$ comunas evaluadas en test) valida que las clasificaciones del modelo son sólidas y no dependen de artefactos muestrales.

### 7.4 Jerarquía de Importancia de Características (*Feature Importance*)
El análisis de reducción media de impureza de Gini demostró qué dimensiones determinan en mayor medida el perfil de conectividad comunal:

```
Importancia de Variables en Random Forest:
1. Población comunal (poblacion_2024):             30.3%  ██████████████████████████████
2. Porcentaje de ruralidad (pct_ruralidad_proxy):  28.5%  ████████████████████████████
3. Conexiones fijas / 1.000 (conexiones_x1000hab): 15.4%  ███████████████
4. Antenas móviles / 1.000 (antenas_x1000hab):     14.0%  ██████████████
5. Tasa pobreza ingresos (tasa_pobreza_ingresos):  11.7%  ███████████
```

### Interpretación Tecnológica:
1. **La escala demográfica y el territorio mandan (58.8% acumulado):** El tamaño poblacional y la ruralidad condicionan la viabilidad comercial del despliegue de infraestructura de telecomunicaciones por parte de las empresas operadoras.
2. **Equilibrio Fija/Móvil (29.4% conjunto):** La penetración fija residencial (15.4%) y la densidad de infraestructura móvil (14.0%) exhiben pesos equivalentes y complementarios. Una comuna no puede ser clasificada certeramente sin observar simultáneamente ambas tecnologías.
3. **El componente socioeconómico (11.7%):** La tasa de pobreza modula la capacidad de pago y la tasa de adopción de planes de internet en los hogares.

---

## 8. IMPLICANCIAS DE POLÍTICA PÚBLICA

Los hallazgos del modelo proporcionan evidencia empírica directa para la toma de decisiones gubernamentales y de regulación en telecomunicaciones:

1. **Subsidio a la demanda móvil en el Norte Chico:** El 80% de las comunas de Coquimbo se ubica en el estrato bajo. Dado el costo prohibitivo del tendido de fibra óptica en zonas cordilleranas y valles transversales, el Fondo de Desarrollo de las Telecomunicaciones (FDT) debe focalizar recursos en garantizar cuotas de datos móviles universales y redes 5G de banda baja (700 MHz) de amplio alcance en estas comunas.
2. **Coexistencia y redundancia en el Gran Santiago:** Las comunas urbanas medias y altas muestran alta concentración de conexiones fijas, pero baja densidad de antenas móviles por habitante. Ante emergencias sísmicas o catástrofes climáticas, la saturación celular en estas urbes densas es un riesgo de seguridad pública que requiere exigir mayor resiliencia y respaldo energético en estaciones base.
3. **El rezago periurbano de la Región Metropolitana:** Comunas como San Pedro, Alhué y María Pinto presentan perfiles idénticos a los valles interiores de Coquimbo a pesar de pertenecer a la capital del país. Las políticas de conectividad no pueden basarse en agregados regionales que ocultan la pobreza rural metropolitana.

---

## 9. GUÍA DE EJECUCIÓN Y REPRODUCIBILIDAD

El proyecto fue diseñado bajo el principio de reproducibilidad total. Para ejecutar el pipeline completo desde la terminal:

```bash
# 1. Clonar el repositorio y posicionarse en el directorio raíz
cd "proyecto-brecha-digital"

# 2. Instalar las dependencias de ciencia de datos
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl

# 3. Ejecutar la preparación e integración de datos
python src/preparar_datos.py

# 4. Ejecutar el pipeline de modelado, clustering, clasificación y figuras
python src/modelado.py
```

### Inventario de Artefactos Producidos
- **Datos procesados:** `data/processed/dataset_comunas_chile.csv` y `dataset_comunas_con_clusters.csv`.
- **Figuras generadas:** 9 figuras PNG de alta resolución en `outputs/figures/` (`01_distribuciones.png` a `09_metricas_rf.png`).
- **Modelos serializados:** `kmeans_model.pkl`, `scaler_clustering.pkl` y `random_forest.pkl` en `outputs/models/`.

---

## 10. CONCLUSIÓN GENERAL

La **Etapa 3** consolida la transformación del proyecto desde un análisis descriptivo básico hacia un **modelo integral de evaluación de la brecha digital comunal**. Al resolver la totalidad de los bugs y sesgos técnicos detectados en la auditoría inicial —en particular mediante la incorporación del catastro nacional de 13.823 antenas móviles y la adopción del F1-Macro como estándar de evaluación—, el estudio ofrece un diagnóstico fidedigno, matemáticamente respaldado y transparente de la realidad digital en la Macrozona Centro-Norte de Chile.

---

*Universidad Diego Portales · Facultad de Ingeniería y Ciencias · Noviembre 2026*
