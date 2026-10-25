# Guía y Estructura Completa para Presentación en Canva
## Análisis de la Brecha Digital y Calidad de Red en Chile (UDP - Data Science)

**Autores:** Diego Lara · Cristóbal Castillo  
**Profesor:** Jorge Rozas  
**Institución:** Universidad Diego Portales (UDP) — Ramo: Data Science

---

## 📌 ¿Por qué Canva y cómo armarla rápidamente?
Canva es una plataforma visual ideal para comunicar resultados de ciencia de datos a audiencias ejecutivas y académicas. 
1. Abre Canva (https://www.canva.com) y elige **"Presentación (16:9)"**.
2. Selecciona una plantilla tecnológica, sobria o ejecutiva (colores sugeridos: azul marino `#0a1628`, cian `#00d4ff`, verde menta `#00e5a0` y blanco).
3. Sube las 9 imágenes que ya están generadas en tu carpeta:  
   `c:\Users\chipi\Desktop\Data Science\outputs\figures\`
4. Sigue la estructura de 14 diapositivas que detallamos a continuación. Cada lámina tiene:
   - **Título sugerido**
   - **Qué colocar en la diapositiva (Bullet points cortos y visuales)**
   - **Imagen/Gráfico a insertar**
   - **Guión oral detallado (lo que debes decir al exponer)**

---

# 📑 Estructura Diapositiva por Diapositiva

---

### Diapositiva 1: Portada
- **Título grande:** Brecha Digital y Calidad de Red en Chile
- **Subtítulo:** Segmentación comunal y modelado predictivo mediante Machine Learning
- **Datos:** Diego Lara · Cristóbal Castillo | Profesor: Jorge Rozas | Data Science 2026
- **Elemento visual:** Fondo tecnológico azul oscuro con mapa o red de nodos.
- **🎙️ Qué decir al exponer:**
  > "Buenas tardes profesor y compañeros. Hoy presentaremos nuestro proyecto de ciencia de datos enfocado en la Brecha Digital en Chile. Nuestro objetivo no fue solo describir la desigualdad digital, sino cuantificarla a nivel comunal e integrar técnicas no supervisadas y supervisadas de Machine Learning para caracterizar y predecir la vulnerabilidad digital de las 346 comunas del país."

---

### Diapositiva 2: Planteamiento del Problema y Pregunta de Investigación
- **Contenido en lámina:**
  - Acceso a internet: de bien de consumo a derecho habilitador esencial (educación, salud, trámites, empleo).
  - Brecha territorial: marcada asimetría entre grandes centros urbanos y comunas rurales/aisladas.
  - **Pregunta central:** *¿Qué factores socioeconómicos y demográficos explican la brecha digital en Chile, y es posible predecir el perfil de vulnerabilidad de una comuna a partir de sus variables estructurales?*
- **🎙️ Qué decir al exponer:**
  > "La pandemia demostró que no tener acceso a internet de alta velocidad excluye a las familias de oportunidades laborales, telemedicina y educación. En Chile existen realidades comunales diametralmente opuestas: comunas del sector oriente de Santiago con penetraciones superiores a 300 conexiones por cada mil habitantes, frente a comunas del sur o cordilleranas donde no superan las 30. Nos planteamos entonces modelar matemáticamente qué factores determinan esta brecha y construir un clasificador capaz de predecir la vulnerabilidad de cada municipio."

---

### Diapositiva 3: Justificación de las Fuentes de Datos
- **Contenido en lámina:**
  - **SUBTEL (Subsecretaría de Telecomunicaciones):** Estadísticas de Internet Fija por Comuna (conexiones 2024-2025). Variable clave de infraestructura digital.
  - **INE (Instituto Nacional de Estadísticas):** Proyecciones poblacionales comunales 2024. Permite calcular tasas relativas (*conexiones por cada 1.000 habitantes*).
  - **CASEN 2022 (Ministerio de Desarrollo Social):** Pobreza por ingresos (%), Pobreza multidimensional (%) e ingreso per cápita promedio.
- **🎙️ Qué decir al exponer:**
  > "¿Por qué estas tres bases de datos? Porque la brecha digital no es solo un problema de cables o antenas, es un fenómeno socio-técnico. SUBTEL nos da la métrica dura de telecomunicaciones; el INE nos permite normalizar los números absolutos respecto a la población real para que comunas pequeñas como Ollagüe o Tortel sean comparables con comunas gigantes como Puente Alto; y CASEN 2022 nos entrega el contexto de vulnerabilidad socioeconómica de los hogares."

---

### Diapositiva 4: Procesamiento y Limpieza de Datos (Pipeline de Datos)
- **Contenido en lámina:**
  - **Deduplicación:** Resolución de las 367 a 346 comunas oficiales (eliminación de duplicados causados por la creación de la Región de Ñuble).
  - **Normalización de Llaves:** Homogeneización de nombres comunales mediante Códigos Únicos Territoriales (CUT) y eliminación de tildes/espacios (*ej: "Aysén" vs "Aisén", "Llaillay" vs "Llay-Llay"*).
  - **Tratamiento de Nulos y Outliers:** Imputación por mediana regional y winsorización en percentiles 1-99 para controlar valores atípicos sin distorsionar la distribución.
  - **Ingeniería de Características:** Creación del indicador `conex_por_1000_hab = (conexiones / poblacion) * 1000`.
- **🎙️ Qué decir al exponer:**
  > "En la preparación de datos enfrentamos varios desafíos reales: primero, los reportes antiguos contenían códigos duplicados por la transición de la Región de Ñuble, resultando en 367 filas que debieron ser filtradas a las 346 comunas oficiales de Chile. Segundo, corregimos desalineaciones de texto en SUBTEL mediante normalización y mapeo de sinónimos. Y tercero, para evitar que una comuna con población minúscula disparara los valores a extremos irreales, aplicamos winsorización en los percentiles extremos."

---

### Diapositiva 5: Análisis Exploratorio de Datos (EDA)
- **Imagen a insertar:** `outputs/figures/01_distribuciones.png` y `outputs/figures/02_heatmap_correlacion.png`
- **Contenido en lámina:**
  - Asimetría positiva severa en conexiones fijas: una minoría de comunas concentra la mayor parte de la infraestructura.
  - **Correlación inversa fuerte:** A mayor tasa de pobreza multidimensional y ruralidad, menor penetración de internet ($r \approx -0.55$).
  - **Correlación positiva:** El ingreso per cápita comunal correlaciona fuertemente con la tasa de conexión ($r \approx +0.62$).
- **🎙️ Qué decir al exponer:**
  > "Al explorar los datos observamos distribuciones fuertemente sesgadas. En el histograma se evidencia que más del 70% de las comunas chilenas están por debajo del promedio nacional de conectividad. Además, la matriz de correlación confirma empíricamente que la brecha digital está intrínsecamente ligada al ingreso y a la ruralidad: las comunas más pobres y aisladas presentan un déficit sistemático de infraestructura."

---

### Diapositiva 6: ¿Por qué y cómo usamos K-Means? (Aprendizaje No Supervisado)
- **Contenido en lámina:**
  - **Objetivo:** Descubrir agrupaciones naturales de vulnerabilidad comunal sin sesgo humano ni etiquetas predefinidas.
  - **¿Cómo funciona el algoritmo?**
    1. Estandarización de variables ($Z$-Score) para equiparar magnitudes.
    2. Inicialización de $k$ centroides en el espacio multidimensional.
    3. Asignación de cada comuna al centroide más cercano (distancia euclidiana).
    4. Recálculo iterativo de centroides hasta convergencia del error cuadrático.
  - **Selección de $k$:** Evaluación de método del codo (Inercia) y Coeficiente de Silueta.
- **🎙️ Qué decir al exponer:**
  > "Dado que no existía una etiqueta oficial que dijera 'esta comuna es vulnerable o no', utilizamos K-Means, el algoritmo no supervisado por excelencia. K-Means busca minimizar la varianza intra-cluster y maximizar la distancia entre grupos. Antes de aplicarlo estandarizamos las variables, porque sin escalado, una variable con valores de millones como los ingresos dominaría por completo a porcentajes que van de 0 a 100."

---

### Diapositiva 7: Determinación de Clusters Óptimos ($k=3$)
- **Imagen a insertar:** `outputs/figures/05_elbow_silhouette.png`
- **Contenido en lámina:**
  - Gráfico de Inercia (Codo) vs Coeficiente de Silueta.
  - **$k=3$ es el punto óptimo:** Máximo global de Silhouette Score (**0.338**).
  - Justificación de negocio/política pública: Divide la realidad comunal en 3 niveles de gestión claros:
    - 🔴 **Alta Vulnerabilidad** (151 comunas - 43.6%)
    - 🟡 **Vulnerabilidad Media** (148 comunas - 42.8%)
    - 🟢 **Baja Vulnerabilidad** (47 comunas - 13.6%)
- **🎙️ Qué decir al exponer:**
  > "Para determinar el número óptimo de clusters evaluamos de $k=2$ a $k=8$. El gráfico de silueta demostró un pico óptimo en $k=3$ con un score de 0.338. Además, desde el punto de vista de políticas públicas, 3 categorías (Alta, Media y Baja vulnerabilidad) proporcionan una taxonomía clara y accionable para focalizar subsidios e inversiones de telecomunicaciones."

---

### Diapositiva 8: Caracterización de los Perfiles de Vulnerabilidad
- **Imagen a insertar:** `outputs/figures/06_clusters_pca_heatmap.png`
- **Contenido en lámina (Tabla Resumen):**
  - **Cluster Alta Vulnerabilidad (151 comunas):** Conectividad promedio bajísima (~78 conex/1.000 hab), alta ruralidad (>45%), alta pobreza multidimensional.
  - **Cluster Vulnerabilidad Media (148 comunas):** Conectividad moderada (~180 conex/1.000 hab), ciudades intermedias, transición socioeconómica.
  - **Cluster Baja Vulnerabilidad (47 comunas):** Alta conectividad (~258 conex/1.000 hab), comunas metropolitanas de altos ingresos, baja pobreza.
- **🎙️ Qué decir al exponer:**
  > "En este mapa de calor y proyección PCA vemos la separación nítida de los tres grupos. El 43.6% de las comunas del país cae en el cluster de alta vulnerabilidad; son localidades rurales donde las empresas privadas de telecomunicaciones no tienen incentivo económico de desplegar fibra óptica. El grupo de baja vulnerabilidad representa apenas al 13.6% de los municipios, evidenciando una fuerte centralización territorial."

---

### Diapositiva 9: Modelado Predictivo Supervisado (Pipeline y Metodología)
- **Contenido en lámina:**
  - **Objetivo:** Predecir el perfil de vulnerabilidad de una comuna a partir de sus atributos socioeconómicos.
  - **Prevención de Data Leakage:**
    - División estratificada: 80% Entrenamiento (276 comunas) / 20% Prueba (70 comunas).
    - Uso de `sklearn.pipeline.Pipeline`: el `StandardScaler` se ajusta **únicamente** sobre el conjunto de Train y se aplica sobre Test.
  - **Modelos evaluados:**
    1. *Regresión Logística Multinomial:* Modelo lineal probabilístico base (baseline interpretable).
    2. *Random Forest Classifier:* Ensamble de 200 árboles de decisión para capturar no-linealidades.
- **🎙️ Qué decir al exponer:**
  > "Una vez generadas las etiquetas con K-Means, pasamos a la fase supervisada: entrenar modelos que puedan predecir este perfil. Para garantizar rigor metodológico usamos scikit-learn Pipelines, evitando cualquier tipo de fuga de datos (data leakage). Comparamos un modelo lineal interpretable como la Regresión Logística frente a un ensamble no lineal como Random Forest."

---

### Diapositiva 10: ¿Cómo Funcionan Concretamente los Modelos?
- **Contenido en lámina:**
  - **Regresión Logística Multinomial:**
    - Modela la probabilidad de pertenencia a cada clase mediante la función Softmax: $P(Y=k|X) = \frac{e^{\beta_k^T X}}{\sum e^{\beta_j^T X}}$.
    - Establece hiperplanos de decisión lineales.
  - **Random Forest Classifier:**
    - Construye un bosque de árboles de decisión independientes entrenados con subconjuntos aleatorios de comunas y variables (Bagging).
    - La predicción final se obtiene por votación mayoritaria, reduciendo la varianza y evitando el sobreajuste (overfitting).
- **🎙️ Qué decir al exponer:**
  > "¿Cómo funciona cada modelo tras bambalinas? La Regresión Logística calcula combinaciones lineales de los atributos y las transforma en probabilidades mediante una función softmax. Si los datos no son perfectamente separables por planos rectos, su rendimiento tiene un techo. En cambio, Random Forest genera cientos de árboles que dividen el espacio mediante reglas lógicas y promedian sus decisiones. Esto le permite capturar interacciones complejas, por ejemplo, cuando una comuna es rural pero tiene altos ingresos mineros."

---

### Diapositiva 11: Resultados y Rendimiento de los Modelos
- **Imagen a insertar:** `outputs/figures/07_confusion_matrices.png` y `outputs/figures/09_comparativa_modelos.png`
- **Contenido en lámina (Tabla Comparativa):**
  | Modelo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score |
  |---|---|---|---|---|
  | **Regresión Logística** | 94.3% | 0.940 | 0.941 | 0.940 |
  | **Random Forest** | **97.1%** | **0.973** | **0.970** | **0.971** |
  - Matriz de confusión: Random Forest cometió únicamente 2 errores de clasificación en todo el conjunto de prueba.
- **🎙️ Qué decir al exponer:**
  > "Ambos modelos tuvieron un desempeño sobresaliente, pero Random Forest alcanzó un 97.1% de Accuracy y F1-Score. En la matriz de confusión pueden ver que de 70 comunas del conjunto de prueba, Random Forest solo confundió 2 comunas limítrofes entre vulnerabilidad media y alta, sin cometer ningún error de tipo crítico (como clasificar una comuna de alta vulnerabilidad como baja)."

---

### Diapositiva 12: Importancia de Variables (Feature Importance)
- **Imagen a insertar:** `outputs/figures/08_feature_importance.png`
- **Contenido en lámina:**
  - Ranking de variables determinantes según reducción de impureza de Gini:
    1. **Conexiones fijas por 1.000 hab:** Máximo peso explicativo (~38%).
    2. **Ingreso promedio comunal:** Factor socioeconómico directo (~24%).
    3. **Porcentaje de ruralidad:** Barrera geográfica y de densidad (~18%).
    4. **Pobreza multidimensional:** Vulnerabilidad estructural del hogar (~12%).
    5. **Población total:** Efecto de escala comunal (~8%).
- **🎙️ Qué decir al exponer:**
  > "Este gráfico de Feature Importance es uno de los resultados más valiosos del proyecto. Demuestra cuantitativamente que la penetración de internet fija y el ingreso comunal explican más del 60% de la condición de vulnerabilidad. La ruralidad actúa como el tercer factor determinante, lo que valida que la falta de conectividad está fuertemente anclada al aislamiento geográfico."

---

### Diapositiva 13: Conclusiones y Recomendaciones de Política Pública
- **Contenido en lámina:**
  - **Diagnóstico:** Chile presenta una fractura digital estructural: 151 comunas requieren intervención urgente.
  - **Eficacia del Modelo:** El pipeline desarrollado permite diagnosticar y re-evaluar la vulnerabilidad comunal anualmente con un 97% de precisión.
  - **Recomendaciones:**
    1. **Subsidio a la Demanda e Infraestructura:** Subsidios cruzados para el despliegue de última milla en las 151 comunas del Cluster 1.
    2. **Conectividad Satelital y Móvil 5G:** Complementar en zonas de difícil acceso donde la fibra óptica no es viable financieramente.
    3. **Capacitación Digital:** Acompañar el fierro con programas de alfabetización digital para adultos mayores y comunidades rurales.
- **🎙️ Qué decir al exponer:**
  > "Para concluir: nuestro modelo no solo clasifica comunas, entrega una herramienta concreta para el Estado. Recomendamos que SUBTEL y el Ministerio de Desarrollo Social utilicen esta segmentación para priorizar subsidios de infraestructura y proyectos de fibra óptica nacional, priorizando las 151 comunas críticas identificadas."

---

### Diapositiva 14: Preguntas y Agradecimientos
- **Contenido en lámina:**
  - Muchas gracias por su atención.
  - Repositorio y código reproducible disponible en GitHub: `github.com/Dmonjer/brecha-digital-chile`.
  - Ronda de preguntas.
- **🎙️ Qué decir al exponer:**
  > "Muchas gracias por su atención. Quedamos a su entera disposición para responder cualquier pregunta o profundizar en los detalles técnicos de los algoritmos y el preprocesamiento."

---

## 🎨 Consejos Visuales para Canva
1. **Tipografía recomendada en Canva:** *Montserrat* o *Inter* para títulos (en Negrita), y *Open Sans* o *Roboto* para cuerpo.
2. **Uso de imágenes:** Sube las figuras de `outputs/figures/` y agrégales un sutil marco o sombra paralela en Canva (Efectos -> Sombras -> Paralela) para darles un acabado profesional.
3. **No satures de texto:** La lámina debe tener los puntos clave y gráficos; el detalle profundo lo darás tú con el guión oral que preparamos arriba.
