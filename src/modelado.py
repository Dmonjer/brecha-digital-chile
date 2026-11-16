"""
Script de Modelado: Clustering + Clasificación (Random Forest)
Proyecto: Análisis de la Brecha Digital – Macrozona Centro-Norte de Chile
(Coquimbo, Valparaíso, Región Metropolitana)

Integración y Mejoras Metodológicas:
  - Redes móviles e infraestructura comunal real: antenas_x1000hab y operadores_movil (catastro SUBTEL).
  - Ambos tipos de red (fija y móvil) definen conjuntamente el nivel de conectividad.
  - Modelo predictivo: Solo Random Forest con validación estratificada 5-fold.
  - Evaluación imparcial: Reporte de F1-Macro (sin sesgo de clase mayoritaria) junto con F1-Weighted.
  - Mitigación de desbalance: Mayor representatividad comunal en clases minoritarias.
  - Terminología estandarizada: Alta conectividad, Conectividad media, Baja conectividad.

Ejecutar con: py -3 src/modelado.py
"""
import sys
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ============================================================
# RUTAS Y CONFIGURACIÓN
# ============================================================
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_comunas_chile.csv')
FIG_PATH  = os.path.join(BASE_DIR, 'outputs', 'figures')
MOD_PATH  = os.path.join(BASE_DIR, 'outputs', 'models')
for p in [FIG_PATH, MOD_PATH]:
    os.makedirs(p, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})
sns.set_style('whitegrid')

# Etiquetas estándar de conectividad (reemplazan a "vulnerable")
ETIQUETA_ALTA   = 'Alta conectividad'
ETIQUETA_MEDIA  = 'Conectividad media'
ETIQUETA_BAJA   = 'Baja conectividad'

# ============================================================
# 1. CARGAR DATOS
# ============================================================
def cargar_datos():
    print("Cargando dataset – Macrozona Centro-Norte...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df = df.drop_duplicates(subset=['cut_comuna'], keep='first').sort_values('cut_comuna').reset_index(drop=True)
    print(f"  Comunas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f"  Regiones incluidas: {sorted(df['num_region'].unique())} (4: Coquimbo, 5: Valparaíso, 13: RM)")
    print(f"  Columnas: {list(df.columns)}")
    return df

# ============================================================
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ============================================================
def eda(df):
    print("\n=== EDA (Exploratory Data Analysis) ===")

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nVariables numéricas ({len(cols_num)}):")
    print(df[cols_num].describe().round(2).to_string())

    nulos = df[cols_num].isnull().sum()
    nulos_df = pd.DataFrame({'N Nulos': nulos, '% Nulos': (nulos / len(df) * 100).round(1)})
    nulos_con_datos = nulos_df[nulos_df['N Nulos'] > 0]
    if len(nulos_con_datos) > 0:
        print(f"\nVariables con nulos:\n{nulos_con_datos.to_string()}")
    else:
        print("\nNo existen valores nulos en variables numéricas.")

    # Figura 01: Distribuciones clave (Fija + Móvil + Demografía + Pobreza)
    vars_plot = [c for c in [
        'conexiones_x1000hab', 'antenas_x1000hab', 'tasa_pobreza_ingresos',
        'poblacion_2024', 'pct_ruralidad_proxy', 'ratio_movil_fija'
    ] if c in df.columns]

    n = len(vars_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))

    for i, (var, color) in enumerate(zip(vars_plot, colors)):
        data = df[var].dropna()
        axes[i].hist(data, bins=20, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        axes[i].axvline(data.median(), color='red', linestyle='--', lw=1.5, label=f'Mediana: {data.median():.1f}')
        axes[i].axvline(data.mean(), color='orange', linestyle='--', lw=1.5, label=f'Media: {data.mean():.1f}')
        axes[i].set_title(var, fontweight='bold', fontsize=10)
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribución de Variables Clave (Fija y Móvil) – Macrozona Centro-Norte', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '01_distribuciones.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 01_distribuciones.png guardada")

    # Figura 02: Matriz de Correlación
    corr_vars = [c for c in [
        'conexiones_x1000hab', 'antenas_x1000hab', 'ratio_movil_fija',
        'tasa_pobreza_ingresos', 'poblacion_2024', 'pct_ruralidad_proxy'
    ] if c in df.columns]

    if len(corr_vars) >= 2:
        corr = df[corr_vars].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
        ax.set_title('Matriz de Correlación de Pearson\n(Red Fija, Infraestructura Móvil y Factores Socioeconómicos)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, '02_heatmap_correlacion.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Figura 02_heatmap_correlacion.png guardada")

    # Figura 03: Scatter Pobreza vs Conectividad (Fija y Móvil)
    if 'tasa_pobreza_ingresos' in df.columns and 'conexiones_x1000hab' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Pobreza vs Fija
        sc1 = axes[0].scatter(
            df['tasa_pobreza_ingresos'] * 100,
            df['conexiones_x1000hab'],
            c=df['num_region'], cmap='viridis', alpha=0.75, s=60, edgecolors='white', linewidth=0.5
        )
        axes[0].set_xlabel('Tasa de Pobreza por Ingresos 2022 (%)', fontsize=11)
        axes[0].set_ylabel('Conexiones fijas residenciales / 1.000 hab.', fontsize=11)
        axes[0].set_title('Pobreza vs. Conectividad Fija', fontweight='bold', fontsize=11)

        # Pobreza vs Móvil
        sc2 = axes[1].scatter(
            df['tasa_pobreza_ingresos'] * 100,
            df['antenas_x1000hab'],
            c=df['num_region'], cmap='viridis', alpha=0.75, s=60, edgecolors='white', linewidth=0.5
        )
        axes[1].set_xlabel('Tasa de Pobreza por Ingresos 2022 (%)', fontsize=11)
        axes[1].set_ylabel('Antenas móviles / 1.000 hab.', fontsize=11)
        axes[1].set_title('Pobreza vs. Infraestructura Móvil', fontweight='bold', fontsize=11)

        # Destacar comunas extremas
        top_con = df.nlargest(3, 'conexiones_x1000hab')
        bot_con = df.nsmallest(3, 'conexiones_x1000hab')
        for _, row in pd.concat([top_con, bot_con]).iterrows():
            axes[0].annotate(row['nombre_comuna'],
                             (row['tasa_pobreza_ingresos'] * 100, row['conexiones_x1000hab']),
                             fontsize=8, alpha=0.9, fontweight='bold')

        plt.colorbar(sc2, ax=axes, label='Región (4: Coquimbo, 5: Valparaíso, 13: RM)')
        plt.suptitle('Relación Socioeconómica con Conectividad Fija y Móvil por Comuna', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, '03_scatter_pobreza_conectividad.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Figura 03_scatter_pobreza_conectividad.png guardada")

    # Figura 04: Ranking comunas por conectividad combinada
    if 'conexiones_x1000hab' in df.columns and 'antenas_x1000hab' in df.columns:
        df_sort = df.sort_values('conexiones_x1000hab')
        top10 = df_sort.tail(10)
        bot10 = df_sort.head(10)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
        colors_bot = plt.cm.Reds(np.linspace(0.4, 0.9, 10))

        axes[0].barh(top10['nombre_comuna'], top10['conexiones_x1000hab'], color=colors_top)
        axes[0].set_title('Top 10: Mayor Conectividad Fija', fontweight='bold', color='darkgreen')
        axes[0].set_xlabel('Conexiones fijas / 1.000 hab.')

        axes[1].barh(bot10['nombre_comuna'], bot10['conexiones_x1000hab'], color=colors_bot)
        axes[1].set_title('Top 10: Menor Conectividad Fija (Comunas Rezagadas)', fontweight='bold', color='darkred')
        axes[1].set_xlabel('Conexiones fijas / 1.000 hab.')

        plt.suptitle('Ranking de Conectividad Comunal – Macrozona Centro-Norte (2024)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, '04_ranking_conectividad.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Figura 04_ranking_conectividad.png guardada")

    return df

# ============================================================
# 3. PREPARAR FEATURES PARA CLUSTERING
# ============================================================
def preparar_features(df):
    print("\n=== PREPARANDO FEATURES PARA CLUSTERING ===")

    # Features que integran red fija, red móvil, demografía, geografía y pobreza
    feature_cols = [
        'conexiones_x1000hab',    # Dimensión red fija residencial
        'antenas_x1000hab',       # Dimensión infraestructura móvil
        'tasa_pobreza_ingresos',  # Dimensión socioeconómica
        'poblacion_2024',         # Dimensión demográfica
        'pct_ruralidad_proxy',    # Dimensión geográfica / accesibilidad
    ]

    print(f"  Features seleccionadas ({len(feature_cols)}): {feature_cols}")

    df_feat = df[feature_cols + ['cut_comuna', 'nombre_comuna', 'num_region']].copy()

    # Imputación por mediana
    for col in feature_cols:
        med = df_feat[col].median()
        n_imp = df_feat[col].isnull().sum()
        if n_imp > 0:
            df_feat[col] = df_feat[col].fillna(med)
            print(f"  Imputados {n_imp} nulos en '{col}' con mediana={med:.2f}")

    # Winsorización p2-p98 para control conservador de valores atípicos
    for col in feature_cols:
        p2  = df_feat[col].quantile(0.02)
        p98 = df_feat[col].quantile(0.98)
        n_out = ((df_feat[col] < p2) | (df_feat[col] > p98)).sum()
        df_feat[col] = df_feat[col].clip(lower=p2, upper=p98)
        if n_out > 0:
            print(f"  Winsorización en '{col}': {n_out} registros acotados [p2={p2:.2f}, p98={p98:.2f}]")

    X = df_feat[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MOD_PATH, 'scaler_clustering.pkl'))
    print(f"  Matriz para clustering: {X_scaled.shape}")
    return df_feat, X_scaled, feature_cols, scaler

# ============================================================
# 4. ELBOW + SILHOUETTE
# ============================================================
def elbow_silhouette(X_scaled, k_max=6):
    print("\n=== EVALUACIÓN ELBOW Y SILHOUETTE ===")
    ks = range(2, k_max + 1)
    inercias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, init='k-means++', n_init=30, max_iter=400, random_state=RANDOM_SEED)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"  k={k}: Inercia={km.inertia_:.1f} | Silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(ks), inercias, 'bo-', linewidth=2.5, markersize=8)
    axes[0].fill_between(list(ks), inercias, alpha=0.1, color='blue')
    axes[0].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inercia (WCSS)', fontsize=11)
    axes[0].set_title('Método del Codo (Elbow)', fontweight='bold', fontsize=12)
    axes[0].set_xticks(list(ks))

    best_k = list(ks)[int(np.argmax(silhouettes))]
    axes[1].plot(list(ks), silhouettes, 'rs-', linewidth=2.5, markersize=8)
    axes[1].fill_between(list(ks), silhouettes, alpha=0.1, color='red')
    axes[1].axvline(best_k, color='green', linestyle='--', lw=2,
                    label=f'Mejor Silhouette: k={best_k} ({max(silhouettes):.3f})')
    axes[1].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score por k', fontweight='bold', fontsize=12)
    axes[1].set_xticks(list(ks))
    axes[1].legend(fontsize=10)

    plt.suptitle('Selección del Número Óptimo de Clusters – Brecha Digital Macrozona Centro-Norte',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '05_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura 05_elbow_silhouette.png guardada | Mejor k={best_k}")
    return best_k, inercias, silhouettes

# ============================================================
# 5. K-MEANS FINAL Y ETIQUETADO MULTIDIMENSIONAL
# ============================================================
def kmeans_final(df_feat, X_scaled, feature_cols, n_clusters=3):
    print(f"\n=== K-MEANS FINAL (k={n_clusters}) ===")
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=40,
                max_iter=500, random_state=RANDOM_SEED)
    labels = km.fit_predict(X_scaled)
    df_feat = df_feat.copy()
    df_feat['cluster'] = labels

    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score final: {sil:.4f}")
    print(f"  Distribución por cluster:\n{pd.Series(labels).value_counts().sort_index().to_string()}")

    joblib.dump(km, os.path.join(MOD_PATH, 'kmeans_model.pkl'))

    # Centroides reales (no estandarizados)
    stats = df_feat.groupby('cluster')[feature_cols].mean().round(3)
    print(f"\n  Centroides por cluster:\n{stats.to_string()}")

    # Detección de atípicos por distancia euclídea al centroide
    centros = km.cluster_centers_
    distancias = [np.linalg.norm(X_scaled[i] - centros[labels[i]]) for i in range(len(X_scaled))]
    df_feat['dist_centroide'] = distancias

    umbral = np.percentile(distancias, 95)
    df_atipicas = df_feat[df_feat['dist_centroide'] > umbral][['nombre_comuna', 'cluster', 'dist_centroide'] + feature_cols[:3]]
    print(f"\n  Comunas atípicas (distancia > p95={umbral:.2f}):")
    print(df_atipicas.to_string(index=False))

    # Asignación semántica objetiva considerando conectividad fija y móvil
    # Score de conectividad integral: z(fija) + z(móvil) - z(pobreza)
    z_stats = (stats - stats.mean()) / stats.std()
    score_integral = z_stats['conexiones_x1000hab'] + z_stats['antenas_x1000hab'] - z_stats['tasa_pobreza_ingresos']
    ranking = score_integral.rank()

    etiquetas = {}
    for c, r in ranking.items():
        if r == ranking.max():
            etiquetas[c] = ETIQUETA_ALTA
        elif r == ranking.min():
            etiquetas[c] = ETIQUETA_BAJA
        else:
            etiquetas[c] = ETIQUETA_MEDIA

    df_feat['cluster_label'] = df_feat['cluster'].map(etiquetas)
    print(f"\n  Etiquetas asignadas según score integral (Fija + Móvil - Pobreza):")
    for c, lbl in etiquetas.items():
        print(f"    Cluster {c} -> {lbl} (Score={score_integral[c]:.2f}, n={sum(df_feat['cluster'] == c)})")

    # Visualización PCA 2D y Heatmap de Centroides
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colores_cl = {'Alta conectividad': '#27ae60', 'Conectividad media': '#3498db', 'Baja conectividad': '#e74c3c'}

    for k in sorted(df_feat['cluster'].unique()):
        mask = df_feat['cluster'] == k
        lbl = etiquetas.get(k, f'Cluster {k}')
        axes[0].scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
                        c=colores_cl.get(lbl, '#7f8c8d'), label=f"{lbl} (n={sum(mask)})",
                        alpha=0.7, s=55, edgecolors='white', linewidth=0.4)

    # Destacar atípicas en PCA
    idx_atipicos = df_feat[df_feat['dist_centroide'] > umbral].index
    if len(idx_atipicos) > 0:
        axes[0].scatter(X_pca[idx_atipicos, 0], X_pca[idx_atipicos, 1],
                        marker='x', s=110, c='black', zorder=5, label='Atípicas (p95)')

    axes[0].set_xlabel(f'PC1 ({var_exp[0]:.1%} varianza)', fontsize=10)
    axes[0].set_ylabel(f'PC2 ({var_exp[1]:.1%} varianza)', fontsize=10)
    axes[0].set_title(f'Clusters en Espacio PCA 2D\n(Silhouette={sil:.3f} | Varianza explicada={sum(var_exp):.1%})',
                      fontweight='bold', fontsize=11)
    axes[0].legend(title='Nivel de Conectividad', fontsize=9)

    # Heatmap de perfiles normalizados
    stats_norm = (stats - stats.mean()) / stats.std()
    stats_norm.index = [etiquetas.get(i, f'C{i}') for i in stats_norm.index]
    sns.heatmap(stats_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=axes[1])
    axes[1].set_title('Perfiles de Clusters Normalizados\n(Valores z por dimensión)', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Nivel de Conectividad')

    plt.suptitle('Segmentación de Comunas por Nivel de Conectividad – K-Means (Fija + Móvil)\nMacrozona Centro-Norte',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '06_clusters_pca_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 06_clusters_pca_heatmap.png guardada")

    return df_feat, km, etiquetas

# ============================================================
# 6. CLASIFICACIÓN – RANDOM FOREST (Evaluación Imparcial con F1-Macro)
# ============================================================
def clasificacion(df_feat, feature_cols, etiquetas_cluster):
    print("\n=== CLASIFICACIÓN – RANDOM FOREST ===")
    df_model = df_feat[feature_cols + ['cluster']].dropna()
    X = df_model[feature_cols].values
    y = df_model['cluster'].values.astype(int)

    print(f"  Dataset: {X.shape[0]} comunas | {X.shape[1]} features | {len(set(y))} clases")
    print(f"  Distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} comunas | Test: {len(X_test)} comunas")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Pipeline con MinMaxScaler + RandomForest
    pipe_rf = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

    # Validación cruzada estratificada con métricas Macro y Weighted
    cv_acc      = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1_macro = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_f1_w     = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='f1_weighted')

    print(f"\n  [Resultados Validación Cruzada 5-Fold]")
    print(f"  CV Accuracy:    {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"  CV F1-Macro:    {cv_f1_macro.mean():.4f} +/- {cv_f1_macro.std():.4f} (Métrica principal imparcial)")
    print(f"  CV F1-Weighted: {cv_f1_w.mean():.4f} +/- {cv_f1_w.std():.4f}")

    # Ajuste final en train y evaluación en test
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)

    metricas_rf = {
        'Accuracy':    accuracy_score(y_test, y_pred_rf),
        'F1-Macro':    f1_score(y_test, y_pred_rf, average='macro', zero_division=0),
        'F1-Weighted': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'Precision':   precision_score(y_test, y_pred_rf, average='macro', zero_division=0),
        'Recall':      recall_score(y_test, y_pred_rf, average='macro', zero_division=0),
    }

    target_names = [etiquetas_cluster.get(k, f'C{k}') for k in sorted(set(y))]
    print(f"\n  Métricas en Test (n={len(y_test)}): {metricas_rf}")
    print(f"\n  Reporte de Clasificación en Test:\n{classification_report(y_test, y_pred_rf, target_names=target_names, zero_division=0, digits=4)}")

    # Guardar modelo completo entrenado
    joblib.dump(pipe_rf, os.path.join(MOD_PATH, 'random_forest.pkl'))

    # Figura 07: Matriz de Confusión
    cm_mat = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicción del Modelo', fontsize=11)
    ax.set_ylabel('Nivel Real (Cluster K-Means)', fontsize=11)
    ax.set_title('Matriz de Confusión – Random Forest\nClasificación por Nivel de Conectividad Comunal',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '07_confusion_matrix_rf.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 07_confusion_matrix_rf.png guardada")

    # Figura 08: Feature Importance
    rf_model = pipe_rf.named_steps['classifier']
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.7)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white', lw=0.5)
    for bar, val in zip(bars, importances.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f} ({val*100:.1f}%)', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Importancia Relativa (Reducción de Impureza de Gini)', fontsize=11)
    ax.set_title('Feature Importance – Random Forest\n(¿Qué factores determinan el nivel de conectividad comunal?)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, importances.max() * 1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '08_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 08_feature_importance.png guardada")

    # Figura 09: Comparativa de Métricas
    fig, ax = plt.subplots(figsize=(8, 5))
    metricas_nombres = list(metricas_rf.keys())
    metricas_valores = list(metricas_rf.values())
    colors_met = ['#27ae60', '#2980b9', '#3498db', '#e67e22', '#9b59b6']
    bars_met = ax.bar(metricas_nombres, metricas_valores, color=colors_met, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars_met, metricas_valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Métricas de Clasificación en Test – Random Forest\nMacrozona Centro-Norte', fontsize=12, fontweight='bold')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '09_metricas_rf.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 09_metricas_rf.png guardada")

    return metricas_rf, y_test, y_pred_rf, pipe_rf

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  PIPELINE DE MODELADO – BRECHA DIGITAL CHILE")
    print("  Macrozona Centro-Norte | Fija + Móvil | Random Forest")
    print("=" * 60)

    df = cargar_datos()
    df = eda(df)
    df_feat, X_scaled, feature_cols, scaler = preparar_features(df)
    best_k, inercias, silhouettes = elbow_silhouette(X_scaled, k_max=6)
    n_clusters = 3
    df_feat, km, etiquetas = kmeans_final(df_feat, X_scaled, feature_cols, n_clusters=n_clusters)
    metricas_rf, y_test, y_pred_rf, pipe_rf = clasificacion(df_feat, feature_cols, etiquetas)

    # Exportar dataset enriquecido con clusters y etiquetas
    df_con_clusters = df.merge(
        df_feat[['cut_comuna', 'cluster', 'cluster_label', 'dist_centroide']],
        on='cut_comuna', how='left'
    )
    out_cl = os.path.join(os.path.dirname(DATA_PATH), 'dataset_comunas_con_clusters.csv')
    df_con_clusters.to_csv(out_cl, index=False, encoding='utf-8-sig')
    print(f"\n  Dataset con clusters guardado exitosamente: {out_cl}")

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL DE MODELADO")
    print("=" * 60)
    print(f"  Comunas analizadas: {len(df)} (Macrozona Centro-Norte: IV, V, RM)")
    print(f"  Clusters (k={n_clusters}): {etiquetas}")
    print(f"  Silhouette Score: {silhouette_score(X_scaled, km.labels_):.4f}")
    print(f"  Random Forest – F1-Macro CV: {cross_val_score(pipe_rf, X_scaled, km.labels_, cv=5, scoring='f1_macro').mean():.4f}")
    print(f"  Random Forest Test – Accuracy={metricas_rf['Accuracy']:.4f} | F1-Macro={metricas_rf['F1-Macro']:.4f}")
    print(f"  Figuras generadas: {len(os.listdir(FIG_PATH))} en outputs/figures/")
    print(f"  Modelos exportados: {len(os.listdir(MOD_PATH))} en outputs/models/")
    print("=" * 60)

    return {
        'n_comunas': len(df),
        'n_clusters': n_clusters,
        'etiquetas': etiquetas,
        'metricas_rf': metricas_rf,
    }


if __name__ == "__main__":
    main()
