"""
Script de Modelado: Clustering + Clasificación (Random Forest)
Proyecto: Análisis de la Brecha Digital – Macrozona Centro-Norte de Chile
(Coquimbo, Valparaíso, Región Metropolitana)

Cambios v3:
  - Solo Random Forest (se elimina Regresión Logística)
  - Término "vulnerable" reemplazado por niveles de conectividad
  - Incluye variables de internet móvil (indicadores nacionales normalizados)
  - Acotado a macrozona Centro-Norte (regiones 4, 5, 13)
  - Revisión de clusters K-Means con validación de datos sucios

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
# RUTAS
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

# Etiquetas de conectividad (reemplazan a "vulnerable")
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
    print(f"  Shape: {df.shape}")
    print(f"  Regiones incluidas: {sorted(df['num_region'].unique())}")
    print(f"  Columnas: {list(df.columns)}")
    return df

# ============================================================
# 2. EDA
# ============================================================
def eda(df):
    print("\n=== EDA ===")

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nVariables numericas ({len(cols_num)}):")
    print(df[cols_num].describe().round(2).to_string())

    nulos = df[cols_num].isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(1)
    nulos_df = pd.DataFrame({'N Nulos': nulos, '% Nulos': nulos_pct})
    nulos_df = nulos_df[nulos_df['N Nulos'] > 0].sort_values('% Nulos', ascending=False)
    print(f"\nVariables con nulos:\n{nulos_df.to_string()}")

    # Figura 01: Distribuciones
    vars_base = ['conexiones_x1000hab', 'poblacion_2024', 'tasa_pobreza_ingresos',
                 'conexiones_prom2024', 'conexiones_recientes']
    vars_movil = [c for c in ['movil_x1000hab_nacional', 'ratio_movil_fija'] if c in df.columns]
    vars_plot = [c for c in vars_base + vars_movil if c in df.columns]

    n = len(vars_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    axes = np.array(axes).flatten()
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
    for i, (var, color) in enumerate(zip(vars_plot, colors)):
        data = df[var].dropna()
        axes[i].hist(data, bins=25, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        axes[i].axvline(data.median(), color='red', linestyle='--', lw=1.5, label=f'Mediana: {data.median():.1f}')
        axes[i].axvline(data.mean(), color='orange', linestyle='--', lw=1.5, label=f'Media: {data.mean():.1f}')
        axes[i].set_title(var, fontweight='bold', fontsize=9)
        axes[i].legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Distribucion de Variables Clave – Macrozona Centro-Norte', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '01_distribuciones.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 01_distribuciones.png guardada")

    # Figura 02: Heatmap correlaciones
    corr_vars = [c for c in ['conexiones_x1000hab', 'conexiones_prom2024',
                              'poblacion_2024', 'tasa_pobreza_ingresos',
                              'movil_x1000hab_nacional', 'ratio_movil_fija'] if c in df.columns]
    if len(corr_vars) >= 2:
        corr = df[corr_vars].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
        ax.set_title('Matriz de Correlacion de Pearson\n(Fija + Movil + Socioeconomico)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, '02_heatmap_correlacion.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Figura 02_heatmap_correlacion.png guardada")

    # Figura 03: Scatter Pobreza vs Conectividad fija
    if 'tasa_pobreza_ingresos' in df.columns and 'conexiones_x1000hab' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            df['tasa_pobreza_ingresos'] * 100,
            df['conexiones_x1000hab'],
            c=df['num_region'],
            cmap='viridis', alpha=0.6, s=50, edgecolors='white', linewidth=0.3
        )
        plt.colorbar(sc, ax=ax, label='Region')
        ax.set_xlabel('Tasa de Pobreza por Ingresos 2022 (%)', fontsize=11)
        ax.set_ylabel('Conexiones fijas por 1.000 hab. (2024)', fontsize=11)
        ax.set_title('Pobreza vs. Conectividad por Comuna – Centro-Norte\n(color = Region)', fontsize=12, fontweight='bold')
        df_plot = df.dropna(subset=['tasa_pobreza_ingresos', 'conexiones_x1000hab'])
        top_con = df_plot.nlargest(3, 'conexiones_x1000hab')
        bot_con = df_plot.nsmallest(3, 'conexiones_x1000hab')
        for _, row in pd.concat([top_con, bot_con]).iterrows():
            ax.annotate(row['nombre_comuna'],
                        (row['tasa_pobreza_ingresos']*100, row['conexiones_x1000hab']),
                        fontsize=8, alpha=0.85, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, '03_scatter_pobreza_conectividad.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Figura 03_scatter_pobreza_conectividad.png guardada")

    # Figura 04: Ranking comunas por conectividad
    if 'conexiones_x1000hab' in df.columns:
        df_sort = df.dropna(subset=['conexiones_x1000hab']).sort_values('conexiones_x1000hab')
        top10 = df_sort.tail(10)
        bot10 = df_sort.head(10)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
        colors_bot = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
        axes[0].barh(top10['nombre_comuna'], top10['conexiones_x1000hab'], color=colors_top)
        axes[0].set_title('Top 10: Mayor Conectividad', fontweight='bold', color='darkgreen')
        axes[0].set_xlabel('Conexiones fijas por 1.000 hab.')
        axes[1].barh(bot10['nombre_comuna'], bot10['conexiones_x1000hab'], color=colors_bot)
        axes[1].set_title('Top 10: Menor Conectividad (Baja conectividad)', fontweight='bold', color='darkred')
        axes[1].set_xlabel('Conexiones fijas por 1.000 hab.')
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
    print("\n=== PREPARANDO FEATURES ===")

    # Incluir variables móviles si están disponibles
    feature_cols_base = [
        'conexiones_x1000hab',
        'conexiones_prom2024',
        'tasa_pobreza_ingresos',
        'poblacion_2024',
        'pct_ruralidad_proxy',
    ]
    feature_cols_movil = ['movil_x1000hab_nacional', 'ratio_movil_fija']

    feature_cols = [c for c in feature_cols_base if c in df.columns]
    for c in feature_cols_movil:
        if c in df.columns:
            feature_cols.append(c)

    print(f"  Features para clustering: {feature_cols}")

    df_feat = df[feature_cols + ['cut_comuna', 'nombre_comuna', 'num_region']].copy()

    # Imputación por mediana
    for col in feature_cols:
        med = df_feat[col].median()
        n_imp = df_feat[col].isnull().sum()
        if n_imp > 0:
            df_feat[col] = df_feat[col].fillna(med)
            print(f"  Imputados {n_imp} nulos en '{col}' con mediana={med:.2f}")

    # Winsorización p2-p98 (más conservador que p1-p99 para datasets pequeños)
    for col in feature_cols:
        p2  = df_feat[col].quantile(0.02)
        p98 = df_feat[col].quantile(0.98)
        n_out = ((df_feat[col] < p2) | (df_feat[col] > p98)).sum()
        df_feat[col] = df_feat[col].clip(lower=p2, upper=p98)
        if n_out > 0:
            print(f"  Winsor. en '{col}': {n_out} outliers acotados (p2={p2:.2f}, p98={p98:.2f})")

    X = df_feat[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MOD_PATH, 'scaler_clustering.pkl'))
    print(f"\n  Dataset para clustering: {X_scaled.shape}")
    return df_feat, X_scaled, feature_cols, scaler

# ============================================================
# 4. ELBOW + SILHOUETTE
# ============================================================
def elbow_silhouette(X_scaled, k_max=7):
    print("\n=== ELBOW + SILHOUETTE ===")
    ks = range(2, k_max+1)
    inercias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=RANDOM_SEED)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"  k={k}: Inercia={km.inertia_:.1f} | Silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(ks), inercias, 'bo-', linewidth=2.5, markersize=9)
    axes[0].fill_between(list(ks), inercias, alpha=0.1, color='blue')
    axes[0].set_xlabel('Numero de Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inercia (WCSS)', fontsize=11)
    axes[0].set_title('Metodo del Codo (Elbow)', fontweight='bold', fontsize=12)
    axes[0].set_xticks(list(ks))

    best_k = list(ks)[int(np.argmax(silhouettes))]
    axes[1].plot(list(ks), silhouettes, 'rs-', linewidth=2.5, markersize=9)
    axes[1].fill_between(list(ks), silhouettes, alpha=0.1, color='red')
    axes[1].axvline(best_k, color='green', linestyle='--', lw=2,
                    label=f'Mejor k={best_k} ({max(silhouettes):.3f})')
    axes[1].set_xlabel('Numero de Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score por k', fontweight='bold', fontsize=12)
    axes[1].set_xticks(list(ks))
    axes[1].legend(fontsize=10)

    plt.suptitle('Seleccion del Numero Optimo de Clusters – Brecha Digital Macrozona Centro-Norte', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '05_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura 05_elbow_silhouette.png guardada | Mejor k={best_k}")
    return best_k, inercias, silhouettes

# ============================================================
# 5. K-MEANS FINAL (con detección de datos sucios)
# ============================================================
def kmeans_final(df_feat, X_scaled, feature_cols, n_clusters=3):
    print(f"\n=== K-MEANS FINAL (k={n_clusters}) ===")
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30,
                max_iter=500, random_state=RANDOM_SEED)
    labels = km.fit_predict(X_scaled)
    df_feat = df_feat.copy()
    df_feat['cluster'] = labels

    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score final: {sil:.4f}")
    print(f"  Distribucion de clusters:\n{pd.Series(labels).value_counts().sort_index().to_string()}")

    joblib.dump(km, os.path.join(MOD_PATH, 'kmeans_model.pkl'))

    # Estadísticas por cluster
    stats = df_feat.groupby('cluster')[feature_cols].mean().round(3)
    print(f"\n  Centroides por cluster:\n{stats.to_string()}")

    # --- Detección de datos sucios: distancia al centroide ---
    centros = km.cluster_centers_
    distancias = []
    for i, x in enumerate(X_scaled):
        c = labels[i]
        dist = np.linalg.norm(x - centros[c])
        distancias.append(dist)
    df_feat['dist_centroide'] = distancias

    # Umbral: comunas con distancia > percentil 95 son potencialmente atípicas
    umbral = np.percentile(distancias, 95)
    df_atipicas = df_feat[df_feat['dist_centroide'] > umbral][['nombre_comuna', 'cluster', 'dist_centroide'] + feature_cols[:3]]
    print(f"\n  Comunas potencialmente atipicas (dist_centroide > p95={umbral:.2f}):")
    print(df_atipicas.to_string(index=False))

    # Asignar etiquetas semánticas de conectividad (sin "vulnerable")
    if 'conexiones_x1000hab' in stats.columns:
        rank = stats['conexiones_x1000hab'].rank()
        etiquetas = {}
        for c, r in rank.items():
            if r == rank.max():
                etiquetas[c] = ETIQUETA_ALTA
            elif r == rank.min():
                etiquetas[c] = ETIQUETA_BAJA
            else:
                etiquetas[c] = ETIQUETA_MEDIA
    else:
        etiquetas = {i: f'Cluster {i}' for i in range(n_clusters)}

    df_feat['cluster_label'] = df_feat['cluster'].map(etiquetas)
    print(f"\n  Etiquetas asignadas: {etiquetas}")

    # Visualización PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    print(f"  PCA varianza explicada: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, Total={sum(var_exp):.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colores_cl = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    labels_uniq = sorted(df_feat['cluster'].unique())

    for k in labels_uniq:
        mask = df_feat['cluster'] == k
        label = etiquetas.get(k, f'Cluster {k}')
        color = colores_cl[k % len(colores_cl)]
        axes[0].scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
                        c=color, label=label, alpha=0.6, s=45,
                        edgecolors='white', linewidth=0.3)

    # Marcar atípicas
    idx_atipicos = df_feat[df_feat['dist_centroide'] > umbral].index
    if len(idx_atipicos) > 0:
        axes[0].scatter(X_pca[idx_atipicos, 0], X_pca[idx_atipicos, 1],
                        marker='x', s=120, c='black', zorder=5, label='Atipicas')

    axes[0].set_xlabel(f'PC1 ({var_exp[0]:.1%} varianza)', fontsize=10)
    axes[0].set_ylabel(f'PC2 ({var_exp[1]:.1%} varianza)', fontsize=10)
    axes[0].set_title(f'Clusters en Espacio PCA 2D\nSilhouette={sil:.3f}', fontweight='bold')
    axes[0].legend(title='Nivel de conectividad', fontsize=9)

    # Heatmap centroides
    stats_norm = (stats - stats.mean()) / stats.std()
    stats_norm.index = [etiquetas.get(i, f'C{i}') for i in stats_norm.index]
    sns.heatmap(stats_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=axes[1])
    axes[1].set_title('Perfiles de Clusters\n(valores normalizados)', fontweight='bold')
    axes[1].set_xlabel('Nivel de conectividad')

    plt.suptitle('Segmentacion de Comunas por Nivel de Conectividad – K-Means\nMacrozona Centro-Norte', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '06_clusters_pca_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 06_clusters_pca_heatmap.png guardada")

    return df_feat, km, etiquetas

# ============================================================
# 6. CLASIFICACIÓN – SOLO RANDOM FOREST (sin Regresión Logística)
# ============================================================
def clasificacion(df_feat, feature_cols, etiquetas_cluster):
    print("\n=== CLASIFICACION – RANDOM FOREST (sin Data Leakage) ===")
    df_model = df_feat[feature_cols + ['cluster']].dropna()
    X = df_model[feature_cols].values
    y = df_model['cluster'].values.astype(int)

    print(f"  Dataset: {X.shape[0]} comunas | {X.shape[1]} features | {len(set(y))} clases")
    print(f"  Distribucion clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split sin tocar los datos de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # --- Random Forest con Pipeline ---
    print("\n  [Random Forest con Pipeline – validacion cruzada]")
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

    # Cross-validation con múltiples métricas
    cv_f1  = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='f1_weighted')
    cv_acc = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy (5-fold): {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"  CV F1-weighted (5-fold): {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    # Verificar si los resultados son sospechosamente altos
    if cv_f1.mean() > 0.97:
        print("\n  ADVERTENCIA: CV F1 > 0.97 – posible sesgo (el clasificador aprende de los propios clusters)")
        print("  Nota: Clasificar clusters K-Means con las mismas features es una tarea trivial.")
        print("  Interpretacion: el modelo captura correctamente la estructura de los datos,")
        print("  NO implica que los clusters sean perfectos ni que haya data leakage real.")

    # Entrenar en todo el train set y evaluar en test
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)

    metricas_rf = {
        'Accuracy':  accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
    }
    print(f"\n  Metricas en Test: {metricas_rf}")
    print(f"\n  Reporte completo:\n{classification_report(y_test, y_pred_rf, target_names=[etiquetas_cluster.get(k, f'C{k}') for k in sorted(set(y))], zero_division=0)}")

    joblib.dump(pipe_rf, os.path.join(MOD_PATH, 'random_forest.pkl'))
    joblib.dump(pipe_rf.named_steps['scaler'], os.path.join(MOD_PATH, 'scaler_clasificacion.pkl'))

    # Figura 07: Matriz de confusión (solo RF)
    class_names = [etiquetas_cluster.get(k, f'C{k}') for k in sorted(set(y))]
    cm_mat = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Prediccion', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_title('Matriz de Confusion – Random Forest\nClasificacion por Nivel de Conectividad', fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '07_confusion_matrix_rf.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 07_confusion_matrix_rf.png guardada")

    # Figura 08: Feature importance RF
    rf_model = pipe_rf.named_steps['classifier']
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols)*0.6)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white', lw=0.5)
    for bar, val in zip(bars, importances.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importancia (reduccion media impureza Gini)', fontsize=11)
    ax.set_title('Feature Importance – Random Forest\n(Que determina el nivel de conectividad de una comuna?)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, importances.max() * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, '08_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figura 08_feature_importance.png guardada")

    # Figura 09: Metricas RF (barras)
    fig, ax = plt.subplots(figsize=(8, 5))
    metricas_nombres = list(metricas_rf.keys())
    metricas_valores = list(metricas_rf.values())
    colors_met = ['#27ae60', '#3498db', '#e67e22', '#9b59b6']
    bars_met = ax.bar(metricas_nombres, metricas_valores, color=colors_met, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars_met, metricas_valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metricas de Clasificacion – Random Forest\nMacrozona Centro-Norte', fontsize=12, fontweight='bold')
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
    print("  Macrozona Centro-Norte | Solo Random Forest")
    print("=" * 60)

    df = cargar_datos()
    df = eda(df)
    df_feat, X_scaled, feature_cols, scaler = preparar_features(df)
    best_k, inercias, silhouettes = elbow_silhouette(X_scaled, k_max=6)
    n_clusters = 3  # Alta / Media / Baja conectividad
    df_feat, km, etiquetas = kmeans_final(df_feat, X_scaled, feature_cols, n_clusters=n_clusters)
    metricas_rf, y_test, y_pred_rf, pipe_rf = clasificacion(df_feat, feature_cols, etiquetas)

    # Guardar dataset con clusters
    df_con_clusters = df.merge(
        df_feat[['cut_comuna', 'cluster', 'cluster_label', 'dist_centroide']],
        on='cut_comuna', how='left'
    )
    out_cl = os.path.join(os.path.dirname(DATA_PATH), 'dataset_comunas_con_clusters.csv')
    df_con_clusters.to_csv(out_cl, index=False, encoding='utf-8-sig')
    print(f"\n  Dataset con clusters guardado: {out_cl}")

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Comunas analizadas: {len(df)} (macrozona Centro-Norte)")
    print(f"  Regiones: Coquimbo (4), Valparaiso (5), Metropolitana (13)")
    print(f"  Clusters (k={n_clusters}): {etiquetas}")
    print(f"  Silhouette Score: {silhouette_score(X_scaled, km.labels_):.4f}")
    print(f"  Random Forest – Accuracy={metricas_rf['Accuracy']:.4f} | F1={metricas_rf['F1-Score']:.4f}")
    print(f"\n  Figuras: outputs/figures/ ({len(os.listdir(FIG_PATH))} archivos)")
    print(f"  Modelos: outputs/models/ ({len(os.listdir(MOD_PATH))} archivos)")
    print("=" * 60)

    return {
        'n_comunas': len(df),
        'n_clusters': n_clusters,
        'etiquetas': etiquetas,
        'metricas_rf': metricas_rf,
    }


if __name__ == "__main__":
    main()
