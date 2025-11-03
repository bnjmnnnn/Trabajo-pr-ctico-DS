import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from limpiar_datos import limpiar_datos
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df_csv = pd.read_csv("8. baseregiones.csv", encoding='UTF-8', sep=',')
df_datos = df_csv.copy()
limpiar_datos(df_datos)
df_datos.columns = df_datos.columns.str.replace('"', '')

# SISTEMA DE FILTROS
print("\n" + "="*60)
print("SISTEMA DE FILTROS PARA ANÁLISIS")
print("="*60)

# Preguntar si quiere filtrar
filtrar = input("\n¿Desea filtrar los datos? (s/n): ").strip().lower()

if filtrar == 's':
    # Mostrar opciones de países
    paises_disponibles = sorted(df_datos['PAIS'].unique())
    print(f"\nPaíses disponibles ({len(paises_disponibles)}):")
    for i, pais in enumerate(paises_disponibles, 1):
        print(f"  {i}. {pais}")
    
    filtrar_pais = input("\n¿Filtrar por PAÍS? (s/n): ").strip().lower()
    if filtrar_pais == 's':
        pais_seleccionado = input("Ingrese el nombre del PAÍS: ").strip().upper()
        if pais_seleccionado in paises_disponibles:
            df_datos = df_datos[df_datos['PAIS'] == pais_seleccionado].copy()
            print(f"Filtrado por país: {pais_seleccionado}")
        else:
            print(f"País '{pais_seleccionado}' no encontrado. Usando todos los países.")
    
    # Mostrar opciones de regiones (después del filtro de país si se aplicó)
    regiones_disponibles = sorted(df_datos['REGION'].unique())
    print(f"\nRegiones disponibles ({len(regiones_disponibles)}):")
    for i, region in enumerate(regiones_disponibles, 1):
        print(f"  {i}. {region}")
    
    filtrar_region = input("\n¿Filtrar por REGIÓN? (s/n): ").strip().lower()
    if filtrar_region == 's':
        region_seleccionada = input("Ingrese el nombre de la REGIÓN: ").strip().upper()
        if region_seleccionada in regiones_disponibles:
            df_datos = df_datos[df_datos['REGION'] == region_seleccionada].copy()
            print(f"Filtrado por región: {region_seleccionada}")
        else:
            print(f"Región '{region_seleccionada}' no encontrada. Usando todas las regiones.")
    
    if len(df_datos) < 100:
        print(f"ADVERTENCIA: Pocos datos ({len(df_datos)} registros). El modelo puede no ser preciso.")
else:
    print("Usando todos los datos sin filtros")

print("="*60)

df_filtrado = df_datos.copy()

# Clasificar como IRREGULAR si más del 50% de la migración es irregular
df_filtrado['TASA_IRREGULARIDAD'] = df_filtrado['RRAA_IRREGULAR'] / (df_filtrado['RRAA_TOTAL'] + 1e-10)
df_filtrado['TIPO_MIGRACION'] = (df_filtrado['TASA_IRREGULARIDAD'] > 0.5).astype(int)  # 1=IRREGULAR, 0=REGULAR


X = df_filtrado[['AÑO ESTIMACION', 'CENSO AJUSTADO', 'SEXO', 'EDAD_NUMERICA', 'PAIS_CODIGO']].dropna()
Y = df_filtrado.loc[X.index, 'TIPO_MIGRACION']

# Mostrar distribución real de personas (sumando las columnas)
total_regular = df_filtrado.loc[X.index, 'RRAA_REGULAR'].sum()
total_irregular = df_filtrado.loc[X.index, 'RRAA_IRREGULAR'].sum()
total_personas = total_regular + total_irregular

print(f"\n" + "="*60)
print("� CLASIFICACIÓN DE MIGRACIÓN")
print("="*60)
print(f"\nCriterio: IRREGULAR si >50% del RRAA_TOTAL es irregular\n")
print(f"📈 DISTRIBUCIÓN DE PERSONAS:")
print(f"   • REGULAR:   {total_regular:>10,.0f} personas ({total_regular/total_personas:.1%})")
print(f"   • IRREGULAR: {total_irregular:>10,.0f} personas ({total_irregular/total_personas:.1%})")
print(f"   • TOTAL:     {total_personas:>10,.0f} personas")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

logreg = LogisticRegression(random_state=16, max_iter=100000, class_weight='balanced')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# MATRIZ DE CONFUSIÓN POR PERSONAS
print("\n" + "="*60)
print("🎯 MATRIZ DE CONFUSIÓN (PERSONAS REALES)")
print("="*60)

# Obtener datos del conjunto de prueba
X_test_df = df_filtrado.loc[X_test.index]

# Cuando el modelo predice REGULAR (y_pred=0):
# - Las personas REGULARES de esas filas son VN (bien clasificadas)
# - Las personas IRREGULARES de esas filas son FN (no fueron detectadas)
regular_pred_mask = (y_pred == 0)
vn_personas = X_test_df.loc[regular_pred_mask, 'RRAA_REGULAR'].sum()
fn_personas = X_test_df.loc[regular_pred_mask, 'RRAA_IRREGULAR'].sum()

# Cuando el modelo predice IRREGULAR (y_pred=1):
# - Las personas IRREGULARES de esas filas son VP (bien clasificadas)
# - Las personas REGULARES de esas filas son FP (mal clasificadas)
irregular_pred_mask = (y_pred == 1)
vp_personas = X_test_df.loc[irregular_pred_mask, 'RRAA_IRREGULAR'].sum()
fp_personas = X_test_df.loc[irregular_pred_mask, 'RRAA_REGULAR'].sum()

print("\n                 Predicho")
print("                REGULAR      IRREGULAR")
print(f"Real REGULAR    {vn_personas:>10,.0f}  {fp_personas:>13,.0f}")
print(f"     IRREGULAR  {fn_personas:>10,.0f}  {vp_personas:>13,.0f}")

total_personas_test = vn_personas + fp_personas + fn_personas + vp_personas
total_regular_real = vn_personas + fp_personas
total_irregular_real = fn_personas + vp_personas

print(f"\n📊 Valoress:")
print(f"✅ Verdaderos positivos (REGULAR bien clasificado):   {vn_personas:>10,.0f} ({vn_personas/total_personas_test:.1%})")
print(f"❌ Falsos negativos (predijo IRREGULAR, era REGULAR): {fp_personas:>10,.0f} ({fp_personas/total_personas_test:.1%})")
print(f"❌ Falsos positivos (predijo REGULAR, era IRREGULAR): {fn_personas:>10,.0f} ({fn_personas/total_personas_test:.1%})")
print(f"✅ Verdaderos negativos (IRREGULAR bien clasificado): {vp_personas:>10,.0f} ({vp_personas/total_personas_test:.1%})")

print(f"\n📊 MÉTRICAS POR CLASE:")
recall_regular = vn_personas / total_regular_real if total_regular_real > 0 else 0
recall_irregular = vp_personas / total_irregular_real if total_irregular_real > 0 else 0
print(f"   • Recall REGULAR:    {recall_regular:.3f} ({recall_regular:.1%}) - detectó {vn_personas:,.0f} de {total_regular_real:,.0f}")
print(f"   • Recall IRREGULAR:  {recall_irregular:.3f} ({recall_irregular:.1%}) - detectó {vp_personas:,.0f} de {total_irregular_real:,.0f}")

accuracy_personas = (vn_personas + vp_personas) / total_personas_test
print(f"\n📊 PRECISIÓN (Accuracy): {accuracy_personas:.3f} ({accuracy_personas:.1%})")
print("="*60)

# VISUALIZACIÓN DE LA MATRIZ DE CONFUSIÓN
print("\n📊 Generando visualización de la matriz de confusión...")

# Crear matriz de confusión para visualización
cm_personas = np.array([
    [vn_personas, fp_personas],
    [fn_personas, vp_personas]
])

# Crear figura
fig, ax = plt.subplots(figsize=(10, 8))

# Usar seaborn para crear heatmap (sin barra de color)
sns.heatmap(cm_personas, annot=True, fmt=',', cmap='Blues', 
            xticklabels=['REGULAR', 'IRREGULAR'],
            yticklabels=['REGULAR', 'IRREGULAR'],
            cbar=False,
            linewidths=2, linecolor='black',
            ax=ax)

# Personalizar el gráfico
ax.set_xlabel('Predicción del Modelo', fontsize=14, fontweight='bold')
ax.set_ylabel('Clase Real', fontsize=14, fontweight='bold')
ax.set_title(f'Matriz de Confusión - Personas Reales\nAccuracy: {accuracy_personas:.1%}', 
             fontsize=16, fontweight='bold', pad=20)

# Agregar texto con métricas adicionales
textstr = f'Total personas (test): {total_personas_test:,.0f}\n'
textstr += f'Recall REGULAR: {recall_regular:.1%}\n'
textstr += f'Recall IRREGULAR: {recall_irregular:.1%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.show()

print("✅ Visualización generada")
