import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from limpiar_datos import limpiar_datos
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv("8. baseregiones.csv", encoding='UTF-8', sep=',')
limpiar_datos(df)
df.columns = df.columns.str.replace('"', '')

df_filtrado = df[
    (df['REGION'] == 'METROPOLITANA DE SANTIAGO')
].copy()

df_filtrado['TASA_REGULARIDAD'] = df_filtrado['RRAA_REGULAR'] / (df_filtrado['RRAA_TOTAL'] + 0.00000000001)
df_filtrado['ES_REGULAR'] = (df_filtrado['TASA_REGULARIDAD'] >= 0.5).astype(int)

rangos_edad = df_filtrado['EDAD'].value_counts().index
for rango in rangos_edad:
    df_filtrado[f'EDAD_{rango}'] = (df_filtrado['EDAD'] == rango).astype(int)
    
paises_top = df_filtrado['PAIS'].value_counts().head(5).index
for pais in paises_top:
    df_filtrado[f'PAIS_{pais}'] = (df_filtrado['PAIS'] == pais).astype(int)

variables_base = ['ANO ESTIMACION', 'CENSO AJUSTADO', 'SEXO'] 
variables_edad = [f'EDAD_{rango}' for rango in rangos_edad]
variables_paises = [f'PAIS_{pais}' for pais in paises_top]

todas_variables = variables_base + variables_edad + variables_paises

X = df_filtrado[todas_variables].dropna()
Y = df_filtrado.loc[X.index, 'ES_REGULAR']  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

print(f"\n📊 BALANCE DE CLASES ORIGINAL:")
print(f"   • Entrenamiento: {Counter(y_train)}")

print(f"\n🔄 Aplicando SMOTE para balancear clases...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"📊 DESPUÉS DE SMOTE:")
print(f"   • Entrenamiento balanceado: {Counter(y_train_sm)}")

logreg = LogisticRegression(random_state=16, max_iter=100000)
logreg.fit(X_train_sm, y_train_sm)

y_pred = logreg.predict(X_test)

# MATRIZ DE CONFUSIÓN
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\n🎯 MATRIZ DE CONFUSIÓN (CON SMOTE):")
print(cnf_matrix)

# MÉTRICAS DE EVALUACIÓN
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\n📊 MÉTRICAS FINALES:")
print(f"   • Precisión: {accuracy:.3f} ({accuracy:.1%})")

print(f"\n📈 REPORTE DETALLADO (CON SMOTE):")
print(metrics.classification_report(y_test, y_pred, 
                                  target_names=['IRREGULAR', 'REGULAR']))