import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from limpiar_datos import limpiar_datos

# Cargar CSV y limpiar nombres de columnas
df = pd.read_csv("8. baseregiones.csv", encoding='latin-1', sep=',')
limpiar_datos(df)
df.columns = df.columns.str.replace('"', '')  # Remover comillas de los nombres de columnas

df_filtrado = df[
    (df['REGION'] == 'VALPARAÍSO')
].copy()

# Convertir variables categóricas a numéricas
df_filtrado['SEXO_NUMERIC'] = df_filtrado['SEXO'].map({'H': 1, 'M': 0})

# Mapear edad a valores numéricos
edad_mapping = {
    '00 A 04': 2, '05 A 09': 7, '10 A 14': 12, '15 A 19': 17,
    '20 A 24': 22, '25 A 29': 27, '30 A 34': 32, '35 A 39': 37,
    '40 A 44': 42, '45 A 49': 47, '50 A 54': 52, '55 A 59': 57,
    '60 A 64': 62, '65 A 69': 67, '70 A 74': 72, '75 A 79': 77,
    '80 Y MAS': 85
}
df_filtrado['EDAD_NUMERIC'] = df_filtrado['EDAD'].map(edad_mapping).fillna(40)

# One-hot encoding para países principales
paises_top = df_filtrado['PAIS'].value_counts().head(5).index
for pais in paises_top:
    df_filtrado[f'PAIS_{pais}'] = (df_filtrado['PAIS'] == pais).astype(int)

variables_base = ['ANO ESTIMACION', 'CENSO AJUSTADO', 'SEXO_NUMERIC', 'EDAD_NUMERIC']
variables_paises = [f'PAIS_{pais}' for pais in paises_top]

X = df_filtrado[variables_base + variables_paises].dropna()
Y = df_filtrado.loc[X.index, 'RRAA_TOTAL']

# 3. Dividir datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 4. Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, Y_train)

# 5. Hacer predicciones
Y_pred = modelo.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f'\n📊 RESULTADOS DE LA EVALUACIÓN - PREDICCIÓN DE MIGRACIÓN:\n')
print(f'🎯 RMSE: {rmse:.2f}')
print(f'   → En promedio, las predicciones se desvían en {rmse:.2f} personas del resultado real')

print(f'\n📈 R²: {r2:.2f} ({r2:.0%})')
print(f'   → El modelo explica el {r2:.0%} de la variabilidad en la migración total')