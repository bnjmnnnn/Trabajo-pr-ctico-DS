import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from limpiar_datos import limpiar_datos

# Cargar y preparar datos
df = pd.read_csv("8. baseregiones.csv", encoding='latin-1', sep=',')
limpiar_datos(df)
df.columns = df.columns.str.replace('"', '')

# Filtrar por región
df_filtrado = df[(df['REGION'] == 'VALPARAÍSO')].copy()

# Preparar variables
df_filtrado['SEXO_NUMERIC'] = df_filtrado['SEXO'].map({'H': 1, 'M': 0})
edad_mapping = {
    '00 A 04': 2, '05 A 09': 7, '10 A 14': 12, '15 A 19': 17,
    '20 A 24': 22, '25 A 29': 27, '30 A 34': 32, '35 A 39': 37,
    '40 A 44': 42, '45 A 49': 47, '50 A 54': 52, '55 A 59': 57,
    '60 A 64': 62, '65 A 69': 67, '70 A 74': 72, '75 A 79': 77,
    '80 Y MAS': 85
}
df_filtrado['EDAD_NUMERIC'] = df_filtrado['EDAD'].map(edad_mapping).fillna(40)

# Variables países
paises_top = df_filtrado['PAIS'].value_counts().head(5).index
for pais in paises_top:
    df_filtrado[f'PAIS_{pais}'] = (df_filtrado['PAIS'] == pais).astype(int)

# Crear X e Y
variables_base = ['ANO ESTIMACION', 'CENSO AJUSTADO', 'SEXO_NUMERIC', 'EDAD_NUMERIC']
variables_paises = [f'PAIS_{pais}' for pais in paises_top]
X = df_filtrado[variables_base + variables_paises].dropna()
Y = df_filtrado.loc[X.index, 'RRAA_TOTAL']

# Entrenar modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, Y_train)
Y_pred = modelo.predict(X_test)

# Resultados
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f'📊 RESULTADOS - PREDICCIÓN DE MIGRACIÓN:')
print(f'🎯 RMSE: {rmse:.2f}')
print(f'📈 R²: {r2:.4f} ({r2:.1%})')