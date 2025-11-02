import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from limpiar_datos import limpiar_datos

try:
    # Intentar varias codificaciones comunes si el archivo contiene caracteres especiales
    df_csv = None
    for _enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df_csv = pd.read_csv("8. baseregiones.csv", encoding=_enc)
            print(f"Archivo leído con encoding: {_enc}")
            break
        except UnicodeDecodeError:
            # probar siguiente encoding
            continue
    if df_csv is None:
        # Si ninguno funcionó, levantar el error original para que sea visible
        df_csv = pd.read_csv("8. baseregiones.csv")
    df_datos = df_csv.copy() # Crear una copia del DataFrame original para trabajar en ella
    limpiar_datos(df_datos)
    print(df_datos)

    # --- Creando el objetivo (Y) ---
    # Se usa .astype(int) para convertir los true y false en 1 y 0
    df_datos['ES_IRREGULAR'] = (df_datos['RRAA_IRREGULAR'] > 0).astype(int)
    print("\nNueva columna 'ES_IRREGULAR' creada (0=No, 1=Sí).")

    # --- 2. Definición de X e Y ---
    print("\n--- PREPARANDO EL MODELO DE IRREGULARIDAD ---")

    # Asegurar que existan columnas numéricas requeridas. Si no existen,
    # crearlas a partir de las columnas categóricas presentes.
    if 'PAIS_CODIGO' not in df_datos.columns and 'PAIS' in df_datos.columns:
        df_datos['PAIS_CODIGO'] = pd.factorize(df_datos['PAIS'])[0]
        print("Columna 'PAIS_CODIGO' creada a partir de 'PAIS' mediante factorize.")

    if 'EDAD_NUMERICA' not in df_datos.columns and 'EDAD' in df_datos.columns:
        df_datos['EDAD_NUMERICA'] = pd.factorize(df_datos['EDAD'])[0]
        print("Columna 'EDAD_NUMERICA' creada a partir de 'EDAD' mediante factorize.")

    #Se separa la X y la y
    X = df_datos[['PAIS_CODIGO', 'EDAD_NUMERICA', 'SEXO', 'AÑO ESTIMACION', 'CODREGEO']]
    y = df_datos['ES_IRREGULAR']
    

    # --- 4. Entrenamiento ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Aqui asignamos el modelo de prediccion y le decimos cuanto tiene que pensar el consejo de arboles.
    model =  RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)


    # --- 5. Evaluación ---
    # Se toma el entrenamiento y comienza la prediccion  

    y_pred = model.predict(X_test)
    precision = accuracy_score(y_test, y_pred) 

    print(f"\n¡¡¡PRECISIÓN GENERAL DEL MODELO: {precision * 100:.2f}% !!!")
    print("(Esta es la nota que saca el modelo al predecir el set de prueba)")
    print("\nReporte de Clasificación (0=No Irregular, 1=Sí Irregular):")
    print(classification_report(y_test, y_pred)) # Muestra la precision por filas, los Recall y el F1-score.

except FileNotFoundError:
    print("No se encontro el archivo CSV")
except pd.errors.EmptyDataError:
    print("El archivo se encuentra vacio")
