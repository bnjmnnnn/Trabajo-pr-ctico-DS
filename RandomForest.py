import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from limpiar_datos import limpiar_datos
# Ordenado con IA para fines explicativos.
# Imports para lo gráfico
import matplotlib.pyplot as plt
import seaborn as sns
import os 

try:
    # Intentar varias codificaciones comunes si el archivo contiene caracteres especiales
    df_csv = None
    for _enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df_csv = pd.read_csv("8. baseregiones.csv", encoding=_enc)
            print(f"Archivo leído con encoding: {_enc}")
            break
        except UnicodeDecodeError:
            continue
    if df_csv is None:
        # Si ninguno se digno a funcionar, levanta el error original y ahora sera visible
        df_csv = pd.read_csv("8. baseregiones.csv")
    df_datos = df_csv.copy() # Crea una copia del DataFrame original para trabajar en ella
    limpiar_datos(df_datos)
    print("Datos después de la limpieza (primeras 5 filas):")
    print(df_datos.head())


    # Creando el objetivo (Y)
    # Se usa .astype(int) para convertir los true y false en 1 y 0
    df_datos['ES_IRREGULAR'] = (df_datos['RRAA_IRREGULAR'] > 0).astype(int)
    print("\nNueva columna 'ES_IRREGULAR' creada (0=No, 1=Sí).")

    # Definición de X e Y
    print("\n--- PREPARANDO EL MODELO DE IRREGULARIDAD ---")

    # Asegurar que existan columnas numéricas requeridas.
    #.factorize() Les asigna un numero a cada dato que sea texto, para que pueda ser usado.
    if 'PAIS_CODIGO' not in df_datos.columns and 'PAIS' in df_datos.columns:
        df_datos['PAIS_CODIGO'] = pd.factorize(df_datos['PAIS'])[0]
        print("Columna 'PAIS_CODIGO' creada a partir de 'PAIS' mediante factorize.")

    if 'EDAD_NUMERICA' not in df_datos.columns and 'EDAD' in df_datos.columns:
        df_datos['EDAD_NUMERICA'] = pd.factorize(df_datos['EDAD'])[0]
        print("Columna 'EDAD_NUMERICA' creada a partir de 'EDAD' mediante factorize.")

    #Se separa la X y la y
    # Definimos las columnas de características en las que nos centraremos.
    features = ['PAIS_CODIGO', 'EDAD_NUMERICA', 'SEXO', 'AÑO ESTIMACION', 'CODREGEO']
    X = df_datos[features]
    y = df_datos['ES_IRREGULAR']
    

    # Entrenamiento, aqui esta intentando aprender
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Aqui asignamos el modelo de prediccion y le decimos cuanto tiene que pensar el consejo de arboles.
    model =  RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)


    # Evaluación
    # Se toma el entrenamiento y comienza la prediccion  

    y_pred = model.predict(X_test)
    precision = accuracy_score(y_test, y_pred) 

    print(f"\n¡¡¡PRECISIÓN GENERAL DEL MODELO: {precision * 100:.2f}% !!!")
    print("(Esta es la nota que saca el modelo al predecir el set de prueba)")
    print("\nReporte de Clasificación (0=No Irregular, 1=Sí Irregular):")
    print(classification_report(y_test, y_pred)) # Muestra la precision por filas, los Recall y el F1-score.

    # Generación de Gráficos
    # Esta es la sección para los gráficos
    print("\n--- GENERANDO GRÁFICOS DE RENDIMIENTO ---")

    # 1. Matriz de Confusión
    try:
        cm = confusion_matrix(y_test, y_pred)
        # plt.figure() limpia cualquier gráfico anterior
        plt.figure(figsize=(8, 6))
        #Aqui van las especificaciones de que datos debe tomar para el grafico.
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred. No Irregular (0)', 'Pred. Sí Irregular (1)'], 
                    yticklabels=['Real No Irregular (0)', 'Real Sí Irregular (1)'])
        plt.title('Matriz de Confusión') #El titulo we
        plt.ylabel('Valor Real')# El lado y
        plt.xlabel('Predicción del Modelo')# El lado x
        
        # Guardar el gráfico en un archivo
        ruta_matriz = "matriz_confusion.png"
        plt.savefig(ruta_matriz)
        plt.close() # Cierra la figura para liberar memoria
        print(f"Gráfico 'Matriz de Confusión' guardado en: {ruta_matriz}")

    except Exception as e:
        print(f"Error al generar la Matriz de Confusión: {e}")

    # 2. Importancia de Características
    try:
        importances = model.feature_importances_
        # Crear un DataFrame para facilitar el gráfico
        feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
        plt.title('Importancia de las Características (Features)')
        plt.xlabel('Nivel de Importancia')
        plt.ylabel('Característica')
        plt.tight_layout() # Ajusta el gráfico para que no se corten las etiquetas
        
        # Guardar el gráfico en un archivo
        ruta_features = "importancia_features.png"
        plt.savefig(ruta_features)
        plt.close()
        print(f"Gráfico 'Importancia de Características' guardado en: {ruta_features}")

    except Exception as e:
        print(f"Error al generar el gráfico de Importancia de Características: {e}")


    # Ejemplo de Predicción
    print("\n--- EJEMPLO DE PREDICCIÓN CON UN DATO REAL ---")
    
    # Tomar el primer dato del conjunto de prueba para mostrarlo
    ejemplo_features = X_test.iloc[0]
    ejemplo_prediccion = y_pred[0] # La predicción para la primera fila
    ejemplo_real = y_test.iloc[0] # El valor real para la primera fila

    print(f"\nDatos de entrada (Features) para el primer caso de prueba:")
    print(ejemplo_features)
    
    # Formatear la predicción y el valor real para que sean más legibles
    prediccion_texto = "SÍ es Irregular" if ejemplo_prediccion == 1 else "NO es Irregular"
    real_texto = "SÍ es Irregular" if ejemplo_real == 1 else "NO es Irregular"

    print(f"\n-> Predicción del Modelo: {ejemplo_prediccion} ({prediccion_texto})")
    print(f"-> Valor Real:           {ejemplo_real} ({real_texto})")

    if ejemplo_prediccion == ejemplo_real:
        print("\n¡El modelo acertó en este caso!")
    else:
        print("\nEl modelo falló en este caso.")


except FileNotFoundError:
    print("Error: No se encontró el archivo '8. baseregiones.csv'.")
except pd.errors.EmptyDataError:
    print("Error: El archivo '8. baseregiones.csv' se encuentra vacío.")
except ImportError:
    print("Error: Falta la biblioteca 'limpiar_datos'. Asegúrate de que el archivo 'limpiar_datos.py' esté en la misma carpeta.")
except Exception as e:
    print(f"Ha ocurrido un error inesperado: {e}")

