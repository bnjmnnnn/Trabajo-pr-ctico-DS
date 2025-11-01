import pandas as pd
from pathlib import Path
import sys
#Herramientas para el Machine Lerning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Importamos la misma función de limpieza que ya arreglamos
try:
    from limpiar_datos import limpiar_datos 
except ImportError:
    print("Error: No se pudo encontrar el archivo 'limpiar_datos.py'.")
    print("Asegúrate de que esté en la misma carpeta que RandomForest.py")
    sys.exit(1)

# --- 1. Carga de Datos (Robusta) ---
# Aqui se leen los datos del archivo baseregiones.csv
# Se aplica pathlib oara encontrar la ruta correcta del archivo, .resolve().parent
# asi no hay problemas encontrando el archivo siempre y cuando este en la misma carpeta que este codigo.
csv_file = "baseregiones.csv"
csv_file_path = str(Path(__file__).resolve().parent / csv_file)
df = None

print(f"Buscando archivo en: {csv_file_path}")
# Aqui se aplican distintos formatos de lectura por si llega a fallar una.
# se usa try...exept
try:
    df = pd.read_csv(csv_file_path, encoding="utf-8")
    print(f"Archivo leído con encoding=utf-8.")
except UnicodeDecodeError:
    df = pd.read_csv(csv_file_path, encoding="latin-1")
    print(f"Archivo leído con encoding=latin-1.")
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en la ruta '{csv_file_path}'.")
    sys.exit(1)
except Exception as e:
    print(f"Ocurrió un error inesperado al leer el archivo: {e}")
    sys.exit(1)

# --- 2. Limpieza y Creación del Objetivo ---
# Esto permite que solo se ejecute siempre y cuando los datos cargen exitosamente
if df is not None:
    
    # Aplicar la limpieza
    print("\nIniciando limpieza de datos...")
    df_limpio = limpiar_datos(df.copy()) # El .copy() es para crear una copia del DataFrame
    print("--- Limpieza de datos finalizada ---")

    
    # --- Creando el objetivo (Y) ---
    # Se usa .astype(int) para convertir los true y false en 1 y 0
    df_limpio['ES_IRREGULAR'] = (df_limpio['RRAA_IRREGULAR'] > 0).astype(int)
    print("\nNueva columna 'ES_IRREGULAR' creada (0=No, 1=Sí).")

    
    # --- 3. Definición de X e Y ---
    print("\n--- Preparando el modelo de IRREGULARIDAD ---")
    #Se separa la x de la y
    caracteristicas = ['PAIS', 'EDAD', 'SEXO', 'AÑO ESTIMACION', 'REGION']
    objetivo = 'ES_IRREGULAR' 
    
    # --- Guardar las opciones válidas ---
    # Esto es para mostrar por pantalla qué escribir.
    
    opciones_validas = {
        'PAIS': sorted(df_limpio['PAIS'].unique()),
        'EDAD': sorted(df_limpio['EDAD'].unique()),
        'SEXO': sorted(df_limpio['SEXO'].unique()),
        'REGION': sorted(df_limpio['REGION'].unique())
    }
    #Creamos un Dataframe para X e Y
    X = df_limpio[caracteristicas]
    y = df_limpio[objetivo]

    # Convertir X a version numerica, 0 y 1 tambien
    X_numerico = pd.get_dummies(X, dtype=int)
    
    # --- Guardar el "molde" de las columnas ---
    # Esto es importante. Guardamos la lista exacta de columnas con las que se entrenó osea los dummies.
    # Principalmente para comprobar el correcto ingreso de los datos.
    columnas_modelo = X_numerico.columns
    
    print(f"Características (X) convertidas a {X_numerico.shape[1]} columnas numéricas.")
    
    
    # --- 4. Entrenamiento ---
    # Divide los datos para su analisis.
    # stratify=y Es para corregir proporciones, muchos 1 o muchos 0.
    # Esto mejora la exactitud del modelo
    print("\nDividiendo datos y entrenando el modelo...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_numerico, y, test_size=0.2, stratify=y, random_state=42
    )
    #Aqui le decimos cuanto tiene que pensar el consejo de arboles.
    modelo = RandomForestClassifier(random_state=42, n_estimators=100)
    modelo.fit(X_train, y_train)
    print("¡Modelo entrenado!")

    
    # --- 5. Evaluación ---
    # Se toma el entrenamiento y comienza la prediccion
    print("\n--- Evaluación del Modelo de IRREGULARIDAD ---")
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred) # calcula la precision
    
    print(f"\n¡¡¡PRECISIÓN GENERAL DEL MODELO: {precision * 100:.2f}% !!!")
    print("(Esta es la nota que saca el modelo al predecir el set de prueba)")
    print("\nReporte de Clasificación (0=No Irregular, 1=Sí Irregular):")
    print(classification_report(y_test, y_pred)) # Muestra la precision por filas, los Recall y el F1-score.


    print("\n--- Listo para hacer predicciones ---")
    print("Escribe 'salir' en cualquier momento para terminar.")
    
    while True:
        print("\n--- Ingrese los datos del perfil a predecir ---")
        
        # Usaremos esta variable para saber si debemos salir del bucle principal
        salir_programa = False
        
        # --- PAIS ---
        # .srtip() limpia espacios en blanco
        # .lower() convierte a minusculas, para que acepte tanto salir como Salir.
        while True:
            print(f"\nOpciones de PAIS: {opciones_validas['PAIS']}")
            in_pais = input("Ingrese PAIS: ").strip()
            
            if in_pais.lower() == 'salir':
                salir_programa = True
                break 
            
            if in_pais in opciones_validas['PAIS']:
                break 
            else:
                print(f"¡Error! '{in_pais}' no es una opción válida. Intente de nuevo o ingrese 'salir' para terminar.")
                
        
        if salir_programa: break 

        # --- EDAD ---
        
        while True:
            print(f"\nOpciones de EDAD: {opciones_validas['EDAD']}")
            in_edad = input("Ingrese EDAD: ").strip()
            
            if in_edad.lower() == 'salir':
                salir_programa = True
                break 
            
            if in_edad in opciones_validas['EDAD']:
                break 
            else:
                print(f"¡Error! '{in_edad}' no es una opción válida. Intente de nuevo o ingrese 'salir' para terminar.")
                
        
        if salir_programa: break

        # --- SEXO ---
       
        while True:
            print(f"\nOpciones de SEXO: {opciones_validas['SEXO']}")
            in_sexo = input("Ingrese SEXO: ").strip()
            
            if in_sexo.lower() == 'salir':
                salir_programa = True
                break
            
            if in_sexo in opciones_validas['SEXO']:
                break 
            else:
                print(f"¡Error! '{in_sexo}' no es una opción válida. Intente de nuevo o ingrese 'salir' para terminar.")
        
        if salir_programa: break

        # --- AÑO ---
        while True:
            in_ano = input("\nIngrese AÑO ESTIMACION Solo años entre 2018 y 2023: ").strip()
            
            if in_ano.lower() == 'salir':
                salir_programa = True
                break
            try:
               ano_num = int(in_ano)
               if 2018 <= ano_num <= 2023:
                  break
               else:
                     print(f"¡Error! '{in_ano}' no está entre 2018 y 2023. Intente de nuevo o ingrese 'salir' para terminar.")

            except ValueError:
                print(f"¡Error! '{in_ano}' no es un número válido. Intente de nuevo o ingrese 'salir' para terminar.")
        
        if salir_programa: break

        # --- REGION ---
        while True:
            print(f"\nOpciones de REGION: {opciones_validas['REGION']}")
            in_region = input("Ingrese REGION: ").strip()
            
            if in_region.lower() == 'salir':
                salir_programa = True
                break
            
            if in_region in opciones_validas['REGION']:
                break 
            else:
                print(f"¡Error! '{in_region}' no es una opción válida. Intente de nuevo.")
        
        if salir_programa: break

        # --- Preparar los datos del usuario ---
        try:
            # Crear un DataFrame de 1 fila con la entrada
            datos_usuario = {
                'PAIS': [in_pais],
                'EDAD': [in_edad],
                'SEXO': [in_sexo],
                'AÑO ESTIMACION': [int(in_ano)], # Convertir año a número
                'REGION': [in_region]
            }
            df_usuario = pd.DataFrame(datos_usuario)

            # Convertir a dummies
            df_usuario_dummies = pd.get_dummies(df_usuario, dtype=int)
            
            # Reindexar con las columnas del modelo
            # .reindex() Toma un dataframe y lo reordena basandose en otro
            df_para_predecir = df_usuario_dummies.reindex(columns=columnas_modelo, fill_value=0)

            # --- Hacer la Predicción ---
            # Se basa en una logica de votos, algo asi como una democracia.
            prediccion_clase = modelo.predict(df_para_predecir)
            prediccion_prob = modelo.predict_proba(df_para_predecir)
            
            # --- Mostrar Resultado ---
            clase_predicha = prediccion_clase[0]
            confianza = prediccion_prob[0][clase_predicha] * 100

            print("\n================== RESULTADO ==================")
            if clase_predicha == 1:
                print(f"  Segun los datos es probable que sea IRREGULAR (Clase 1)")
            else:
                print(f"  Segun los datos es probable que sea NO IRREGULAR (Clase 0)")
            
            print(f"  Confianza del modelo: {confianza:.2f}%")
            print("=============================================\n")

        except Exception as e:
            print(f"\n¡Error en la predicción! Revisa los datos.")
            print(f"Detalle del error: {e}")
            
        # --- Preguntar si desea continuar ---
        print("\n" + "-"*45)
        continuar = input("¿Quieres seguir? (s/n): ").strip().lower()
        if continuar != 's':
            print("Saliendo del programa... ¡Chao!")
            break
            
else:
    print("\nNo se pudo cargar el DataFrame. No se puede continuar.")

    print("\nNo se pudo cargar el DataFrame. No se puede continuar.")
