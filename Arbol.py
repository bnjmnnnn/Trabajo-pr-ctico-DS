import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Importamos la misma función de limpieza que ya arreglamos
try:
    from limpiar_datos import limpiar_datos 
except ImportError:
    print("Error: No se pudo encontrar el archivo 'limpiar_datos.py'.")
    print("Asegúrate de que esté en la misma carpeta que Codigobase2.py")
    sys.exit(1)

# --- 1. Carga de Datos (Robusta) ---
csv_file = "baseregiones.csv"
csv_file_path = str(Path(__file__).resolve().parent / csv_file)
df = None

print(f"Buscando archivo en: {csv_file_path}")

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
if df is not None:
    
    # Aplicar la limpieza
    print("\nIniciando limpieza de datos...")
    df_limpio = limpiar_datos(df.copy())
    print("--- Limpieza de datos finalizada ---")

    
    # --- Creando el objetivo (Y) ---
    df_limpio['ES_IRREGULAR'] = (df_limpio['RRAA_IRREGULAR'] > 0).astype(int)
    print("\nNueva columna 'ES_IRREGULAR' creada (0=No, 1=Sí).")

    
    # --- 3. Definición de X e Y ---
    print("\n--- Preparando el modelo de IRREGULARIDAD ---")
    
    caracteristicas = ['PAIS', 'EDAD', 'SEXO', 'AÑO ESTIMACION', 'REGION']
    objetivo = 'ES_IRREGULAR' 
    
    # --- ¡NUEVO! Guardar las opciones válidas ---
    # Esto es para ayudarte a saber qué escribir
    opciones_validas = {
        'PAIS': sorted(df_limpio['PAIS'].unique()),
        'EDAD': sorted(df_limpio['EDAD'].unique()),
        'SEXO': sorted(df_limpio['SEXO'].unique()),
        'REGION': sorted(df_limpio['REGION'].unique())
    }
    
    X = df_limpio[caracteristicas]
    y = df_limpio[objetivo]

    # Convertir X a números
    X_numerico = pd.get_dummies(X, dtype=int)
    
    # --- ¡NUEVO! Guardar el "molde" de las columnas ---
    # Esto es crucial. Guardamos la lista exacta de columnas con las que se entrenó.
    columnas_modelo = X_numerico.columns
    
    print(f"Características (X) convertidas a {X_numerico.shape[1]} columnas numéricas.")
    
    
    # --- 4. Entrenamiento ---
    print("\nDividiendo datos y entrenando el modelo...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_numerico, y, test_size=0.2, random_state=42
    )
    
    modelo = DecisionTreeClassifier(random_state=42, max_depth=10)
    modelo.fit(X_train, y_train)
    print("¡Modelo entrenado!")

    
    # --- 5. Evaluación (La hacemos una vez) ---
    print("\n--- Evaluación del Modelo de IRREGULARIDAD ---")
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"\n¡¡¡PRECISIÓN GENERAL DEL MODELO: {precision * 100:.2f}% !!!")
    print("(Esta es la nota que saca el modelo al predecir el set de prueba)")
    print("\nReporte de Clasificación (0=No Irregular, 1=Sí Irregular):")
    print(classification_report(y_test, y_pred))


    # --- 6. ¡NUEVO! Bucle Interactivo de Predicción ---
    print("\n--- Listo para hacer predicciones ---")
    print("Escribe 'salir' en cualquier momento para terminar.")
    
    while True:
        print("\n--- Ingrese los datos del perfil a predecir ---")
        
        # --- PAIS ---
        print(f"\nOpciones de PAIS: {opciones_validas['PAIS']}")
        in_pais = input("Ingrese PAIS: ").strip()
        if in_pais.lower() == 'salir': break
        
        # --- EDAD ---
        print(f"\nOpciones de EDAD: {opciones_validas['EDAD']}")
        in_edad = input("Ingrese EDAD: ").strip()
        if in_edad.lower() == 'salir': break
        
        # --- SEXO ---
        print(f"\nOpciones de SEXO: {opciones_validas['SEXO']}")
        in_sexo = input("Ingrese SEXO: ").strip()
        if in_sexo.lower() == 'salir': break

        # --- AÑO ---
        in_ano = input("\nIngrese AÑO ESTIMACION (ej: 2024): ").strip()
        if in_ano.lower() == 'salir': break

        # --- REGION ---
        print(f"\nOpciones de REGION: {opciones_validas['REGION']}")
        in_region = input("Ingrese REGION: ").strip()
        if in_region.lower() == 'salir': break

        # --- Preparar los datos del usuario ---
        try:
            # 1. Crear un DataFrame de 1 fila con la entrada
            datos_usuario = {
                'PAIS': [in_pais],
                'EDAD': [in_edad],
                'SEXO': [in_sexo],
                'AÑO ESTIMACION': [int(in_ano)], # Convertir año a número
                'REGION': [in_region]
            }
            df_usuario = pd.DataFrame(datos_usuario)

            # 2. Convertir a dummies (creará solo las columnas de la entrada)
            df_usuario_dummies = pd.get_dummies(df_usuario, dtype=int)
            
            # 3. EL TRUCO: Reindexar.
            #    Crea un DataFrame de 1 fila con TODAS las columnas del modelo (lleno de 0s)
            #    y luego "pega" las columnas del usuario (los 1s) donde correspondan.
            df_para_predecir = df_usuario_dummies.reindex(columns=columnas_modelo, fill_value=0)

            # --- Hacer la Predicción ---
            # .predict() da la respuesta (0 o 1)
            prediccion_clase = modelo.predict(df_para_predecir)
            
            # .predict_proba() da el "nivel de confianza"
            prediccion_prob = modelo.predict_proba(df_para_predecir)
            
            # --- Mostrar Resultado ---
            clase_predicha = prediccion_clase[0]
            confianza = prediccion_prob[0][clase_predicha] * 100

            print("\n================== RESULTADO ==================")
            if clase_predicha == 1:
                print(f"  El perfil es: IRREGULAR (Clase 1)")
            else:
                print(f"  El perfil es: NO IRREGULAR (Clase 0)")
            
            print(f"  Confianza del modelo: {confianza:.2f}%")
            print("=============================================\n")

        except Exception as e:
            print(f"\n¡Error en la predicción! Asegúrate de escribir los valores exactos.")
            print(f"Detalle del error: {e}")
            
else:
    print("\nNo se pudo cargar el DataFrame. No se puede continuar.")