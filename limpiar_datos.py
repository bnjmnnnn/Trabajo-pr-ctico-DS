import pandas as pd

def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
    else:
        # Primero transformar SEXO y EDAD (ANTES de convertir a numérico)
        # Transformar H y M a 1 y 0 respectivamente en la columna "SEXO"
        df['SEXO'] = df['SEXO'].map({'H': 1, 'M': 0})
        print(f"Columna SEXO codificada: H->1, M->0")

        # Pasar EDAD de rangos a promedio numerico
        edad_map = {
            "00 A 04": 2,
            "05 A 09": 7,
            "10 A 14": 12,
            "15 A 19": 17,
            "20 A 24": 22,
            "25 A 29": 27,
            "30 A 34": 32,
            "35 A 39": 37,
            "40 A 44": 42,
            "45 A 49": 47,
            "50 A 54": 52,
            "55 A 59": 57,
            "60 A 64": 62,
            "65 A 69": 67,
            "70 A 74": 72,
            "75 A 79": 77,
            "80 O MÁS": 85,
            "IGNORADA": -1
        }
        
        # Crear columna EDAD_NUMERICA
        df['EDAD_NUMERICA'] = df['EDAD'].map(edad_map)
        
        # Verificar si hay valores no mapeados
        valores_no_mapeados = df[df['EDAD_NUMERICA'].isna()]['EDAD'].unique()
        if len(valores_no_mapeados) > 0:
            print(f"\n Valores de EDAD no reconocidos: {valores_no_mapeados}")
            df['EDAD_NUMERICA'] = df['EDAD_NUMERICA'].fillna(-1)
        
        print(f"Columna EDAD_NUMERICA creada")
        
        # Codificar PAIS a números
        pais_map = {
            "ARGENTINA": 1,
            "BOLIVIA": 2,
            "BRASIL": 3,
            "COLOMBIA": 4,
            "ECUADOR": 5,
            "PERÚ": 6,
            "VENEZUELA": 7,
            "PARAGUAY": 8,
            "URUGUAY": 9,
            "ESTADOS UNIDOS": 10,
            "ESPAÑA": 11,
            "MÉXICO": 12,
            "CUBA": 13,
            "R. DOMINICANA": 14,
            "HAITÍ": 15,
            "CHINA": 16,
            "ALEMANIA": 17,
            "FRANCIA": 18,
            "ITALIA": 19,
            "OTRO PAÍS": 20,
            "PAÍS IGNORADO": 0
        }

        # Crear columna PAIS_CODIGO
        df['PAIS_CODIGO'] = df['PAIS'].map(pais_map)
        
        # Verificar si hay países no mapeados
        valores_pais_no_mapeados = df[df['PAIS_CODIGO'].isna()]['PAIS'].unique()
        if len(valores_pais_no_mapeados) > 0:
            print(f"\nPaíses no reconocidos: {valores_pais_no_mapeados}")
            df['PAIS_CODIGO'] = df['PAIS_CODIGO'].fillna(0)
        
        print(f"Columna PAIS_CODIGO creada")
        
        # Arreglo para convertir columnas numéricas 
        columnas_numericas = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL", "ESTIMACION"]
        
        # Convertir a numérico (por si hay strings mezclados)
        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"Columnas numéricas convertidas y NaN rellenados con 0")

        # Verificar valores negativos (solo en columnas numéricas)
        negativos = (df[columnas_numericas] < 0).any(axis=1)
        
        if negativos.any():
            print(f"\nExisten {negativos.sum()} filas con valores negativos:")
            for col in columnas_numericas: 
                num_negativos = (df[col] < 0).sum()
                if num_negativos > 0:
                    print(f"  - {col}: {num_negativos} valores negativos")
            
            # Convertir todos los valores negativos a su valor absoluto
            df[columnas_numericas] = df[columnas_numericas].abs()
            print(f"\nValores negativos convertidos a positivos (valor absoluto)")
