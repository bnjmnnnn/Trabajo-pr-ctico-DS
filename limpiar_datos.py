import pandas as pd
def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
        return df
    else:
        # Columnas a verificar y limpiar
        columnas = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL"]
        
        # Verificar que las columnas existen
        columnas_existentes = [col for col in columnas if col in df.columns]
        if not columnas_existentes:
            print("No se encontraron las columnas esperadas para limpiar")
            return df
        
        print(f"Limpiando columnas: {columnas_existentes}")
        
        # Convertir a numérico primero, coerciendo errores a NaN
        for col in columnas_existentes:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Reemplazar valores vacíos y NaN con 0 
        df[columnas_existentes] = df[columnas_existentes].fillna(0)
        print(f"Valores nulos y vacíos reemplazados con 0 en: {', '.join(columnas_existentes)}")
        
        # Verificar valores negativos antes de convertir
        negativos = (df[columnas_existentes] < 0).any(axis=1) # Guarda si hay negativos en alguna fila
        
        if negativos.any():
            print(f"\nExisten {negativos.sum()} filas con valores negativos:")
            for col in columnas_existentes: 
                num_negativos = (df[col] < 0).sum() # Cuenta de negativos por columna
                if num_negativos > 0:
                    print(f"  - {col}: {num_negativos} valores negativos")
            
            # Convertir todos los valores negativos a su valor absoluto
            df[columnas_existentes] = df[columnas_existentes].abs()
            print(f"\nValores negativos convertidos a positivos (valor absoluto) en: {', '.join(columnas_existentes)}")
        
        return df