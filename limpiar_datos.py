def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
    else:
        # Columnas a verificar y limpiar
        columnas = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL"]
        
        # Reemplazar valores vacíos y NaN con 0 
        df[columnas] = df[columnas].replace("", 0).fillna(0)
        print(f"Valores nulos y vacíos reemplazados con 0 en: {', '.join(columnas)}")
        
        # Verificar valores negativos antes de convertir
        negativos = (df[columnas] < 0).any(axis=1) # Guarda si hay negativos en alguna fila
        
        if negativos.any():
            print(f"\nExisten {negativos.sum()} filas con valores negativos:")
            for col in columnas: 
                num_negativos = (df[col] < 0).sum() # Cuenta de negativos por columna ("CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL")
                if num_negativos > 0:
                    print(f"  - {col}: {num_negativos} valores negativos")
            
            # Convertir todos los valores negativos a su valor absoluto
            df[columnas] = df[columnas].abs()
            print(f"\nValores negativos convertidos a positivos (valor absoluto) en: {', '.join(columnas)}")