import pandas as pd

def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
        return df
    
    # --- 1: Limpieza de columnas de texto (LA PARTE CLAVE) ---
    # Esto elimina espacios y rellena valores faltantes.
    columnas_texto = ['SEXO', 'EDAD', 'PAIS', 'REGION']
    columnas_texto_existentes = [col for col in columnas_texto if col in df.columns and df[col].dtype == 'object']
    
    if columnas_texto_existentes:
        print(f"Limpiando espacios y nulos en texto: {', '.join(columnas_texto_existentes)}")
        for col in columnas_texto_existentes:
            # .str.strip() -> Elimina espacios al inicio y al final
            df[col] = df[col].str.strip()
            # .fillna() -> Rellena valores vacíos (NaN) con "IGNORADO"
            df[col] = df[col].fillna('IGNORADO') 

    # --- 2: Columnas numéricas ---
    columnas_numericas = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL", "ESTIMACION"]
    columnas_existentes = [col for col in columnas_numericas if col in df.columns]
    
    if not columnas_existentes:
        print("No se encontraron las columnas numéricas esperadas.")
        return df 
    
    print(f"\nLimpiando columnas numéricas: {columnas_existentes}")
    
    for col in columnas_existentes:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[columnas_existentes] = df[columnas_existentes].fillna(0)
    
    # --- 3: Lógica de negativos ---
    negativos_mask = (df[columnas_existentes] < 0)
    if negativos_mask.any().any():
        print(f"\nReemplazando valores negativos con 0...")
        df[columnas_existentes] = df[columnas_existentes].clip(lower=0)
    
    # --- 4: Convertir a tipo Entero ---
    try:
        for col in columnas_existentes:
            df[col] = df[col].astype('int64') 
    except Exception as e:
        print(f"No se pudo convertir a entero: {e}")
            
    print("--- Limpieza de datos finalizada ---")
    return df