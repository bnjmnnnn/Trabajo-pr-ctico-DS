import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from limpiar_datos import limpiar_datos

try:

    df_csv = pd.read_csv("8. baseregiones.csv") # leer el archivo CSV y guardarlo en un DataFrame
    df_datos = df_csv.copy() # Crear una copia del DataFrame original para trabajar en ella
    limpiar_datos(df_datos)
    
    # SISTEMA DE FILTROS
    print("\n" + "="*60)
    print("SISTEMA DE FILTROS PARA ANÁLISIS")
    print("="*60)
    
    # Preguntar si quiere filtrar
    filtrar = input("\n¿Desea filtrar los datos? (s/n): ").strip().lower()
    
    if filtrar == 's':
        # Mostrar opciones de países
        paises_disponibles = sorted(df_datos['PAIS'].unique())
        print(f"\nPaíses disponibles ({len(paises_disponibles)}):")
        for i, pais in enumerate(paises_disponibles, 1):
            print(f"  {i}. {pais}")
        
        filtrar_pais = input("\n¿Filtrar por PAÍS? (s/n): ").strip().lower()
        if filtrar_pais == 's':
            pais_seleccionado = input("Ingrese el nombre del PAÍS: ").strip().upper()
            if pais_seleccionado in paises_disponibles:
                df_datos = df_datos[df_datos['PAIS'] == pais_seleccionado].copy()
                print(f"Filtrado por país: {pais_seleccionado}")
            else:
                print(f"País '{pais_seleccionado}' no encontrado. Usando todos los países.")
        
        # Mostrar opciones de regiones (después del filtro de país si se aplicó)
        regiones_disponibles = sorted(df_datos['REGION'].unique())
        print(f"\nRegiones disponibles ({len(regiones_disponibles)}):")
        for i, region in enumerate(regiones_disponibles, 1):
            print(f"  {i}. {region}")
        
        filtrar_region = input("\n¿Filtrar por REGIÓN? (s/n): ").strip().lower()
        if filtrar_region == 's':
            region_seleccionada = input("Ingrese el nombre de la REGIÓN: ").strip().upper()
            if region_seleccionada in regiones_disponibles:
                df_datos = df_datos[df_datos['REGION'] == region_seleccionada].copy()
                print(f"Filtrado por región: {region_seleccionada}")
            else:
                print(f"Región '{region_seleccionada}' no encontrada. Usando todas las regiones.")
        
        # Mostrar resumen de datos filtrados
        print(f"\nTotal de registros después del filtro: {len(df_datos)}")
        if len(df_datos) < 100:
            print(f"ADVERTENCIA: Pocos datos ({len(df_datos)} registros). El modelo puede no ser preciso.")
    else:
        print("Usando todos los datos sin filtros")
    
    print("="*60)
    print(df_datos)
    
    # Predecir 
    X = df_datos[['SEXO', 'EDAD_NUMERICA', 'AÑO ESTIMACION', 'CODREGEO', 'PAIS_CODIGO', 'CENSO AJUSTADO']]
    y = df_datos['RRAA_TOTAL']  # Variable dependiente 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Dividir los datos en entrenamiento y prueba

    model = LinearRegression() # Crear el modelo de regresión lineal
    model.fit(X_train, y_train) # Entrenar el modelo

    y_pred = model.predict(X_test) # Hacer predicciones

    mse = mean_squared_error(y_test, y_pred) # Calcular el error cuadrático medio
    r2 = r2_score(y_test, y_pred) # Calcular R²

    print("\n" + "="*60)
    print("RESULTADOS DEL MODELO")
    print("="*60)
    print(f"Error cuadrático medio (MSE): {mse:.0f}")
    print(f"R² (Coeficiente de determinación): {r2:.1f} ({r2*100:.1f}%)")
    print(f"Registros totales: {len(df_datos)}")
    print(f"Registros entrenamiento: {len(X_train)}")
    print(f"Registros prueba: {len(X_test)}")
    print("="*60)

    # Grafico Regresion Lineal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Histórico y predicción 2024
    historico = df_datos.groupby('AÑO ESTIMACION')['RRAA_TOTAL'].sum().sort_index()
    
    print("\n" + "="*60)
    print("HISTÓRICO DE RRAA_TOTAL POR AÑO")
    print("="*60)
    for anio, total in historico.items():
        print(f'Año {int(anio)}: {total:>10,.0f}')
    print("="*60)
    
    # Predicción para 2024
    ano_max = int(historico.index.max())
    df_pred = df_datos[df_datos['AÑO ESTIMACION'] == ano_max].copy()
    
    if len(df_pred) > 0:
        df_pred['AÑO ESTIMACION'] = ano_max + 1
        
        # Ajustar CENSO AJUSTADO para 2024 basado en tendencia histórica
        if len(historico) > 1:
            crecimiento_promedio = historico.pct_change().mean()
            df_pred['CENSO AJUSTADO'] *= (1 + crecimiento_promedio)
        
        X_pred = df_pred[['SEXO', 'EDAD_NUMERICA', 'AÑO ESTIMACION', 'CODREGEO', 'PAIS_CODIGO', 'CENSO AJUSTADO']].fillna(0)
        prediccion_2024 = model.predict(X_pred).sum()
        
        valor_real_ultimo = historico.iloc[-1]
        crecimiento = (prediccion_2024 - valor_real_ultimo) / valor_real_ultimo if valor_real_ultimo > 0 else 0
        
        print(f"\nPREDICCIÓN PARA {ano_max + 1}")
        print("="*60)
        print(f"Real {ano_max}:      {valor_real_ultimo:,.0f}")
        print(f"Estimado {ano_max + 1}: {prediccion_2024:,.0f}")
        print(f"Crecimiento:       {prediccion_2024 - valor_real_ultimo:,.0f} ({crecimiento*100:+.1f}%)")
        print("="*60)
        
        # Graficar histórico con predicción
        ax1.plot(historico.index, historico.values, 'o-', linewidth=2, markersize=8, label='Histórico', color='blue')
        ax1.plot(ano_max + 1, prediccion_2024, '*', markersize=20, color='red', label=f'Predicción {ano_max + 1}')
        ax1.set_title('RRAA_TOTAL: Histórico y Predicción', fontweight='bold')
        ax1.set_xlabel('Año')
        ax1.set_ylabel('RRAA_TOTAL')
        ax1.legend()
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No hay datos suficientes\npara predicción', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    # Gráfico 2: Valores reales vs predichos
    ax2.scatter(y_test, y_pred, alpha=0.7, color='blue', s=60)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Línea perfecta')
    ax2.set_xlabel("Valores reales")
    ax2.set_ylabel("Valores predichos")
    ax2.set_title(f"Ajuste del Modelo (R² = {r2:.4f})", fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("No se encontro el archivo CSV")
except pd.errors.EmptyDataError:
    print("El archivo se encuentra vacio")
