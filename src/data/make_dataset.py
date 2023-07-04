#Importaos las librerias a usar
import pandas as pd
import numpy as np 
import os

# Leemos el set de datos a transformar o limpiar
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../../data/raw/', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    return df

#Trasformamos los datos
def data_preparation(df):
    df = df[['TV','Sales']]
    return df  

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv("data_train.csv")
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['TV', 'Sales'],'radio_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('data_test.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,['TV','Sales'],'radio_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('data_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['TV','Sales'],'radio_score.csv')

if __name__ == "__main__":
    main()