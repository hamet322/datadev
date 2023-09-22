#Importaos las librerias a usar
import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split

# Leemos el set de datos a transformar o limpiar
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

#Trasformamos los datos
def data_preparation(df):
    df = df[['TV','Radio','Newspaper','Sales']]
    return df  

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación

def main():
    
    # Cargar el archivo advertising.csv
    df = read_file_csv("advertising.csv")
    
    # Dividir el conjunto de datos en entrenamiento, validación y scoring
    train_df, temp_df = train_test_split(df, train_size=0.7, random_state=42)
    val_df, score_df = train_test_split(temp_df, train_size=0.67, random_state=42)
    
    # Preparar y exportar los datos de entrenamiento
    train_df = data_preparation(train_df)
    data_exporting(train_df, ['TV', 'Radio', 'Newspaper', 'Sales'], 'radio_train.csv')

    # Preparar y exportar los datos de validación
    val_df = data_preparation(val_df)
    data_exporting(val_df, ['TV', 'Radio', 'Newspaper', 'Sales'], 'radio_val.csv')

    # Preparar y exportar los datos de scoring
    score_df = data_preparation(score_df)
    data_exporting(score_df, ['TV', 'Radio', 'Newspaper', 'Sales'], 'radio_score.csv')

    '''
    # Matriz de Entrenamiento
    df1 = read_file_csv("advertising.csv")
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['TV','Radio','Newspaper', 'Sales'],'radio_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('advertising.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,['TV','Radio','Newspaper','Sales'],'radio_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('advertising.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['TV','Radio','Newspaper','Sales'],'radio_score.csv')'''

if __name__ == "__main__":
    main()