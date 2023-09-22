#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle 
import os
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('data/processed', filename))
    X_train = df.drop(['Sales'],axis=1)
    y_train = df[['Sales']]
    print(filename, ' cargado correctamente')

    #Escalamos los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)


    # Entrenamos el modelo con toda la muestra
    linear=linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = 'models/best_model.pkl'
    pickle.dump(linear, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('radio_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == '__main__':
    main()



