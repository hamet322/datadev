#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os
from sklearn.preprocessing import StandardScaler

# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = 'models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_eval = df.drop(['Sales'],axis=1)
    y_eval = df[['Sales']]

    #Escalamos los datos
    scaler = StandardScaler()
    X_eval = scaler.fit_transform(X_eval)
    
    #Predecimos con el modelo importado correct
    y_eval_test=model.predict(X_eval)

    #Coeficiente
    model.coef_
    #Erro medio cuadrado
    print('Error medio cuadrado:', r2_score(y_eval, y_eval_test))
    

# Validación desde el inicio
def main():
    df = eval_model('radio_val.csv')
    print('Finalizó la validación del Modelo')



if __name__=='__main__':
    main()

