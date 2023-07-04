#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os

# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../../data/processed', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Sales'],axis=1)
    y_test = df[['Sales']]
    y_pred_test=model.predict(X_test)

    #Coeficiente
    model.coef_
    #Erro medio cuadrado
    print('Error medio cuadrado:', r2_score(y_test, y_pred_test))
    

# Validación desde el inicio
def main():
    df = eval_model('radio_val.csv')
    print('Finalizó la validación del Modelo')



if __name__=='__main__':
    main()

