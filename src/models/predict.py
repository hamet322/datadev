#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn import linear_model
import pickle
import os



# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('C:/Users/ecolg/Python/Proyecto Pandas/git practice/data/processed', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = 'C:/Users/ecolg/Python/Proyecto Pandas/git practice/models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('C:/Users/ecolg/Python/Proyecto Pandas/git practice/data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inici
def main():
    df = score_model('radio_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__=="__main__":
    main()



