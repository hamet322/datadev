#importar librerias
import pickle

import numpy as np
from flask import Flask, request, render_template
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
with open('models/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Inicializa la variable de predicción
    
    if request.method == 'POST':
        # Obtiene los valores ingresados por el usuario
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])
        
        # Realizar la predicción con el modelo
        prediction = model.predict([[tv, radio, newspaper]])

        prediction = np.round(prediction[0],2)
    
    # Renderiza la página inicial, incluyendo el valor de predicción
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
