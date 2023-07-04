# Ejecución de Tests

### Paso 1: Fork del Repositorio Original

Inicie sesión en Github. Luego, vaya al enlace del proyecto original y dé click al botón "Fork". Esto copiará todo el proyecto en su usuario de Github.


### Paso 2: Levantar Entorno Avanzado de Python

```
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```


### Paso 3: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```


### Paso 4: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/datadev.git
```


### Paso 5: Instalar los pre-requisitos

```
cd datadev/

pip install -r requirements.txt
```


### Paso 6: Ejecutar las pruebas en el entorno

```
cd src

python make_dataset.py

python train.py

python evaluate.py

python predict.py
```


### Paso 7: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```
