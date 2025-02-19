# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import json as js 
import ast  
import re 
import os

import importlib
import src.services as services
importlib.reload(services)
from src.services import exito_actor, exito_director, score_titulo, votos_titulo, cantidad_filmaciones_dia, cantidad_filmaciones_mes
from fastapi import FastAPI

app = FastAPI()

# Importamos funciones de servicio
from src.services import (
    cantidad_filmaciones_mes,
    cantidad_filmaciones_dia,
    score_titulo,
    votos_titulo,
    exito_actor,
    exito_director
)

# Creamos la instancia de FastAPI
app = FastAPI()

# Cargamos los csv data_movies, data_cast y data_crew
data_movies = pd.read_csv('transformados_processed/data_movies.parquet', low_memory=False)
data_cast = pd.read_csv('transformados_processed/data_cast.parquet', low_memory=False)
data_crew = pd.read_csv('transformados_processed/data_crew.parquet', low_memory=False)

@app.get("/")
def root():
    return {"message": "¡API funcionando correctamente!"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def get_cantidad_filmaciones_mes(mes: str):
    """
    Endpoint para consultar cuántas películas se estrenaron en el mes dado (ej: 'enero' o '2').
    """
    resultado = cantidad_filmaciones_mes(mes, data_movies)
    # Podemos retornar un dict para que sea JSON:
    return {"resultado": resultado}

@app.get("/cantidad_filmaciones_dia/{dia}")
def get_cantidad_filmaciones_dia(dia: str):
    """
    Endpoint para consultar cuántas películas se estrenaron en el día de la semana (ej: 'lunes' o '1').
    """
    resultado = cantidad_filmaciones_dia(dia, data_movies)
    return {"resultado": resultado}

@app.get("/score_titulo/{titulo}")
def get_score_titulo(titulo: str):
    """
    Devuelve el score de la película y su año de estreno.
    """
    resultado = score_titulo(titulo, data_movies)
    return {"resultado": resultado}

@app.get("/votos_titulo/{titulo}")
def get_votos_titulo(titulo: str):
    """
    Devuelve la cantidad de votos y promedio de la película, siempre que tenga al menos 2000 valoraciones.
    """
    resultado = votos_titulo(titulo, data_movies)
    return {"resultado": resultado}

@app.get("/exito_actor/{nombre_actor}")
def get_exito_actor(nombre_actor: str):
    """
    Devuelve la cantidad de filmaciones, retorno total y promedio del actor.
    """
    resultado = exito_actor(nombre_actor, data_cast, data_movies)
    return {"resultado": resultado}

@app.get("/exito_director/{nombre_director}")
def get_exito_director(nombre_director: str):
    """
    Devuelve la info de las películas dirigidas por el director y su retorno.
    """
    resultado = exito_director(nombre_director, data_crew, data_movies)
    return {"resultado": resultado}
