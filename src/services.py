# Acá almacenaremos las funciones para probar consultas e implementacion de la API

import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def validar_df(df):
    res = pd.DataFrame({
        "Tipo de Dato": df.dtypes,
        "Valores No Nulos": df.count(),
        "Valores Nulos": df.isna().sum(),
        "Valores Únicos": df.astype(str).nunique(dropna=True),
        "Valores Cero": (df == 0).sum(),
        "Inconsistentes ('?')": df.isin(['?']).sum(),
        "Valores Vacíos (string)": df.isin(["", " ", "NA", "NULL", "None"]).sum(),
        "valores_negativos": (df.select_dtypes(include=['int64', 'float64']) < 0).sum()
    })
    res = res.reindex(df.columns)
    return res

# Función para convertir los tipos de datos de las columnas de un DataFrame según un diccionario de tipos.
def convertir_tipos(df, diccionario):
    # Se recorre cada par (columna, tipo esperado) en el diccionario.
    for columna, tipo_esperado in diccionario.items():
        # Primero se verifica que la columna exista en el DataFrame.
        if columna in df.columns:
            # Si el tipo esperado es 'int', se convierte la columna a número entero:
            # 1. pd.to_numeric intenta convertir a número; en caso de error, reemplaza con NaN.
            # 2. fillna(0) reemplaza los NaN por 0.
            # 3. astype(int) convierte la columna a entero.
            if tipo_esperado == int:
                df[columna] = pd.to_numeric(df[columna], errors='coerce').fillna(0).astype(int)
            # Si el tipo esperado es 'float', se convierte la columna a número flotante.
            elif tipo_esperado == float:
                df[columna] = pd.to_numeric(df[columna], errors='coerce')
            # Si el tipo esperado es 'bool', se convierte la columna a booleano.
            # La conversión se hace comparando el valor (tras limpiarlo) con 'true' o '1'.
            elif tipo_esperado == bool:
                df[columna] = df[columna].apply(lambda x: str(x).strip().lower() in ['true', '1'] if pd.notna(x) else False)
            # Si el tipo esperado es una lista o un diccionario, se verifica si el valor es una cadena
            # y se utiliza la función auxiliar 'convertir_a_estructura' para intentar convertirlo.
            elif tipo_esperado in [list, dict]:
                df[columna] = df[columna].apply(lambda x: convertir_a_estructura(x, tipo_esperado) if isinstance(x, str) else x)
            # Si el tipo esperado es una fecha (ya sea 'datetime' o 'pd.Timestamp'),
            # se convierte la columna a formato datetime usando pd.to_datetime.
            elif tipo_esperado in [datetime, pd.Timestamp]:
                df[columna] = pd.to_datetime(df[columna], errors='coerce')
            # En caso de que no se trate de un tipo especial, se convierte la columna a cadena de texto.
            else:
                df[columna] = df[columna].astype(str).fillna("")
    # Se retorna el DataFrame con las conversiones aplicadas.
    return df

# Función auxiliar para convertir strings en listas o diccionarios de forma segura
def convertir_a_estructura(valor, tipo_esperado):
    """
    Convierte un valor string en una lista o diccionario si es posible, de lo contrario retorna el valor original.

    Parámetros:
    -----------
    valor : str
        Valor que potencialmente es una lista o un diccionario en formato string.
    tipo_esperado : type
        Tipo esperado (list o dict).

    Retorno:
    --------
    list/dict/None
        Retorna el objeto convertido o None si la conversión falla.
    """
    try:
        if isinstance(valor, str):
            resultado = ast.literal_eval(valor)
            return resultado if isinstance(resultado, tipo_esperado) else None
        return valor  # Si ya es list o dict, lo devuelve tal cual
    except (ValueError, SyntaxError):
        return None

############################################################################################################

def validar_estructura_df(df, diccionario):
    """
    Valida la estructura de cada fila del DataFrame comparando con el diccionario de tipos esperado.
    
    La función revisa cada fila para determinar si:
      - Contiene todas las columnas esperadas.
      - Cada valor (no nulo) en la fila es del tipo especificado en el diccionario.
    
    Parámetros:
      df : pd.DataFrame
         DataFrame a validar.
      diccionario : dict
         Diccionario donde las llaves son los nombres de columna y los valores son los tipos esperados.
         
    Retorna:
      tuple: (cantidad_inconsistentes, lista_indices)
         - cantidad_inconsistentes: número de filas que no cumplen con la estructura.
         - lista_indices: lista con los índices (números de fila) de las filas inconsistentes.
         
    Si el DataFrame no contiene todas las columnas esperadas, se lanza un ValueError.
    """
    inconsistentes = []
    
    # Verificar que el DataFrame contenga todas las columnas esperadas
    columnas_esperadas = set(diccionario.keys())
    columnas_df = set(df.columns)
    if not columnas_esperadas.issubset(columnas_df):
        raise ValueError("El DataFrame no contiene todas las columnas esperadas.")
    
    # Recorrer cada fila del DataFrame
    for idx, row in df.iterrows():
        fila_inconsistente = False
        # Validar cada columna según el tipo esperado
        for col, tipo_esperado in diccionario.items():
            valor = row[col]
            # Solo se evalúa pd.isna si el valor es un escalar; si no, se omite esta comprobación
            if np.isscalar(valor) and pd.isna(valor):
                continue
            
            # Validación según el tipo esperado
            if tipo_esperado == int:
                if not isinstance(valor, int):
                    fila_inconsistente = True
                    break
            elif tipo_esperado == float:
                if not isinstance(valor, (float, int)):
                    fila_inconsistente = True
                    break
            elif tipo_esperado == bool:
                if not isinstance(valor, bool):
                    fila_inconsistente = True
                    break
            elif tipo_esperado in [list, dict]:
                if not isinstance(valor, tipo_esperado):
                    fila_inconsistente = True
                    break
            elif tipo_esperado in [datetime, pd.Timestamp]:
                if not (isinstance(valor, pd.Timestamp) or isinstance(valor, datetime)):
                    fila_inconsistente = True
                    break
            elif tipo_esperado == str:
                if not isinstance(valor, str):
                    fila_inconsistente = True
                    break
            else:
                if not isinstance(valor, tipo_esperado):
                    fila_inconsistente = True
                    break
        
        if fila_inconsistente:
            inconsistentes.append(idx)
    
    if inconsistentes:
        print(f"Se encontraron {len(inconsistentes)} filas inconsistentes: {inconsistentes}")
    else:
        print("No se encontraron filas inconsistentes.")
    
    return inconsistentes

############################################################################################################

# Esta función valida las filas del CSV y detecta las que tienen datos incorrectos
def validar_estructura_csv(df, diccionario):
    """
    Valida la estructura del CSV y detecta las filas con datos incorrectos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos a validar.
    diccionario : dict
        Diccionario que define los tipos esperados para cada columna.
    
    Retorno:
    --------
    list
        Lista de tuplas que contiene el número de fila y los campos con datos inconsistentes.
    """
    inconsistencias = []  # Aquí guardaremos las filas con problemas

    # Recorremos todas las filas del DataFrame
    for index, fila in df.iterrows():
        campos_inconsistentes = []  # Campos con datos incorrectos en esta fila

        # Recorremos cada columna de la fila
        for columna, tipo_esperado in diccionario.items():
            if columna in df.columns:
                valor = fila[columna]

                # Validar el tipo de dato del valor
                if not validar_tipo(valor, tipo_esperado):
                    campos_inconsistentes.append(columna)  # Si es incorrecto, añadimos la columna a la lista
                    print(f"Fila {index + 1}, Columna '{columna}': Valor inconsistente -> {valor} ({type(valor)})")

        # Si hay campos inconsistentes, guardamos la fila en la lista de inconsistencias
        if campos_inconsistentes:
            inconsistencias.append((index, campos_inconsistentes))

    return inconsistencias

# Esta función valida si un valor tiene el tipo correcto
def validar_tipo(valor, tipo_esperado):
    """
    Valida si un valor cumple con el tipo de dato esperado.

    Parámetros:
    -----------
    valor : cualquier tipo
        El valor que se desea validar.
    tipo_esperado : type
        El tipo de dato esperado para el valor.
    
    Retorno:
    --------
    bool
        True si el valor es del tipo correcto o es nulo, False en caso contrario.
    """
    # Si el valor es una lista, diccionario o array
    if isinstance(valor, (list, dict, np.ndarray)):
        # Comprobamos si el valor es del tipo correcto
        if tipo_esperado == list and isinstance(valor, list):
            return True
        if tipo_esperado == dict and isinstance(valor, dict):
            return True
        return False  # Si no coincide, es incorrecto

    # Si el valor es nulo (NaN), es válido
    if pd.isnull(valor):
        return True

    # Intentamos convertir cadenas con estructuras anidadas
    if tipo_esperado in [list, dict] and isinstance(valor, str):
        try:
            valor = ast.literal_eval(valor)  # Convertir la cadena a lista o diccionario
        except (ValueError, SyntaxError):
            return False  # Si falla, es incorrecto

    # Finalmente, comprobamos si el valor es del tipo esperado
    return isinstance(valor, tipo_esperado)

def formato_fecha(df, columna_fecha):
    """
    Convierte una columna de tipo object a formato datetime.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene la columna a convertir.
    columna_fecha : str
        Nombre de la columna que contiene las fechas en formato string.

    Retorno:
    --------
    pd.DataFrame
        El DataFrame con la columna convertida a formato datetime.
    """
    # Convertir la columna a formato datetime
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce', format='%Y-%m-%d')
    
    # Verificar si hay errores en la conversión
    nulos = df[columna_fecha].isnull().sum()
    if nulos > 0:
        print(f"Advertencia: {nulos} valores no pudieron convertirse a formato fecha.")
    
    return df

############################################################################################################

def obtener_campos_json(df, columna):
    """
    Extrae la lista de nombres de campos (keys) presentes en los diccionarios
    contenidos en los valores de la columna especificada del DataFrame.
    
    Parámetros:
      df : pd.DataFrame
         DataFrame que contiene la columna con datos JSON.
      columna : str
         Nombre de la columna a inspeccionar.
         
    Retorno:
      list : Lista con los nombres de los campos (union de keys) encontrados.
    """
    campos = set()
    
    # Se recorren los valores no nulos de la columna
    for valor in df[columna].dropna():
        try:
            # Si el valor es una cadena, intentar evaluarla a estructura Python
            if isinstance(valor, str):
                valor = ast.literal_eval(valor)
            # Si es una lista, se itera sobre sus elementos
            if isinstance(valor, list):
                for elemento in valor:
                    if isinstance(elemento, dict):
                        campos.update(elemento.keys())
            # Si es un diccionario, se agregan sus keys directamente
            elif isinstance(valor, dict):
                campos.update(valor.keys())
        except Exception:
            # Si ocurre algún error en la evaluación o extracción, se omite ese valor
            continue
    
    return list(campos)




def extraer_campo(json_obj, campo):
    """
    Función auxiliar que extrae el valor de un campo específico de un objeto JSON.
    Si json_obj es una cadena, se evalúa a estructura Python.
    Si es una lista, se busca en cada diccionario el primer valor encontrado para ese campo.
    Retorna None si no se encuentra o si ocurre un error.
    """
    try:
        if isinstance(json_obj, str):
            json_obj = ast.literal_eval(json_obj)
        if isinstance(json_obj, dict):
            return json_obj.get(campo)
        elif isinstance(json_obj, list):
            for d in json_obj:
                if isinstance(d, dict) and campo in d:
                    return d.get(campo)
            return None
        else:
            return None
    except Exception:
        return None

def extraer_campos_json(df, columna_json, campos):
    """
    Extrae campos específicos de un objeto JSON contenido en una columna y
    retorna un nuevo DataFrame que contiene únicamente los campos extraídos
    junto con la columna 'movie_id' del DataFrame original.

    Parámetros:
      df : pd.DataFrame
         DataFrame que contiene la columna con los objetos JSON y la columna 'movie_id'.
      columna_json : str
         Nombre de la columna que contiene los objetos JSON.
      campos : list
         Lista con los nombres de los campos a extraer.
         
    Retorno:
      pd.DataFrame
         Nuevo DataFrame con columnas correspondientes a cada campo extraído y la columna 'movie_id'.
    """
    datos_extraidos = []
    
    # Se recorre cada fila del DataFrame
    for index, row in df.iterrows():
        json_obj = row[columna_json]
        fila_extraida = {}
        
        # Extrae cada campo solicitado
        for campo in campos:
            fila_extraida[campo] = extraer_campo(json_obj, campo)
        
        # Agrega la columna 'movie_id' desde el DataFrame original para referencia
        # Se asume que la columna se llama 'movie_id'; de no ser así, ajusta el nombre.
        fila_extraida['movie_id'] = row.get('movie_id', None)
        
        datos_extraidos.append(fila_extraida)
    
    return pd.DataFrame(datos_extraidos)

############################################################################################################


# Diccionarios para mapear nombres de meses y días a números
MES_MAP = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 
    'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 
    'noviembre': 11, 'diciembre': 12
}
# Diccionario inverso: de número a nombre (con la primera letra en mayúscula)
MES_MAP_INV = {v: k.capitalize() for k, v in MES_MAP.items()}

DIA_MAP = {
    'lunes': 1, 'martes': 2, 'miercoles': 3, 'miércoles': 3, 
    'jueves': 4, 'viernes': 5, 'sabado': 6, 'sábado': 6, 'domingo': 7
}
# Diccionario inverso para días: 0 corresponde a Lunes, etc.
DIA_MAP_INV = {1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 4: 'Jueves', 5: 'Viernes', 6: 'Sábado', 7: 'Domingo'}

def cantidad_filmaciones_mes(mes, df_movies: pd.DataFrame) -> str:
    """
    Devuelve la cantidad de películas estrenadas en el mes indicado.
    Se acepta el mes como nombre (e.g. "Enero", "febrero") o como número (1-12).
    En la salida se muestra el nombre del mes (ej. "Enero").
    """
    # Verificar si mes es numérico o una cadena que represente un número
    if isinstance(mes, (int, float)) or (isinstance(mes, str) and mes.isdigit()):
        mes_num = int(mes)
        if not (1 <= mes_num <= 12):
            return f"El número de mes '{mes_num}' no es válido. Debe estar entre 1 y 12."
        mes_nombre = MES_MAP_INV[mes_num]
    else:
        mes_str = mes.lower()
        if mes_str not in MES_MAP:
            return f"El mes '{mes}' no es válido."
        mes_num = MES_MAP[mes_str]
        mes_nombre = mes_str.capitalize()
    
    # Asegurar que release_date es datetime (esto modifica df_movies in situ)
    if not np.issubdtype(df_movies['release_date'].dtype, np.datetime64):
        df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], errors='coerce')
    
    count = df_movies[df_movies['release_date'].dt.month == mes_num].shape[0]
    return f"{count} película(s) fueron estrenadas en el mes de {mes_nombre}."

def cantidad_filmaciones_dia(dia, df_movies: pd.DataFrame) -> str:
    """
    Devuelve la cantidad de películas estrenadas en el día de la semana indicado.
    Se acepta el día como nombre (por ejemplo, "Lunes", "martes") o como número (1..7).
    Internamente, pandas usa dayofweek = 0..6 (lunes=0, domingo=6),
    así que sumamos 1 para que lunes sea 1, ... y domingo sea 7.
    """
    # 1. Verificar si la entrada 'dia' es numérica
    if isinstance(dia, (int, float)) or (isinstance(dia, str) and dia.isdigit()):
        dia_num = int(dia)
        # Debe estar entre 1 y 7
        if not (1 <= dia_num <= 7):
            return f"El número de día '{dia_num}' no es válido. Debe estar entre 1 (lunes) y 7 (domingo)."
        dia_nombre = DIA_MAP_INV[dia_num]
    else:
        # 2. Si no es numérico, buscamos en DIA_MAP
        dia_str = dia.lower()
        if dia_str not in DIA_MAP:
            return f"El día '{dia}' no es válido."
        dia_num = DIA_MAP[dia_str]
        dia_nombre = DIA_MAP_INV[dia_num]

    # 3. Asegurar que release_date es datetime
    if not np.issubdtype(df_movies['release_date'].dtype, np.datetime64):
        df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], errors='coerce')

    # 4. Contar cuántas películas tienen dayofweek+1 = dia_num
    #    (pandas: lunes=0, martes=1, ..., domingo=6; sumamos 1 para lunes=1,... domingo=7)
    mask = (df_movies['release_date'].dt.dayofweek + 1 == dia_num)
    count = df_movies[mask].shape[0]

    return f"{count} película(s) fueron estrenadas en el día {dia_nombre}."

def score_titulo(titulo: str, df_movies: pd.DataFrame) -> str:
    """
    Dado el título de una película, retorna un mensaje con el título, 
    el año de estreno y el score/popularidad.

    Ejemplo de retorno:
      "La película X fue estrenada en el año X con un score/popularidad de X."
    """
    # Filtrar por título (ignorando mayúsculas/minúsculas)
    fila = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if fila.empty:
        return f"No se encontró la película '{titulo}'."
    
    # Asegurarse de que release_date sea datetime
    if not pd.api.types.is_datetime64_any_dtype(fila['release_date']):
        fila = fila.copy()
        fila['release_date'] = pd.to_datetime(fila['release_date'], errors='coerce')
    
    # Extraer año de estreno; si no se conoce, se marca como "desconocido"
    fecha = fila.iloc[0]['release_date']
    anio = fecha.year if pd.notna(fecha) else "desconocido"
    
    # Extraer score/popularidad
    score = fila.iloc[0]['vote_average']
    return f"La película '{titulo}' fue estrenada en el año {anio} con un score/popularidad de {score}."

def votos_titulo(titulo: str, df_movies: pd.DataFrame) -> str:
    """
    Dado el título de una película, retorna un mensaje con el título, la cantidad de votos 
    y el promedio de las votaciones, siempre que la película cuente con al menos 2000 valoraciones.
    En caso contrario, se retorna un mensaje indicando que no se cumple la condición.

    Ejemplo de retorno:
      "La película X fue estrenada en el año X. Cuenta con X valoraciones, con un promedio de X."
    """
    # Filtrar por título (ignorando mayúsculas/minúsculas)
    fila = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if fila.empty:
        return f"No se encontró la película '{titulo}'."
    
    # Asegurarse de que release_date sea datetime
    if not pd.api.types.is_datetime64_any_dtype(fila['release_date']):
        fila = fila.copy()
        fila['release_date'] = pd.to_datetime(fila['release_date'], errors='coerce')
    
    vote_count = fila.iloc[0]['vote_count']
    if vote_count < 2000:
        return f"La película '{titulo}' no cumple con la condición de tener al menos 2000 valoraciones."
    
    vote_average = fila.iloc[0]['vote_average']
    fecha = fila.iloc[0]['release_date']
    anio = fecha.year if pd.notna(fecha) else "desconocido"
    
    return (f"La película '{titulo}' fue estrenada en el año {anio}. "
            f"Cuenta con {vote_count} valoraciones, con un promedio de {vote_average}.")



def exito_actor(nombre_actor: str, df_cast: pd.DataFrame, df_movies: pd.DataFrame) -> str:
    """
    Dado el nombre de un actor, retorna un mensaje indicando la cantidad de filmaciones en las que ha participado,
    el retorno total obtenido y el promedio de retorno por filmación. Se excluyen roles de directores, ya que se
    asume que esta información proviene exclusivamente del DataFrame de reparto (df_cast).

    Ejemplo de retorno:
      "El actor X ha participado en Y filmaciones, consiguiendo un retorno total de Z y un promedio de A por filmación."
    """
    # Filtrar df_cast para encontrar filas donde el actor coincide (sin distinguir mayúsculas/minúsculas)
    df_actor = df_cast[df_cast['name'].str.lower() == nombre_actor.lower()]
    if df_actor.empty:
        return f"No se encontró al actor '{nombre_actor}'."
    
    # Obtener los movie_id únicos en los que aparece el actor
    peliculas_ids = df_actor['movie_id'].unique()
    
    # Filtrar df_movies para obtener las películas correspondientes
    df_actor_movies = df_movies[df_movies['movie_id'].isin(peliculas_ids)]
    if df_actor_movies.empty:
        return f"No se encontraron películas para el actor '{nombre_actor}'."
    
    # Calcular el total y promedio del retorno
    total_return = df_actor_movies['return'].sum()
    cantidad = df_actor_movies.shape[0]
    promedio_return = total_return / cantidad if cantidad > 0 else 0
    
    peliculas_detalles = "\n".join(
        [f"- {row['title']} (Fecha: {row['release_date'].date() if pd.notna(row['release_date']) else 'N/A'}), "
         f"Retorno: {row['return']:.2f}, Costo: {row['budget']}, Ganancia: {row['revenue']}"
         for idx, row in df_actor_movies.iterrows()]
    )
    
    return (f"El actor '{nombre_actor}' ha participado en {cantidad} filmación(es), "
            f"consiguiendo un retorno total de {total_return:.2f} y un promedio de {promedio_return:.2f} por filmación.\n"
            f"Lista de Peliculas:\n{peliculas_detalles}")

def exito_director(nombre_director: str, df_crew: pd.DataFrame, df_movies: pd.DataFrame) -> str:
    """
    Dado el nombre de un director, retorna un mensaje con el éxito del mismo medido a través del retorno.
    Además, se devuelve una lista con el nombre de cada película, la fecha de lanzamiento, el retorno individual,
    el costo (budget) y la ganancia (revenue) de la misma.

    Ejemplo de retorno:
      "Director: X
       Películas:
       - Título: Y, Fecha: YYYY-MM-DD, Retorno: Z, Costo: W, Ganancia: R
       - ... "
    """
    # Filtrar df_crew para seleccionar únicamente las filas donde el job es "Director" (sin distinción de mayúsculas/minúsculas)
    df_director = df_crew[(df_crew['job'].str.casefold() == 'director') &
                          (df_crew['name'].str.casefold() == nombre_director.casefold())]
    if df_director.empty:
        return f"No se encontró al director '{nombre_director}'."
    
    # Obtener los movie_id únicos para los que el director trabajó
    peliculas_ids = df_director['movie_id'].unique()
    
    # Filtrar df_movies para obtener los detalles de esas películas
    df_director_movies = df_movies[df_movies['movie_id'].isin(peliculas_ids)]
    if df_director_movies.empty:
        return f"No se encontraron películas para el director '{nombre_director}'."
    
    mensaje = f"Director: {nombre_director}\nPelículas:\n"
    for idx, row in df_director_movies.iterrows():
        # Extraer fecha (convertida a string si es datetime), retorno, presupuesto y recaudación
        fecha = row['release_date'].date() if pd.notna(row['release_date']) else "N/A"
        retorno = row['return']
        presupuesto = row['budget']
        recaudacion = row['revenue']
        titulo = row['title']
        mensaje += (f"- {titulo} (Fecha: {fecha}), Retorno: {retorno:.2f}, "
                    f"Costo: {presupuesto}, Ganancia: {recaudacion}\n")
    return mensaje
