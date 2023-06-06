# DATASETS Y APLICACION EN ESCALAMIENTO DE DATOS PYTHON
<div align="center">
<img src="https://img.shields.io/badge/Codigo-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
<img src="https://img.shields.io/badge/Notebook-Colab-F9AB00?style=for-the-badge&logo=google%20colab&logoColor=F9AB00" alt="Colab Badge"/>
<img src="https://img.shields.io/badge/Licencia-MIT-009E60?style=for-the-badge&logo=Creative%20Commons&logoColor=009E60" alt="MIT License Badge"/>
<img src="https://img.shields.io/badge/Tematica-Analitica_de_Datos-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=007ACC" alt="Visual Studio Code Badge"/>
</div>
<br>
En el mundo de la analítica de datos, un dataset es simplemente un conjunto de información que nos ayuda a entender y analizar algo en particular. Podemos imaginarlo como una colección de datos organizados que nos dan una idea más clara de lo que está sucediendo. Ahora, ¿qué pasa cuando tenemos diferentes tipos de datos en nuestro dataset? Algunos pueden estar en una escala de 1 a 10, otros en una escala de 1 a 100, y así sucesivamente. Esto puede causar problemas, porque las diferencias en la escala pueden afectar nuestros análisis.

Aquí es donde entra en juego el escalamiento de datos. Es una técnica que nos ayuda a poner todos los datos en la misma escala, para que podamos compararlos y analizarlos de manera más justa.Así que aplicamos el escalamiento de datos. Esto significa ajustar todas las puntuaciones dentro de un rango específico, como de 0 a 1. De esta manera, todos los atributos tienen el mismo peso y se pueden comparar de manera equitativa.

Una vez que hemos escalado los datos, podemos realizar diferentes análisis o construir modelos utilizando el dataset transformado. Por ejemplo, podemos crear gráficas de líneas para ver las puntuaciones escaladas de los atributos y analizar las tendencias o patrones.
# Escalamiento de Datos en un caso de prediccion de colmenas de abejas
## Descripción
El código proporcionado realiza diversas visualizaciones y análisis de datos de producción de miel en relación al número de colonias de abejas. Utiliza la biblioteca Pandas para la manipulación de datos, la biblioteca Scikit-learn para la regresión lineal y la biblioteca Matplotlib para la visualización de gráficos.

## Requisitos

- Python 3.x
- Bibliotecas:
  - pandas
  - sklearn
  - matplotlib
  - seaborn (opcional)

## Instalación

1. Asegúrate de tener Python 3.x instalado en tu sistema.
2. Abre una terminal o línea de comandos.
3. Ejecuta el siguiente comando para instalar las bibliotecas requeridas:

   ```shell
   pip install pandas sklearn matplotlib seaborn
   ```
   Nota: Puedes utilizar un entorno virtual para mantener las dependencias del proyecto aisladas.
   
Descarga el archivo honeyproduction.csv y asegúrate de que se encuentre en el mismo directorio que el script de Python.
Abre el script de Python en tu editor de código preferido.
Ejecuta el script y verás los diferentes gráficos y análisis de datos generados.
Si tienes alguna pregunta o necesitas más información, no dudes en preguntar.
## Importar bibliotecas y cargar datos
```python
import pandas as pd                              # Importar la biblioteca pandas para el manejo de datos
from sklearn.linear_model import LinearRegression  # Importar el modelo de regresión lineal de sklearn

data = pd.read_csv("./honeyproduction.csv")       # Cargar los datos del archivo CSV en un DataFrame llamado 'data'
data                                              # Mostrar el DataFrame 'data'

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib para visualización

plt.ylabel("honey production")                    # Establecer el título del eje y como "honey production"
plt.xlabel("number colonies")                     # Establecer el título del eje x como "number colonies"
plt.scatter(data["numcol"],data["totalprod"],color="blue")  # Crear un gráfico de dispersión con los datos de "numcol" en el eje x y "totalprod" en el eje y
plt.show()                                        # Mostrar el gráfico

from sklearn import linear_model                  # Importar el módulo de modelos lineales de sklearn

regresion = linear_model.LinearRegression()       # Crear un objeto de regresión lineal llamado 'regresion'

colonies = data["numcol"].values.reshape((-1, 1))  # Obtener los valores de "numcol" y darle una forma adecuada para el modelo

modelo = regresion.fit(colonies, data["totalprod"])  # Ajustar el modelo de regresión lineal con los datos de "colonies" en el eje x y "totalprod" en el eje y

print("Intersección (b)", modelo.intercept_)       # Imprimir el valor de la intersección (b)
print("Pendiente (m)", modelo.coef_)               # Imprimir el valor de la pendiente (m)

entrada = [[0], [100000], [200000], [300000],[400000],[500000]]  # Definir una lista de valores de entrada
modelo.predict(entrada)                           # Predecir los valores de salida para los datos de entrada

plt.scatter(entrada, modelo.predict(entrada), color="red")  # Crear un gráfico de dispersión con los valores de entrada y las predicciones en color rojo
plt.plot(entrada, modelo.predict(entrada), color="red", linewidth=3)  # Crear una línea que conecte los puntos en el gráfico
plt.ylabel("honey production (tons per month)")   # Establecer el título del eje y como "honey production (tons per month)"
plt.xlabel("number colonies")                     # Establecer el título del eje x como "number colonies"
plt.scatter(data["numcol"], data["totalprod"], color="blue", alpha=0.55)  # Crear un gráfico de dispersión con los datos originales en color azul y transparencia 0.55
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente

plt.ylabel("honey production")                    # Establecer el título del eje y como "honey production"
plt.xlabel("number colonies")                     # Establecer el título del eje x como "number colonies"
data["numcol"].plot(kind="bar", color="red", figsize=(10, 6))  # Crear un gráfico de barras con los datos de "numcol" en color rojo y tamaño de figura (10, 6)
data["totalprod"].plot(kind="line", color="blue", secondary_y=True)  # Crear una línea con los datos de "totalprod" en color azul y eje y secundario
plt.legend(["number colonies", "honey production"])  # Agregar leyendas al gráfico
plt.title("Honey production vs number of colonies")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente
import numpy as np                               # Importar la biblioteca numpy para cálculos numéricos

plt.axes(projection="polar")                      # Crear un gráfico polar

angle = np.linspace(0, 2*np.pi, len(data))         # Calcular los ángulos para cada punto en el gráfico polar
distance = data["totalprod"] / data["numcol"]      # Calcular las distancias para cada punto en el gráfico polar

plt.scatter(angle, distance, color="green")        # Crear un gráfico de dispersión en el gráfico polar con los ángulos y distancias calculados
plt.xlabel("angle")                               # Establecer el título del eje x como "angle"
plt.ylabel("distance")                            # Establecer el título del eje y como "distance"
plt.title("Spiral plot of honey production vs number of colonies")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente

data[["numcol", "totalprod"]].boxplot()           # Crear un gráfico de caja y bigotes con las dos columnas

plt.ylabel("value")                               # Establecer el título del eje y como "value"
plt.title("Box plot of number of colonies and honey production")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente
import seaborn as sns                            # Importar la biblioteca seaborn para visualización estadística

sns.heatmap(data[["numcol", "totalprod"]], annot=True)  # Crear un mapa de calor con las dos columnas

plt.xlabel("column")                              # Establecer el título del eje x como "column"
plt.ylabel("row")                                 # Establecer el título del eje y como "row"
plt.title("Heat map of number of colonies and honey production")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente

data[["numcol", "totalprod"]].plot(kind="bar")    # Crear un gráfico de columnas con las dos columnas

plt.xlabel("colonies")                            # Establecer el título del eje x como "colonies"
plt.ylabel("production")                          # Establecer el título del eje y como "production"
plt.title("Bar plot of number of colonies and honey production")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente

data[["numcol", "totalprod"]].plot(kind="line")   # Crear un gráfico de líneas con las dos columnas

plt.xlabel("row")                                 # Establecer el título del eje x como "row"
plt.ylabel("value")                               # Establecer el título del eje y como "value"
plt.title("Line plot of number of colonies and honey production")  # Establecer el título del gráfico
plt.show()                                        # Mostrar el gráfico

import matplotlib.pyplot as plt                  # Importar la biblioteca matplotlib nuevamente

data["numcol"].plot(kind="pie")                   # Crear un gráfico circular con la columna "numcol"

plt.ylabel("number colonies")                      # Establecer el título del eje y como "number colonies"
plt.title("Pie plot of number of colonies")        # Establecer el título del gráfico
plt.show()                                         # Mostrar el gráfico
```
# Escalamiento de Datos en Base a Atributos de Calidad de un Nuevo Producto Software

Breve descripción del proyecto.

Este proyecto tiene como objetivo realizar el escalamiento de datos utilizando atributos de calidad para evaluar prototipos de diseño de un nuevo producto software. Se aplicará un escalamiento de datos para normalizar las puntuaciones de los atributos de calidad y así obtener una comparación equitativa entre los prototipos.

## Requisitos

- Numpy
- Matplotlib
- Pandas
- Scikit-learn

## Instalación

Puedes instalar las dependencias del proyecto utilizando el siguiente comando:
```cmd
pip install numpy matplotlib pandas scikit-learn
```

## Uso
# Instalando Librerias
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```
# Generar datos aleatorios de evaluación de prototipos
```python
num_prototipos = 10
ergonomia = np.random.randint(1, 10, size=num_prototipos)
estetica = np.random.randint(1, 10, size=num_prototipos)
funcionalidad = np.random.randint(1, 10, size=num_prototipos)
innovacion = np.random.randint(1, 10, size=num_prototipos)
viabilidad = np.random.randint(1, 10, size=num_prototipos)
```
# Crear DataFrame con los datos
```python
df = pd.DataFrame({
    'ID del prototipo': range(1, num_prototipos + 1),
    'Ergonomía': ergonomia,
    'Estética': estetica,
    'Funcionalidad': funcionalidad,
    'Innovación': innovacion,
    'Viabilidad de producción': viabilidad
})
```
# Graficar evaluación de prototipos antes del escalamiento
```python
df.plot(x='ID del prototipo', y=['Ergonomía', 'Estética', 'Funcionalidad', 'Innovación', 'Viabilidad de producción'], kind='bar', stacked=True)
plt.xlabel('ID del prototipo')
plt.ylabel('Puntuación')
plt.title('Evaluación de prototipos de diseño (antes del escalamiento)')
plt.legend(loc='upper right')
plt.show()
```
# Aplicar escalamiento de datos a las columnas de evaluación
```python
scaler = MinMaxScaler()
df[['Ergonomía', 'Estética', 'Funcionalidad', 'Innovación', 'Viabilidad de producción']] = scaler.fit_transform(df[['Ergonomía', 'Estética', 'Funcionalidad', 'Innovación', 'Viabilidad de producción']])
```
# Graficar evaluación de prototipos después del escalamiento
```python
df.plot(x='ID del prototipo', y=['Ergonomía', 'Estética', 'Funcionalidad', 'Innovación', 'Viabilidad de producción'], kind='line')
plt.xlabel('ID del prototipo')
plt.ylabel('Puntuación escalada')
plt.title('Evaluación de prototipos de diseño (después del escalamiento)')
plt.legend(loc='upper right')
plt.show()
```
## Contribución

Si deseas contribuir a este proyecto, puedes seguir los siguientes pasos:

1. Haz un fork de este repositorio.
2. Crea una rama para tu contribución: `git checkout -b contribucion`.
3. Realiza las modificaciones y mejoras deseadas.
4. Realiza un commit de tus cambios: `git commit -m "Descripción de los cambios"`.
5. Haz push de tus cambios a tu repositorio: `git push origin contribucion`.
6. Crea una Pull Request en este repositorio.
7. Opcionalmente puedes enviarme un correo puedes accederlo: <a href="mailto:marlonstivenprod@ufps.edu.co?subject=Peticcion%20de%20acceso%20al%20proyecto&body=Hola%20quiero%20tener%20acceso%20a%20su%20repositorio,%20podrias%20hablar%20conmigo">contactame aqui</a>

## Licencia

Este proyecto se distribuye bajo la Licencia MIT.




