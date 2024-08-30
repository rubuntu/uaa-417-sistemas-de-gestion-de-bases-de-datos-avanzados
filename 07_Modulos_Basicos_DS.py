# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="bPXOH6rmDE0R"
# %matplotlib inline
import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 144

# %% id="9oArG3MyDE0S"
# %%capture
# !pip install expectexception
import expectexception

# %% [markdown] id="3oX7ZNnXDE0S"
# # Herramientas de datos básicas: NumPy, Matplotlib, Pandas
#
# Python es un lenguaje de programación potente y flexible, pero no tiene herramientas integradas para análisis matemático o visualización de datos. Para la mayoría de los análisis de datos, confiaremos en algunas bibliotecas útiles. Exploraremos tres bibliotecas que son muy comunes para el análisis y la visualización de datos.

# %% [markdown] id="PAeur3C0DE0T"
# ## NumPy
#
# El primero de ellos es NumPy. Las características principales de NumPy son tres: sus funciones matemáticas (por ejemplo, `sin`, `log`, `floor`), su submódulo `random` (útil para muestreo aleatorio) y el objeto NumPy `ndarray`.
#
# Una matriz NumPy es similar a una matriz matemática de n dimensiones. Por ejemplo,
#
# $$\begin{bmatrix}
#     x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
#     x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
#     \vdots & \vdots & \vdots & \ddots & \vdots \\
#     x_{d1} & x_{d2} & x_{d3} & \dots  & x_{dn}
# \end{bmatrix}$$
#
# Una matriz NumPy podría ser unidimensional (por ejemplo, [1, 5, 20, 34, ...]), bidimensional (como arriba) o muchas dimensiones. Es importante tener en cuenta que todas las filas y columnas de la matriz bidimensional tienen la misma longitud. Esto será válido para todas las dimensiones de las matrices.
#
# Comparemos esto con las listas.

# %% id="mH9KgY1IDE0T"
# to access NumPy, we have to import it
import numpy as np

# %% id="VawnMHpkDE0T" outputId="975d18b4-6d82-4749-b3d8-94d39dc40f15" colab={"base_uri": "https://localhost:8080/"}
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(list_of_lists)

# %% id="QSsJhJtQDE0T" outputId="ea41c87d-f3b2-4378-f3d2-0e19804309d3" colab={"base_uri": "https://localhost:8080/"}
an_array = np.array(list_of_lists)
print(an_array)

# %% id="u2fIdR3xDE0U" outputId="79b11517-69d4-42b0-ee26-3f63e403683d" colab={"base_uri": "https://localhost:8080/"}
non_rectangular = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
print(non_rectangular)

# %% id="CMVEBU-ADE0U" outputId="8367c7c5-56b1-46d7-a1ef-06078f77f0fb" colab={"base_uri": "https://localhost:8080/"}
non_rectangular_array = np.array(non_rectangular, dtype=object)
print(non_rectangular_array)


# %% [markdown] id="Ll2siQ62DE0U"
# ¿Por qué se imprimieron de manera diferente? Investiguemos su _forma_ y _tipo de datos_ (`dtype`).

# %% id="H4z5nAvRDE0U" outputId="98bbabe2-36b5-4048-9c69-02f0552e8b88" colab={"base_uri": "https://localhost:8080/"}
print(an_array.shape, an_array.dtype)
print(non_rectangular_array.shape, non_rectangular_array.dtype)

# %% [markdown] id="8m5QT_3uDE0U"
# El primer caso, `an_array`, es una matriz bidimensional de 3x3 (de números enteros). Por el contrario, `non_rectangular_array` es una matriz unidimensional de longitud 3 (de _objetos_, es decir, objetos de `lista`).
#
# También podemos crear una variedad de matrices con las funciones convenientes de NumPy.

# %% id="NXTsFkUbDE0V" outputId="0703f3ed-bbf4-4693-d143-477808ecc94a" colab={"base_uri": "https://localhost:8080/"}
np.linspace(1, 10, 10)

# %% id="jT_LEeGzDE0V" outputId="c6cb10d2-2783-40f6-97ae-d09aabaa0a57" colab={"base_uri": "https://localhost:8080/"}
np.arange(1, 10, 1)

# %% id="u6WDOYilDE0V" outputId="69c5d2ad-3221-4e64-f190-f75f578be686" colab={"base_uri": "https://localhost:8080/"}
np.logspace(1, 10, 10)

# %% id="EEYR5CFcDE0V" outputId="2aed38ee-2ba7-4a1e-d8ed-fb6fb024e8de" colab={"base_uri": "https://localhost:8080/"}
np.zeros(10)

# %% id="WpXvSVmxDE0V" outputId="fbbddc18-0723-4f8b-d229-029f7e20e4e0" colab={"base_uri": "https://localhost:8080/"}
np.diag([1,2,3,4])

# %% id="sQyfe2v2DE0V" outputId="2a8260a9-b524-4f3b-8c0d-97fe3d0914f2" colab={"base_uri": "https://localhost:8080/"}
np.eye(5)

# %% [markdown] id="TgZanZm2DE0V"
# También podemos convertir el `dtype` de una matriz después de su creación.

# %% id="5a-TJas2DE0V" outputId="2361886c-a4cf-418b-d8b7-fc818d78906a" colab={"base_uri": "https://localhost:8080/"}
print(np.logspace(1, 10, 10).dtype)
print(np.logspace(1, 10, 10).astype(int).dtype)

# %% [markdown] id="_aahQATnDE0V"
# ¿Por qué importa todo esto?
#
# Las matrices suelen ser más eficientes en términos de código y recursos computacionales para ciertos cálculos. Computacionalmente, esta eficiencia proviene del hecho de que preasignamos un bloque de memoria contiguo para los resultados de nuestro cálculo.
#
# Para explorar las ventajas del código, intentemos hacer algunos cálculos con estos números.
#
# Primero, simplemente calculemos la suma de todos los números y observemos las diferencias en el código necesario para `list_of_lists`, `an_array` y `non_rectangular_array`.

# %% id="dWiVmAPIDE0W" outputId="5b263c63-0ed9-4c9f-81c5-c467591446f0" colab={"base_uri": "https://localhost:8080/"}
print(sum([sum(inner_list) for inner_list in list_of_lists]))
print(an_array.sum())

# %% [markdown] id="6xUDi6CMDE0W"
# Sumar los números en una matriz es mucho más fácil que en una lista de listas. No tenemos que profundizar en una jerarquía de listas, simplemente usamos el método "suma" de "ndarray". ¿Esto todavía funciona para `non_rectangular_array`?

# %% id="uusGqf22DE0W" outputId="9bb46b46-3329-44c2-e9ca-fc477c958635" colab={"base_uri": "https://localhost:8080/"}
# what happens here?
print(non_rectangular_array.sum())

# %% [markdown] id="iHm_1zaaDE0W"
# Recuerde que `non_rectangular_array` es una matriz unidimensional de objetos de `lista`. El método `suma` intenta sumarlos: primera lista + segunda lista + tercera lista. La adición de listas da como resultado una _concatenación_.

# %% id="I_DjDF2DDE0W" outputId="d3ff2a8f-bfe8-4357-8a5a-b411190ec322" colab={"base_uri": "https://localhost:8080/"}
# concatenate three lists
print([1, 2] + [3, 4, 5] + [6, 7, 8, 9])

# %% [markdown] id="oj4TVPlVDE0W"
# El contraste se vuelve aún más claro cuando intentamos sumar filas o columnas individualmente.

# %% id="--rN7_xpDE0W" outputId="042e7bd0-61d3-4b7d-d325-60abf169efdc" colab={"base_uri": "https://localhost:8080/"}
print('Array row sums: ', an_array.sum(axis=1))
print('Array column sums: ', an_array.sum(axis=0))

# %% id="6P1G2jUmDE0W" outputId="e04db6bd-1b7f-4cdd-cf75-074868a2e5ca" colab={"base_uri": "https://localhost:8080/"}
print('List of list row sums: ', [sum(inner_list) for inner_list in list_of_lists])

def column_sum(list_of_lists):
    running_sums = [0] * len(list_of_lists[0])
    for inner_list in list_of_lists:
        for i, number in enumerate(inner_list):
            running_sums[i] += number

    return running_sums

print('List of list column sums: ', column_sum(list_of_lists))

# %% [markdown] id="ADFM4hFQDE0X"
# Generalmente es mucho más natural hacer operaciones matemáticas con matrices que con listas.

# %% id="xKEpXLByDE0X" outputId="1cf3686e-38a2-4e54-81ba-98d58ab6b1cf" colab={"base_uri": "https://localhost:8080/"}
a = np.array([1, 2, 3, 4, 5])
print(a + 5) # add a scalar
print(a * 5) # multiply by a scalar
print(a / 5.) # divide by a scalar (note the float!)

# %% id="2ZpsRLepDE0X" outputId="57e9e1ea-c485-40ad-d956-6832c0bd0347" colab={"base_uri": "https://localhost:8080/"}
b = a + 1
print(a + b) # add together two arrays
print(a * b) # multiply two arrays (element-wise)
print(a / b.astype(float)) # divide two arrays (element-wise)

# %% [markdown] id="mT56YJrsDE0X"
# Las matrices también se pueden utilizar para álgebra lineal, actuando como vectores, matrices, tensores, etc.

# %% id="E4uPkC4dDE0X" outputId="a847fba5-ca0a-4502-d2f0-3b1235ce5891" colab={"base_uri": "https://localhost:8080/"}
print(np.dot(a, b)) # inner product of two arrays
print(np.outer(a, b)) # outer product of two arrays

# %% [markdown] id="n-lj7FkeDE0X"
# Los arrays tienen mucho que ofrecernos en términos de representación y análisis de datos, ya que podemos aplicar fácilmente funciones matemáticas a conjuntos de datos o secciones de conjuntos de datos. La mayoría de las veces no tendremos ningún problema al usar matrices, pero es bueno tener en cuenta las restricciones en torno a la forma y el tipo de datos.
#
# Estas restricciones en torno a `shape` y `dtype` permiten que los objetos `ndarray` tengan un rendimiento mucho mayor en comparación con una `list` general de Python.  Hay algunas razones para esto, pero las dos principales resultan de la naturaleza escrita de `ndarray`, ya que esto permite el almacenamiento de memoria contigua y la búsqueda de funciones consistente.  Cuando se suma una "lista" de Python, Python necesita descubrir en tiempo de ejecución la forma correcta de agregar cada elemento de la lista.  Cuando se suma un `ndarray`, `NumPy` ya conoce el tipo de cada elemento (y son consistentes), por lo que puede sumarlos sin verificar la función de suma correcta para cada elemento.
#
# Veamos esto en acción haciendo algunos perfiles básicos.  Primero crearemos una lista de 100000 elementos aleatorios y luego cronometramos la función de suma.

# %% id="Z2EDUSLaDE0X"
time_list = [np.random.random() for _ in range(100000)]
time_arr = np.array(time_list)

# %% id="-7DtYUhzDE0X" outputId="0ed5b732-8795-4d5d-940e-c307482199a4" colab={"base_uri": "https://localhost:8080/"}
# %%timeit
sum(time_list)

# %% id="tBk9un3eDE0X" outputId="2241ba38-6ddf-4194-9e53-438cf85a7dff" colab={"base_uri": "https://localhost:8080/"}
# %%timeit
np.sum(time_arr)

# %% [markdown] id="n0doHKONDE0Y"
# ### Agregación de datos básicos
#
# Exploremos algunos ejemplos más del uso de matrices, esta vez usando el submódulo "aleatorio" de NumPy para crear algunos "datos falsos".

# %% id="gJvlxQQ9DE0c" outputId="0752503e-7213-4df4-eeeb-35c336267bdd" colab={"base_uri": "https://localhost:8080/"}
jan_coffee_sales = np.random.randint(25, 200, size=(4, 7))
print(jan_coffee_sales)

# %% id="pEtbLGUrDE0c" outputId="be004cbf-ece1-405c-fdc2-0839d3a9bc45" colab={"base_uri": "https://localhost:8080/"}
# mean sales
print('Mean coffees sold per day in January: %d' % jan_coffee_sales.mean())

# %% id="2436eCfiDE0c" outputId="a11f90e3-1a85-4d29-f562-b4e489c8f820" colab={"base_uri": "https://localhost:8080/"}
# mean sales for Monday
print('Mean coffees sold on Monday in January: %d' % jan_coffee_sales[:, 1].mean())

# %% id="1NtdfSJbDE0d" outputId="8ace9c17-6014-41d7-c491-6c318ba4070e" colab={"base_uri": "https://localhost:8080/"}
# day with most sales
# remember we count dates from 1, not 0!
print('Day with highest sales was January %d' % (jan_coffee_sales.argmax() + 1))

# %% id="1zxZna8pDE0d" outputId="c50c35b2-a991-4b3c-d64f-9f5e3f34422d" colab={"base_uri": "https://localhost:8080/"}
# is there a weekly periodicity?
normalized_sales = (jan_coffee_sales - jan_coffee_sales.mean()) / abs(jan_coffee_sales - jan_coffee_sales.mean()).max()
print(np.mean(np.arccos(normalized_sales) / (2 * np.pi) * 7, axis=0))

# %% [markdown] id="b4UxDDa2DE0d"
# Algunas de las funciones (`arccos` y `argmax`) que usamos anteriormente no existen en Python estándar y nos las proporciona NumPy. Además, vemos que podemos usar la forma de una matriz para ayudarnos a calcular estadísticas sobre un subconjunto de nuestros datos (por ejemplo, la cantidad media de cafés vendidos los lunes). Pero una de las cosas más poderosas que podemos hacer para explorar datos es simplemente visualizarlos.

# %% [markdown] id="UgpDf9CMDE0d"
# ### Cambiando de forma
#
# A menudo querremos tomar matrices que tengan una forma y transformarlas en una forma diferente que sea más adecuada para una operación específica.

# %% id="nD-OGorkDE0d"
mat = np.random.rand(20, 10)

# %% id="0tx9zMUHDE0d" outputId="c87d9118-b797-401b-a8f8-d3eb7aa3219b" colab={"base_uri": "https://localhost:8080/"}
mat.reshape(40, 5).shape

# %% id="WR3ci0N-DE0d" outputId="cc51d3d0-aef8-4f6e-b9d0-50982ea5107c" colab={"base_uri": "https://localhost:8080/"}
# %%expect_exception ValueError

mat.reshape(30, 5)

# %% id="TRTDe626DE0d" outputId="96236562-1d30-476f-d7bd-4b08a8f4946a" colab={"base_uri": "https://localhost:8080/"}
mat.ravel().shape

# %% id="vko0yFo-DE0e" outputId="f09899c8-452a-4ed7-cf4f-572f8d8e6000" colab={"base_uri": "https://localhost:8080/"}
mat.transpose().shape

# %% [markdown] id="-1nIoC61DE0e"
# ### Combinando matrices

# %% id="DoIYynYoDE0e" outputId="a317bb7c-15df-4a10-e20e-126b00cb65c8" colab={"base_uri": "https://localhost:8080/"}
print(a)
print(b)

# %% id="ial1Xy2vDE0e" outputId="2920e13f-0d37-40bf-bc2d-5af89827bb55" colab={"base_uri": "https://localhost:8080/"}
np.hstack((a, b))

# %% id="HX2KK1ltDE0e" outputId="1c2a0c99-c617-4091-9ab2-0f5d8f5fd3ac" colab={"base_uri": "https://localhost:8080/"}
np.vstack((a, b))

# %% id="b-OD4gXODE0e" outputId="cc66b840-dcd6-48f3-c02b-baaa6c8eb0ad" colab={"base_uri": "https://localhost:8080/"}
np.dstack((a, b))

# %% [markdown] id="CQTl9QdKDE0e"
# ### Funciones universales
#
# `NumPy` define un `ufunc` que le permite ejecutar funciones de manera eficiente sobre matrices.  Muchas de estas funciones están integradas, como `np.cos`, y se implementan en código `C` compilado de alto rendimiento.  Estas funciones pueden realizar "difusión", lo que les permite manejar automáticamente operaciones entre matrices de diferentes formas, por ejemplo, dos matrices con la misma forma, o una matriz y un escalar.

# %% [markdown] id="L3MMN7-8DE0f"
# ## Matplotlib
#
# Matplotlib es la biblioteca de trazado de Python más popular. Nos permite visualizar datos rápidamente al proporcionar una variedad de tipos de gráficos (por ejemplo, de barras, de dispersión, de líneas, etc.). También proporciona herramientas útiles para organizar múltiples imágenes o componentes de imágenes dentro de una figura, lo que nos permite crear visualizaciones más complejas según sea necesario.
#
# ¡Visualicemos algunos datos! En las siguientes celdas, generaremos algunos datos. Por ahora nos centraremos en cómo se producen los gráficos en lugar de cómo se generan los datos.

# %% id="XsZkQKG2DE0f"
import matplotlib.pyplot as plt


# %% id="IFN2QQryDE0f" outputId="1235bb33-4932-4ec4-a47e-77a60a671fa5" colab={"base_uri": "https://localhost:8080/", "height": 489}
def gen_stock_price(days, initial_price):
    # stock price grows or shrinks linearly
    # not exceeding 10% per year (heuristic)
    trend = initial_price * (np.arange(days) * .1 / 365 * np.random.rand() * np.random.choice([1, -1]) + 1)
    # noise will be about 2%
    noise = .02 * np.random.randn(len(trend)) * trend
    return trend + noise

days = 365
initial_prices = [80, 70, 65]
for price in initial_prices:
    plt.plot(np.arange(-days, 0), gen_stock_price(days, price))
plt.title('Stock price history for last %d days' % days)
plt.xlabel('Time (days)')
plt.ylabel('Price (USD)')
plt.legend(['Company A', 'Company B', 'Company C'])

# %% id="QQ4mG9iLDE0f" outputId="f052c275-d866-4b4e-a781-72cd46775b39" colab={"base_uri": "https://localhost:8080/", "height": 619}
from scipy.stats import linregress

def gen_football_team(n_players, mean_shoe, mean_jersey):
    shoe_sizes = np.random.normal(size=n_players, loc=mean_shoe, scale=.15 * mean_shoe)
    jersey_sizes = mean_jersey / mean_shoe * shoe_sizes + np.random.normal(size=n_players, scale=.05 * mean_jersey)

    return shoe_sizes, jersey_sizes

shoes, jerseys = gen_football_team(16, 11, 100)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Football team equipment profile')

ax1 = plt.subplot(221)
ax1.hist(shoes)
ax1.set_xlabel('Shoe size')
ax1.set_ylabel('Counts')

ax2 = plt.subplot(223)
ax2.hist(jerseys)
ax2.set_xlabel('Chest size (cm)')
ax2.set_ylabel('Counts')

ax3 = plt.subplot(122)
ax3.scatter(shoes, jerseys, label='Data')
ax3.set_xlabel('Shoe size')
ax3.set_ylabel('Chest size (cm)')

fit_line = linregress(shoes, jerseys)
ax3.plot(shoes, fit_line[1] + fit_line[0] * shoes, 'r', label='Line of best fit')

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles[::-1], labels[::-1])


# %% id="HaJl-aJIDE0f" outputId="07e15bfa-c38e-484c-c56c-96d0c1d1171f" colab={"base_uri": "https://localhost:8080/", "height": 489}
def gen_hourly_temps(days):
    ndays = len(days)
    seasonality = (-15 * np.cos((np.array(days) - 30) * 2.0 * np.pi / 365)).repeat(24) + 10
    solar = -3 * np.cos(np.arange(24 * ndays) * 2.0 * np.pi / 24)
    weather = np.interp(list(range(len(days) * 24)), list(range(0, 24 * len(days), 24 * 2)), 3 * np.random.randn(np.ceil(float(len(days)) / 2).astype(int)))
    noise = .5 * np.random.randn(24 * len(days))

    return seasonality + solar + weather + noise

days = np.arange(365)
hours = np.arange(days[0] * 24, (days[-1] + 1) * 24)
plt.plot(hours, gen_hourly_temps(days))
plt.title('Hourly temperatures')
plt.xlabel('Time (hours since Jan. 1)')
plt.ylabel('Temperature (C)')

# %% [markdown] id="VAAAEfuMDE0f"
# En los ejemplos anteriores hemos utilizado el omnipresente comando `plot`, `subplot` para organizar múltiples gráficos en una imagen y `hist` para crear histogramas. También hemos utilizado paradigmas de trazado tanto de "máquina de estados" (es decir, usando una secuencia de comandos `plt.method`) como "orientados a objetos" (es decir, creando objetos de figuras y mutándolos). El paquete Matplotlib es muy flexible y las posibilidades de visualizar datos están limitadas en su mayoría por la imaginación. Una excelente manera de explorar Matplotlib y otros paquetes de visualización de datos es consultando sus [páginas de galería](https://matplotlib.org/gallery.html).

# %% [markdown] id="no7lGGpgDE0f"
# # Pandas
#
# NumPy es útil para manejar datos, ya que nos permite aplicar funciones de manera eficiente a conjuntos de datos completos o seleccionar partes de ellos. Sin embargo, puede resultar difícil realizar un seguimiento de los datos relacionados que pueden estar almacenados en diferentes matrices, o del significado de los datos almacenados en diferentes filas o columnas de la misma matriz.
#
# Por ejemplo, en la sección anterior teníamos una matriz unidimensional para tallas de zapatos y otra matriz unidimensional para tallas de camisetas. Si quisiéramos buscar la talla de calzado y camiseta de un jugador en particular, tendríamos que recordar su posición en cada conjunto.
#
# Alternativamente, podríamos combinar las dos matrices unidimensionales para crear una matriz bidimensional con filas `n_players` y dos columnas (una para la talla de zapato, otra para la talla de camiseta). Pero una vez que combinamos los datos, ahora tenemos que recordar qué columna es la talla de zapato y qué columna es la talla de camiseta.
#
# El paquete Pandas presenta una herramienta muy poderosa para trabajar con datos en Python: el DataFrame. Un DataFrame es una tabla. Cada columna representa un tipo diferente de datos (a veces llamado **campo**). Las columnas tienen nombre, por lo que podría tener una columna llamada `'shoe_size'` y una columna llamada `'jersey_size'`. No tengo que recordar qué columna es cuál, porque puedo referirme a ellas por su nombre. Cada fila representa un **registro** o **entidad** diferente (por ejemplo, jugador). También puedo nombrar las filas, de modo que en lugar de recordar qué fila de mi matriz corresponde a Ronaldinho, puedo nombrar la fila 'Ronaldinho' y buscar su talla de zapato y su talla de camiseta por nombre.

# %% id="DKC_BL_sDE0g" outputId="9d85b955-96ba-4f52-a1a1-7a029bb2d25d" colab={"base_uri": "https://localhost:8080/", "height": 488}
import pandas as pd

players = ['Ronaldinho', 'Pele', 'Lionel Messi', 'Zinedine Zidane', 'Didier Drogba', 'Ronaldo', 'Yaya Toure',
           'Frank Rijkaard', 'Diego Maradona', 'Mohamed Aboutrika', "Samuel Eto'o", 'George Best', 'George Weah',
           'Roberto Donadoni']
shoes, jerseys = gen_football_team(len(players), 10, 100)

df = pd.DataFrame({'shoe_size': shoes, 'jersey_size': jerseys}, index = players)

df

# %% id="I0WvXSwxDE0g" outputId="3a1bd3fc-e591-4951-96ea-e0e0467d1ff0" colab={"base_uri": "https://localhost:8080/", "height": 488}
# we can also make a dataframe using zip

df = pd.DataFrame(list(zip(shoes, jerseys)), columns = ['shoe_size', 'jersey_size'], index = players)

df

# %% [markdown] id="L_egRelkDE0g"
# El DataFrame tiene similitudes tanto con un "dict" como con un "ndarray" de NumPy. Por ejemplo, podemos recuperar una columna del DataFrame usando su nombre, tal como recuperaríamos un elemento de un "dict" usando su clave.

# %% id="nE5a1hv6DE0g" outputId="983a73fd-d9eb-4815-9e64-9eee297f570f" colab={"base_uri": "https://localhost:8080/", "height": 523}
df['shoe_size']

# %% [markdown] id="kw7PgSdJDE0g"
# Y podemos aplicar funciones fácilmente al DataFrame, tal como lo haríamos con una matriz NumPy.

# %% id="aREqASDsDE0g" outputId="ee65c179-61a0-437f-ea78-504585a9624e" colab={"base_uri": "https://localhost:8080/", "height": 488}
np.log(df)

# %% id="bxBTHuS6DE0h" outputId="75945f8e-18ad-4c68-a94f-ea407ef3c7d4" colab={"base_uri": "https://localhost:8080/", "height": 147}
df.mean()

# %% [markdown] id="zVwiiC9BDE0h"
# Exploraremos la aplicación de funciones y el análisis de datos en un DataFrame con más profundidad más adelante. Primero necesitamos saber cómo recuperar, agregar y eliminar datos de un DataFrame.
#
# Ya hemos visto cómo recuperar una columna, ¿qué pasa con recuperar una fila? La sintaxis más flexible es utilizar el método `loc` del DataFrame.

# %% id="k_SajZUoDE0h" outputId="f9d30083-c281-4091-b29a-c7b15847c007" colab={"base_uri": "https://localhost:8080/", "height": 147}
df.loc['Ronaldo']

# %% id="v5GuLxFCDE0h" outputId="845e33e1-c14e-44be-9e9e-69c691a3e66c" colab={"base_uri": "https://localhost:8080/", "height": 147}
df.loc[['Ronaldo', 'George Best'], 'shoe_size']

# %% id="VXd9KAuCDE0h" outputId="2bdfab92-cae4-4031-b0c0-bdc35df7b650" colab={"base_uri": "https://localhost:8080/", "height": 303}
# can also select position-based slices of data
df.loc['Ronaldo':'George Best', 'shoe_size']

# %% id="pdY63ELKDE0h" outputId="b5185cc8-5842-43a0-e572-bb14f392e458" colab={"base_uri": "https://localhost:8080/", "height": 206}
# for position-based indexing, we will typically use iloc
df.iloc[:5]

# %% id="cC9EYFuuDE0h" outputId="9a4c7bf1-de5b-4d17-f371-8959a41f3f14" colab={"base_uri": "https://localhost:8080/", "height": 147}
df.iloc[2:4, 0]

# %% id="q2HC-ZXXDE0h" outputId="9aa09355-93db-4a9f-b5e1-d170660dc8c8" colab={"base_uri": "https://localhost:8080/", "height": 206}
# to see just the top of the DataFrame, use head
df.head()

# %% id="BjLWjz4FDE0i" outputId="ad3501fb-bc0e-4888-e876-6d6b491af5cd" colab={"base_uri": "https://localhost:8080/", "height": 206}
# of for the bottom use tail
df.tail()

# %% [markdown] id="2VX_cQxKDE0i"
# Al igual que con un `dict`, podemos agregar datos a nuestro DataFrame simplemente usando la misma sintaxis que usaríamos para recuperar datos, pero combinándolos con una asignación.

# %% id="wNWzK_y6DE0i" outputId="3ab3d98f-a3d1-486c-bea6-e09f59e09955" colab={"base_uri": "https://localhost:8080/", "height": 206}
# adding a new column
df['position'] = np.random.choice(['goaltender', 'defense', 'midfield', 'attack'], size=len(df))
df.head()

# %% id="KrNdQuyZDE0i" outputId="d966eb83-fcbd-472e-9bf2-16349074dc01" colab={"base_uri": "https://localhost:8080/", "height": 178}
# adding a new row
df.loc['Dylan'] = {'jersey_size': 91, 'shoe_size': 9, 'position': 'midfield'}
df.loc['Dylan']

# %% [markdown] id="zjthIWd2DE0i"
# Para eliminar datos, podemos usar el método `drop` del DataFrame.

# %% id="NIEjFX8EDE0i" outputId="fece7852-d538-4fec-a4b8-30e664f0be68" colab={"base_uri": "https://localhost:8080/", "height": 488}
df.drop('Dylan')

# %% id="8YHEqoyfDE0i" outputId="002ef503-0a48-4878-f2f3-8ddac54c5abf" colab={"base_uri": "https://localhost:8080/", "height": 519}
df.drop('position', axis=1)

# %% [markdown] id="VeyEZNlqDE0j"
# Observe que cuando ejecutamos `df.drop('position', axis=1)`, había una entrada para `Dylan` aunque acabábamos de ejecutar `df.drop('Dylan')`. Tenemos que tener cuidado al usar `drop`; muchas funciones de DataFrame devuelven una _copia_ del DataFrame. Para que el cambio sea permanente, necesitamos reasignar `df` a la copia devuelta por `df.drop()` o tenemos que usar la palabra clave `inplace`.

# %% id="nkEG7zs5DE0j" outputId="678568f1-89a3-4f1d-f968-33b19f672517" colab={"base_uri": "https://localhost:8080/", "height": 488}
df = df.drop('Dylan')
df

# %% id="XrgEVndpDE0j" outputId="503da8d2-45f4-414b-93f0-eb5da1b6971d" colab={"base_uri": "https://localhost:8080/", "height": 488}
df.drop('position', axis=1, inplace=True)
df

# %% [markdown] id="hHsirJUqDE0j"
# Exploraremos Pandas con mucho más detalle más adelante en el curso, ya que tiene muchas herramientas poderosas para el análisis de datos. Sin embargo, incluso con estas herramientas ya puedes empezar a descubrir patrones en los datos y sacar conclusiones interesantes.
