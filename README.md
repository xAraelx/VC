En este repositorio se encontrarán distintas tareas prácticas de **Visión por Computador**, con el objetivo de aprender acerca
de Anaconda, Python y la librería OpenCV.

### Participantes de este repositorio:
- [Sergio Rodriguez Rubio](https://github.com/SergioRodriguezEnt)
- [Arael Jesús Almeida González](https://github.com/xAraelx)

------

### [PRÁCTICA 1](https://github.com/xAraelx/VC/tree/main/P1)

[Guía Practica 1](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P1).

En esta primera práctica se encontrará 4 tareas distintas, todas ellas desarrolladas dentro de un mismo cuaderno Jupyter.

Estructura de la carpeta
```
P1
|- README.md             #Breve descripción de la solucion a las tareas
|- VC_P1.ipynb           #Cuaderno jupyter con tareas desarrolladas
```


### [PRÁCTICA 2](https://github.com/xAraelx/VC/tree/main/P2)

[Guía Practica 2](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P2)

En esta segunda práctica se encontrará 3 tareas distintas, todas ellas desarrolladas dentro de un mismo cuaderno Jupyter.
Además del cuaderno Jupyter, en la carpeta de esta práctica se encuentra la imagen usada para la detección de bordes con Canny y Sobel, y la canción que se controla con las manos en la última tarea.

Para la realización de dichas tareas, se incluye en la carpeta de esta práctica un README donde se desarrolla el proceso de realización de cada tarea.

Estructura de la carpeta
```
P2
|- AmpliacionTarea2_P2.mp4    #Video demostracion de la ampliacion de la tarea 2
|- mandril.jpg                #Imagen base para las tareas 1 y 2
|- musica.mp3                 #Música controlada en la tarea 3
|- README.md                  #Descripción de la solución a las tareas
|- ResusltadoTarea1_P2.png    #Imagen del resultado de la tarea 1
|- ResusltadoTarea2_P2.png    #Imagen del resultado de la tarea 2
|- ResusltadoTarea3_P2.mp4    #Video del resultado de la tarea 3
|- VC_P2.ipynb                #Cuaderno jupyter con tareas desarrolladas
```


### [PRÁCTICA 3](https://github.com/xAraelx/VC/tree/main/P3)

[Guía Practica 3](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3)

En esta tercera práctica se encuentran 2 tareas de detección y reconocimiento de formas, desarrolladas dentro de un mismo cuaderno Jupyter y sin usar modelos de aprendizaje profundo:
- **Tarea 1**: contar el dinero (monedas y, como extensión, billetes de euro) de una imagen y en vivo con la cámara.
- **Tarea 2**: clasificar microplásticos (pellets, alquitrán y fragmentos) y evaluar el resultado con la imagen de test, sus métricas y la matriz de confusión.

Además del cuaderno, en la carpeta se encuentran las imágenes de entrenamiento y de test, las imágenes de los resultados y los gifs de las demostraciones en vivo.

Para la realización de dichas tareas, se incluye en la carpeta de esta práctica un README donde se desarrolla el proceso de realización de cada tarea.

Estructura de la carpeta
```
P3
|- Billetes.gif                       #Demostración en vivo de la detección de billetes
|- Billetes.png                       #Imagen esquema con billetes y monedas para la tarea 1
|- DeteccionBilletesMonedasImg.png    #Resultado de la tarea 1 sobre Billetes.png
|- DeteccionMicroPlasticos.png        #Resultado de la tarea 2 sobre MPs_test.jpg
|- FRA.png, PEL.png, TAR.png          #Subimágenes de entrenamiento de cada clase
|- *-03-olympus-10-01-2020.JPG        #Imágenes de entrenamiento (fragment, pellet, tar)
|- MatrizConfu.png                    #Ejemplo de matriz de confusión del enunciado
|- MatrizDeConfucion.png              #Matriz de confusión obtenida en la tarea 2
|- Monedas.jpg                        #Imagen ideal de monedas del enunciado
|- Monedas1.gif, Monedas2.gif         #Demostraciones en vivo del contador de monedas
|- MPs_test.jpg                       #Imagen de test de la tarea 2
|- MPs_test_bbs.csv                   #Anotaciones de la imagen de test
|- README.md                          #Descripción de la solución a las tareas
|- VC_P3.ipynb                        #Cuaderno jupyter con tareas desarrolladas
```