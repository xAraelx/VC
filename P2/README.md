# Práctica 2

Este repositorio contiene el cuaderno `VC_P2.ipynb`, con tres ejercicios sobre detección de bordes (Canny y Sobel) y una aplicación de control de audio por gestos.

### 👥 Autores:
- [Sergio Rodriguez Rubio](https://github.com/SergioRodriguezEnt)
- [Arael Jesús Almeida González](https://github.com/xAraelx)

------

## Archivos necesarios

- `mandril.jpg` — imagen utilizada en los tres ejercicios.
- `musica.mp3`  — musica que se podrá controlar en la tarea 3 

## Recursos necesarios

Se requieres hacer una instalacion de los paquetes `sounddevice` y `soundfile`

- `%pip install sounddevice soundfile`

RECORDATORIO: Ya se encuentra dentro del cuaderno una celda donde poder ejecutar la instaliacion de dichos paquetes.

## Creditos de la musica

Musica usada We Shop Song - Philip Milman: 
Creative Commons ► Attribution 3.0 Unported ► CC BY 3.0
https://creativecommons.org/licenses/...
"You are free to use, remix, transform, and build upon the material
for any purpose, even commercially. You must give appropriate credit."

Composed by
Philip Milman ► https://pmmusic.pro/

## Tarea 1 — Conteo de píxeles blancos por filas con Canny

En esta primera tarea, la imagen se convierte a escala de grises y se le aplica `cv2.Canny(gris, 100, 200)`, obteniendo una imagen binaria (0/255) con los bordes detectados.

## Tarea 2 — Umbralizado de Sobel y comparación con Canny

Una vez ya se conocen los resultados obtenidos con canny, en esta tarea se busca comparar dichos resultado con los que daría otro método de detección de bordes, en este caso el llamado Sobel.


## Tarea 3 — Demostradores de visión en tiempo real: movimiento, color de piel y rostros
 
La tarea se compone de una función común y dos demostradores independientes construidos sobre ella, ambos basados en detección de movimiento sobre color de piel: uno con efecto de burbujas, inspirado en *Messa di voce*, y otro con efecto de estela.
 
