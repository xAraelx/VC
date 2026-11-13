# Práctica 4 y 4b

Este repositorio contiene el cuaderno `VC_P4.ipynb`, que reúne las dos partes de la práctica 4:

- **P4 — Detección:** un prototipo que procesa vídeo para detectar y seguir personas y vehículos, localizar sus matrículas y contarlos. El detector de matrículas se ha entrenado de tres formas distintas: YOLO11n, RT-DETR con Ultralytics y RT-DETR con el repositorio oficial, sin Ultralytics.
- **P4b — OCR:** lee el texto de las matrículas y compara un OCR clásico (Tesseract) con un modelo de lenguaje visual (PaddleOCR-VL).

### Autores:
- [Sergio Rodriguez Rubio](https://github.com/SergioRodriguezEnt)
- [Arael Jesús Almeida González](https://github.com/xAraelx)

------

## Índice

1. [Estructura de la carpeta](#estructura-de-la-carpeta)
2. [Recursos necesarios](#recursos-necesarios)
3. [Tarea 1 — Detección y seguimiento](#tarea-1--detección-y-seguimiento-de-personas-vehículos-y-matrículas)
   - [Dataset](#dataset-de-matrículas)
   - [Entrenamiento](#entrenamiento)
   - [RT-DETR sin Ultralytics](#rt-detr-sin-ultralytics-extra)
   - [Comparativa de detectores](#comparativa-de-los-tres-detectores-extra)
   - [Pipeline de vídeo](#pipeline-de-vídeo)
   - [Los tres detectores en el vídeo](#los-tres-detectores-en-el-vídeo)
   - [Flujo](#extra-flujo-por-direcciones)
   - [Anonimización](#extra-anonimización)
4. [Tarea 2 — OCR](#tarea-2--reconocimiento-de-caracteres-en-las-matrículas)
   - [OCR](#ocr)
   - [Comparativa Tesseract vs PaddleOCR-VL](#comparativa-tesseract-vs-paddleocr-vl)
   - [Los dos OCR en el vídeo](#los-dos-ocr-en-el-vídeo)
5. [Referencias](#referencias)

------

## Estructura de la carpeta

Los ficheros marcados con **(Drive)** no están en el repositorio por su tamaño. Están en el [DRIVE](https://drive.google.com/drive/folders/1z3AbIQVJxg9jMuxpYFPb0yNetClAjz53?usp=sharing) con la misma estructura de carpetas, así que basta con copiarlos dentro de `P4/`.

```
P4
|- VC_P4.ipynb                     #Cuaderno jupyter con las tareas desarrolladas
|- README.md                       #Descripción de la solución a las tareas
|- load_rtdetr.py                  #Carga de los modelos del repositorio oficial de RT-DETR
|- tracker_general.yaml            #Configuración del tracker de personas y vehículos (BoT-SORT)
|- tracker_matriculas.yaml         #Configuración del tracker de matrículas (ByteTrack)
|- mandril.jpg                     #Imagen con la que se tapan personas y matrículas (copiar de ../P2/mandril.jpg)
|- RT-DETR/                        #Repositorio oficial de RT-DETR (se clona aparte, ver Recursos necesarios)
|- datos/                          #Dataset de matrículas (enlace en el apartado Dataset)
|   |- images/, annotations/       #Imágenes y anotaciones originales en JSON (caja + texto de la matrícula)
|   |- labels/                     #Etiquetas en formato YOLO (las genera la sección 1)
|   |- data.yaml, train.txt, val.txt
|   |- coco_train.json, coco_val.json  #El mismo reparto en formato COCO, para el repositorio oficial
|- weights/
|   |- rtdetrv2_r50vd_6x_coco_ema.pth  #(Drive) Pesos de COCO del repositorio oficial
|- runs/
|   |- detect/train-3/             #Entrenamiento de YOLO11n: curvas y args.yaml; weights/ (Drive)
|   |- detect/rtdetr-2/            #Entrenamiento de RT-DETR-l con Ultralytics: curvas y args.yaml; weights/ (Drive)
|   |- rtdetr_nativo/              #Entrenamiento de RT-DETRv2-L sin Ultralytics: log.txt, results.png y config
|       |- best.pth, best_ligero.pth, last.pth   #(Drive) Checkpoints; best_ligero = pesos EMA en FP16
|- results/
    |- <detector>_<ocr>.mp4        #(Drive) Vídeo anotado de cada ejecución
    |- <detector>_<ocr>_anonimo.mp4 #(Drive) Vídeo anonimizado
    |- <detector>_<ocr>_detecciones.csv #Detecciones y seguimiento de cada fotograma
    |- <detector>_<ocr>_totales.csv     #Total de objetos únicos por clase
    |- <detector>_<ocr>_matriculas.csv  #Texto leído de cada matrícula, votos y si está confirmada
    |- <detector>_<ocr>_flujo.csv       #Salidas por dirección de cada clase
    |- yolo_paddle.gif, yolo_paddle_anonimo.gif  #Fragmentos de los vídeos para este README
    |- comparativa_detectores/     #Tablas, gráficas y curvas de los tres detectores
    |- comparativa_ocr/            #Tablas, gráficas y lecturas de Tesseract vs PaddleOCR-VL
```

Ejecuciones guardadas en `results/`: `yolo_paddle`, `yolo_tesseract`, `rtdetr_paddle`, `rtdetr_tesseract` y `rtdetr_nativo_paddle`.

El vídeo de test (`C0142.MP4`) no se incluye. La imagen del mandril está en `P2/mandril.jpg` y hay que copiarla en `P4/` antes de ejecutar el cuaderno. Los pesos de partida de Ultralytics (`yolo11n.pt`, `yolo11s.pt`, `rtdetr-l.pt`) se descargan solos.

## Recursos necesarios

Se ha usado un *environment* `VC_P4` con Python 3.11:

- `pip install ultralytics lap pandas psutil "supervision<0.31"` (`supervision` 0.31 ya no incluye `ByteTrack`)
- `pip install pytesseract` y el programa [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (en Windows se busca en `C:\Program Files\Tesseract-OCR`)
- `pip install "transformers>=5.0.0" pillow` para PaddleOCR-VL
- Para el RT-DETR sin Ultralytics: clonar el [repositorio oficial](https://github.com/lyuwenyu/RT-DETR) en la carpeta `RT-DETR/` e instalar `pip install tensorboard faster_coco_eval thop` (`thop` solo para contar los GFLOPs)
- PyTorch con CUDA ([instrucciones](https://pytorch.org/get-started/locally/)): sin GPU funciona, pero PaddleOCR-VL y RT-DETR son mucho más lentos.

### Equipo de pruebas

Todos los tiempos de este README se han medido en este equipo:

| Componente | Valor |
|---|---|
| SO | Windows 11 |
| CPU | AMD Ryzen 7 7735HS |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| Python | 3.11.18 |
| PyTorch | 2.6.0 (CUDA 12.4) |
| Ultralytics | 8.4.190 |

------

## Tarea 1 — Detección y seguimiento de personas, vehículos y matrículas

### Dataset de matrículas

Los modelos preentrenados no detectan matrículas, así que se ha entrenado un detector propio con el dataset de matrículas creado entre los alumnos de Visión por Computador (ULPGC, curso 2026/27): **1976 imágenes y 4511 matrículas**. Cada imagen tiene un JSON con la caja de cada matrícula (centro y tamaño normalizados) y su texto (campo `plate`), que se usa después en la comparativa de OCR. Las matrículas ilegibles tienen como texto `"None"`.

La sección 1 del cuaderno convierte las anotaciones al formato de YOLO y reparte las imágenes al azar (semilla 42): **1581 imágenes (3584 matrículas) para entrenamiento y 395 (927 matrículas) para validación**. Los tres detectores usan exactamente este reparto; para el repositorio oficial se exporta además a formato COCO.

Enlace al dataset: [DRIVE](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI)

<details>
<summary>Autoría del dataset</summary>

- **Organización y creación del dataset:** Alejandra Rodríguez Silva
- **Script para anotar:** José Marcial Galván Franco y Solmaire Basulto Santana
- **Anotaciones:** Arael Jesús Almeida González, Juan Francisco Almeida Santana, Eduardo Arbelo Rua-Figueroa, Solmaire Basulto Santana, Ubay Antonio Batista Santana, Texenery Bordón Rodríguez, Pablo Campos Rico, Dylan A. Castrillón Suárez, Pablo Damas Negrín, Vidal de León Giménez, Naiara Díaz Hernández, Alejandro David Fariña Afonso, José Marcial Galván Franco, Ricardo García Rodríguez, Álvaro García Suárez, Carla Gómez García, David González Espino, Marcos González Gómez, Néstor Jesús Henríquez Medina, Juan Daniel López Melainine, Iván Luján Moreno, Alejandro José Martel Torres, Félix José Miguel Velásquez, Carlos Luis Miranda Hernández, Mencey Montesdeoca Álamo, Yenedey Morán Delgado, José Miguel Ojeda Hernández, Alba Ramos Quintana, Ángel Sergio Reyes Guedes, Roberto Rodríguez Guillén, Sergio Rodríguez Rubio, Alejandra Rodríguez Silva, Alicia María Rodríguez Trujillo, Carmelo Aidan Romero Pérez, Kilian Santana Delgado, Cristina Santana Martín, Erick Justo Sosa, Enrique Sosa Ojeda y Cristian Tomás Vega Appelqvist.

</details>

### Entrenamiento

Los tres detectores de matrículas parten de pesos preentrenados en COCO, se entrenan a 640 px y se quedan con la época con mejor resultado en validación.

| Modelo | Implementación | Base | Parámetros | Optimizador | Batch | Épocas | Mejor época | Tiempo de entrenamiento |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | Ultralytics | `yolo11n.pt` | 2.6 M | Automático de Ultralytics | 16 | 100 | 88 | ~2 h |
| RT-DETR-l | Ultralytics | `rtdetr-l.pt` | 32.8 M | AdamW, lr 1e-4 | 8 | 100 | 96 | ~3 h |
| RT-DETRv2-L (ResNet-50) | Repositorio oficial | `rtdetrv2_r50vd_6x_coco_ema.pth` | 49.8 M* | AdamW (config del repo) | 4 | 60 de 100 (parado a mano) | 58 | ~7 h |

\* Parámetros contados en el modelo cargado en el cuaderno; el artículo de RT-DETRv2 da 42 M para este modelo. No se ha investigado de dónde sale la diferencia.

El entrenamiento del RT-DETR nativo se paró a mano en la época 60 de las 100 previstas por el tiempo que llevaba. Por época tardó unos 7 minutos, frente a menos de 2 del RT-DETR de Ultralytics. Las causas probables son:
- carga las imágenes sin *workers* (en Windows harían falta en un script aparte), así que la GPU espera a la CPU en cada lote;
- usa un *batch* más pequeño;
- es un modelo algo mayor.

YOLO11n es el más rápido de entrenar con diferencia: su tamaño le permite usar un *batch* de 16.

*Data augmentation*:
- **YOLO11n:** más variación de saturación y brillo que la de por defecto (día, noche, sombras), rotación de ±5°, cizalla y perspectiva leves, *mixup* suave y **sin volteo horizontal**, porque el texto de una matrícula nunca aparece invertido. Mantiene el *mosaic* de Ultralytics.
- **RT-DETRv2-L (nativo):** las mismas transformaciones de color y geometría que YOLO, con las de `torchvision`, sin volteo, sin *mosaic* y sin *mixup*.
- **RT-DETR-l (Ultralytics):** el *data augmentation* por defecto de Ultralytics (que sí incluye volteo horizontal).

Los dos RT-DETR se entrenan a 640 px, la resolución de su preentrenamiento, con AdamW y una tasa de aprendizaje baja, que es la receta habitual de los modelos tipo DETR.

**Evolución del mAP de validación de los tres modelos:**

![Curvas de entrenamiento](results/comparativa_detectores/curvas_entrenamiento.png)

El mAP por época del RT-DETR nativo lo calcula el repositorio con COCOeval, y el de los otros dos el validador de Ultralytics. Por eso estas curvas sirven para ver la tendencia; los valores comparables son los de la tabla de la comparativa.

- **RT-DETRv2-L (nativo):** es el que más rápido aprende. Tras la primera época ya tiene un mAP50 de 0,41 y en la época 8 pasa de 0,70. En la época 23 alcanza el 95 % de su mejor mAP50-95 y a partir de ahí casi no mejora (0,407 en la época 30 y 0,415 en la 58). Parar en la época 60 apenas le ha costado precisión.
- **RT-DETR-l (Ultralytics):** las primeras 8 épocas está casi a cero y tiene caídas bruscas hasta la 20. Desde ahí supera a YOLO y sigue mejorando poco a poco hasta el final (mejor época: 96), así que con más épocas podría ganar algo más.
- **YOLO11n:** sube de forma constante, sin saltos. Alcanza el 95 % de su máximo en la época 68 y su mejor época es la 88; las últimas 20 épocas son prácticamente planas.
- **Sobreajuste:** no se observa en ninguno. Las pérdidas de validación de YOLO11n y RT-DETR-l bajan hasta la última época (por ejemplo, la `val/box_loss` de YOLO pasa de 1,66 en la época 25 a 1,60 en la 100) y el mAP de validación no empeora al final.

<details>
<summary>Curvas completas de cada entrenamiento (pérdidas y métricas)</summary>

**YOLO11n:**
![Entrenamiento YOLO](runs/detect/train-3/results.png)

**RT-DETR-l (Ultralytics):**
![Entrenamiento RT-DETR](runs/detect/rtdetr-2/results.png)

**RT-DETRv2-L (nativo):**
![Entrenamiento RT-DETR nativo](runs/rtdetr_nativo/results.png)

</details>

### RT-DETR sin Ultralytics (extra)

Ultralytics usa licencia AGPL-3.0, que obliga a publicar el código de cualquier aplicación que lo use, mientras que RT-DETR se publicó con licencia Apache 2.0. Por eso, además de la versión de Ultralytics, se ha usado RT-DETR directamente desde su [repositorio oficial](https://github.com/lyuwenyu/RT-DETR): **RT-DETRv2-L** con *backbone* ResNet-50 (sección 2b).

- **Datos:** el repositorio solo lee COCO, así que se exporta a `coco_train.json` y `coco_val.json` el mismo reparto de la sección 1, sin mover imágenes. Al cargar las imágenes se aplica la orientación EXIF, igual que hacen OpenCV y Ultralytics.
- **Entrenamiento:**
  - Se usa el *solver* del repositorio, configurado desde el cuaderno (la configuración final queda en `runs/rtdetr_nativo/rtdetrv2_matriculas.yml`).
  - Se parte de los pesos de COCO y se descarta la cabeza de 80 clases para dejar una sola clase.
  - Se entrena con precisión mixta (AMP), media exponencial de los pesos (EMA) y 3 épocas de calentamiento.
  - El *data augmentation* es equivalente al de YOLO, con las transformaciones de `torchvision`.
  - Sin *workers*, porque en Windows necesitarían un script aparte.
  - Se paró a mano en la época 60 de 100 (unas 7 h). El mejor modelo según el mAP50-95 de validación hasta ese momento queda en `runs/rtdetr_nativo/best.pth`.
- **Integración:** la clase `RTDETRNativo` le da al modelo la misma interfaz que usa el resto del cuaderno (`.names` y `.track()`), así que la comparativa y el pipeline lo usan igual que a los de Ultralytics. Para el modelo general de personas y vehículos se usan los pesos oficiales de COCO, sin reentrenar. Como el repositorio no trae tracking, se usa `ByteTrack` de la librería `supervision`, con los mismos parámetros que los `.yaml` de Ultralytics.
- **Evaluación justa:** las métricas se calculan con las mismas funciones que `model.val()` de Ultralytics (emparejado por IoU de 0,5 a 0,95 y la misma curva precisión-recall) y se guardan las mismas gráficas. Para comparar tamaños, se cuentan solo los pesos EMA en FP16 (`best_ligero.pth`), que es lo que guarda Ultralytics en `best.pt`.

### Comparativa de los tres detectores (extra)

La sección 3 evalúa los tres detectores en las mismas condiciones:
- las 395 imágenes de validación, a 640 px y en el mismo equipo;
- tiempos con *batch* 1, como en el vídeo, después de 10 predicciones de calentamiento;
- la lectura del disco queda fuera de la medida.

Además, calcula para cada modelo el umbral de confianza que maximiza F1, que se usa como `conf_matricula` en el pipeline: cada modelo calibra sus confianzas de forma distinta.

**Detección:**

| Modelo | Precision | Recall | F1 máx. | mAP50 | mAP50-95 | Conf. F1 máx. |
|---|---|---|---|---|---|---|
| YOLO11n | 0.879 | 0.643 | 0.743 | 0.672 | 0.352 | 0.344 |
| RT-DETR-l (Ultralytics) | **0.904** | 0.672 | 0.777 | 0.720 | 0.402 | 0.642 |
| RT-DETRv2-L (nativo) | 0.887 | **0.727** | **0.799** | **0.781** | **0.419** | 0.581 |

**Velocidad y tamaño** (*batch* 1, ms por imagen):

| Modelo | Preproceso | Inferencia | Postproceso | Total | FPS | Parámetros (M) | GFLOPs | Tamaño (MB) |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | 3.4 | **26.1** | 2.1 | **32.2** | **31.1** | **2.6** | **6.6** | **5.5** |
| RT-DETR-l (Ultralytics) | 3.1 | 54.5 | 1.0 | 59.2 | 16.9 | 32.8 | 109.9 | 66.2 |
| RT-DETRv2-L (nativo) | 59.6 | 64.0 | 1.5 | 125.4 | 8.0 | 49.8 | 133.7 | 86.0 |

Tablas generadas por el cuaderno: [`comparativa.md`](results/comparativa_detectores/comparativa.md) y [`comparativa.csv`](results/comparativa_detectores/comparativa.csv).

![Métricas de detección](results/comparativa_detectores/metricas.png)

![Tiempos de los detectores](results/comparativa_detectores/tiempos.png)

<details>
<summary>Curvas de validación y matrices de confusión de cada modelo</summary>

| | YOLO11n | RT-DETR-l (Ultralytics) | RT-DETRv2-L (nativo) |
|---|---|---|---|
| Precisión-recall | ![PR YOLO](results/comparativa_detectores/val_YOLO11n/BoxPR_curve.png) | ![PR RT-DETR](results/comparativa_detectores/val_RT-DETR-l/BoxPR_curve.png) | ![PR nativo](results/comparativa_detectores/val_RT-DETRv2-L__nativo/BoxPR_curve.png) |
| F1 según la confianza | ![F1 YOLO](results/comparativa_detectores/val_YOLO11n/BoxF1_curve.png) | ![F1 RT-DETR](results/comparativa_detectores/val_RT-DETR-l/BoxF1_curve.png) | ![F1 nativo](results/comparativa_detectores/val_RT-DETRv2-L__nativo/BoxF1_curve.png) |
| Matriz de confusión | ![Confusión YOLO](results/comparativa_detectores/val_YOLO11n/confusion_matrix_normalized.png) | ![Confusión RT-DETR](results/comparativa_detectores/val_RT-DETR-l/confusion_matrix_normalized.png) | ![Confusión nativo](results/comparativa_detectores/val_RT-DETRv2-L__nativo/confusion_matrix_normalized.png) |

</details>

**Análisis:**
- **Precisión de detección:** el RT-DETRv2-L nativo es el mejor en casi todo: mAP50 de 0,781 frente a 0,720 del RT-DETR-l y 0,672 de YOLO11n, y mAP50-95 de 0,419 frente a 0,402 y 0,352. Donde más se nota es en el *recall* (0,727 frente a 0,672 y 0,643), es decir, se deja menos matrículas sin detectar. El RT-DETR-l tiene la precisión más alta (0,904), pero la diferencia con los otros dos es pequeña.
- **El punto débil común es el *recall*:** los tres se dejan entre un 27 % y un 36 % de las matrículas de validación, mientras que la precisión está cerca de 0,9. Cuando detectan algo casi siempre es una matrícula, pero no las encuentran todas.
- **Coste:** YOLO11n es con diferencia el más ligero: 31 FPS, 2,6 M de parámetros, 6,6 GFLOPs y 5,5 MB, unas 17 veces menos operaciones que el RT-DETR-l. El RT-DETR-l tarda el doble (17 FPS) y el nativo el cuádruple (8 FPS).
- **El preproceso del nativo:** en inferencia pura el nativo solo tarda 10 ms más que el RT-DETR-l (64 frente a 54,5 ms). Casi la mitad de su tiempo total es el preproceso (59,6 ms), que se hace en la CPU con PIL redimensionando la foto completa. Probablemente se podría bajar mucho haciendo el redimensionado con OpenCV, como Ultralytics.
- **RT-DETR nativo frente al de Ultralytics:** no son el mismo modelo (RT-DETRv2 con ResNet-50 frente a RT-DETR con HGNetv2), ni se han entrenado igual (*data augmentation*, *batch*, épocas). Aun así, el nativo, con solo 60 épocas, supera al de Ultralytics, que entrenó 100. Con estos experimentos no se puede saber qué parte de la mejora viene del modelo y qué parte del entrenamiento.
- **Umbral de confianza:** el umbral que maximiza F1 es 0,34 en YOLO11n, 0,64 en RT-DETR-l y 0,58 en el nativo. Por eso cada modelo usa el suyo en el pipeline; con un umbral común se perjudicaría a alguno.
- **Velocidad en validación y en la comparativa:** los tiempos de esta tabla son mayores que los que da Ultralytics al acabar de entrenar (2,6 ms en YOLO11n y 11,1 ms en RT-DETR-l) porque aquí se mide con *batch* 1, que es como se usan los modelos en el vídeo.

### Pipeline de vídeo

La sección 6 procesa el vídeo de test con cualquiera de los tres detectores (`main(detector="yolo" | "rtdetr" | "rtdetr_nativo", ocr=...)`):

| Detector | Personas y vehículos | Matrículas | Tracking |
|---|---|---|---|
| `yolo` | `yolo11s.pt` a 960 px | YOLO11n a 1280 px | Ultralytics |
| `rtdetr` | `rtdetr-l.pt` a 640 px | RT-DETR-l a 640 px | Ultralytics |
| `rtdetr_nativo` | RT-DETRv2-L de COCO a 640 px | RT-DETRv2-L a 640 px | ByteTrack (`supervision`) |

YOLO se ejecuta por encima de su resolución de entrenamiento porque las matrículas del vídeo son pequeñas y así las detecta mejor. RT-DETR, en cambio, se mantiene a 640 px: a más resolución apenas detectaba matrículas.

Pasos en cada fotograma:
1. **Personas y vehículos:** el detector general con *tracking*, filtrado a persona, bicicleta, coche, moto, autobús y camión. El tracker recibe también las detecciones dudosas (confianza ≥ 0,1) para no perder objetos, pero solo se dibujan y guardan las de confianza ≥ 0,3.
2. **Matrículas:** el detector entrenado, con su propio tracker. Cada matrícula se asocia al vehículo cuya caja contiene su centro; si un vehículo tiene varias, se queda con la más fiable.
3. **Identificadores:** un objeto solo recibe un ID visible cuando su track lleva 5 fotogramas, y los IDs se numeran de forma consecutiva. Así, las detecciones falsas de un solo fotograma no cuentan en los totales.
4. **Conteo:** cada ID cuenta una vez, con la clase que más veces se le ha asignado.
5. **OCR (Tarea 2):** las matrículas se leen en segundo plano para no frenar el vídeo.
6. **Salidas:** el vídeo anotado, el CSV de detecciones, los totales, las matrículas leídas, el flujo y el vídeo anonimizado.

El CSV de detecciones tiene una fila por objeto y fotograma, con el formato propuesto en el enunciado:

```
fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matricula_en_su_caso, confianza_matricula, mx1, my1, mx2, my2, texto_matricula
```

**Resultado con el vídeo de test** (YOLO + PaddleOCR-VL, 2832 fotogramas):

![Vídeo resultado](results/yolo_paddle.gif)

Vídeos completos en el Drive:
[YOLO + PaddleOCR-VL](https://drive.google.com/file/d/1aI0DhMli2aDANVS48pU5NRbpQuzbCnkl/view) ·
[YOLO + Tesseract](https://drive.google.com/file/d/1KayS-jZcH5l2s8aj2qSDpefZ_Pjkt133/view) ·
[RT-DETR + PaddleOCR-VL](https://drive.google.com/file/d/11V7lZGMQW3X42UT2EEv_5jQcCQEOAktS/view) ·
[RT-DETR + Tesseract](https://drive.google.com/file/d/1nuaKgDV1ymaeMIHNA1emPUEenYwzoPgN/view) ·
[RT-DETR nativo + PaddleOCR-VL](https://drive.google.com/file/d/1wDkWRe5HBOJ10Ptq8_MrQLxaDiEAPC5B/view)

| Clase | Total |
|---|---|
| Persona | 53 |
| Coche | 129 |
| Bicicleta | 1 |
| Moto | 2 |
| Autobús | 2 |
| Camión | 2 |
| Matrícula | 31 |

De las 31 matrículas seguidas se leyeron 27, y 12 de ellas quedaron confirmadas por votación ([`yolo_paddle_matriculas.csv`](results/yolo_paddle_matriculas.csv)).

**Problemas observados:**
- El total de coches es más alto que el número real de coches: cuando un coche queda tapado, o sale y vuelve a entrar, el tracker le da un ID nuevo y cuenta dos veces.
- Algunas matrículas lejanas se leen con textos que no existen (por ejemplo, `1000DDD` aparece en tres IDs distintos). Ninguna de esas lecturas llega a confirmarse, porque no se repiten lo suficiente.

### Los tres detectores en el vídeo

El mismo vídeo procesado con los tres detectores y PaddleOCR-VL:

| Clase | YOLO | RT-DETR (Ultralytics) | RT-DETR nativo |
|---|---|---|---|
| Persona | 53 | 51 | 137 |
| Coche | 129 | 236 | 320 |
| Bicicleta | 1 | 6 | 4 |
| Moto | 2 | 3 | 7 |
| Autobús | 2 | 4 | 6 |
| Camión | 2 | 18 | 23 |
| Matrícula (IDs) | 31 | 62 | 95 |
| Matrículas leídas | 27 | 27 | 28 |
| Matrículas confirmadas | 12 | 13 | 16 |

- **Los totales no miden cuántos objetos detecta cada modelo, sino cuántos IDs crea.** El mismo coche puede contar varias veces si su track se rompe, así que un total más alto suele significar un seguimiento peor, no una detección mejor.
- **El RT-DETR nativo es el que más IDs crea** (320 coches y 137 personas, frente a 129 y 53 con YOLO). Su flujo lo confirma: 286 de sus 320 coches acaban como *sin determinar*, es decir, se pierden en medio de la imagen. La causa más probable es el tracker: con Ultralytics se usa BoT-SORT con compensación del movimiento de la cámara y re-identificación por apariencia (`tracker_general.yaml`), y el ByteTrack de `supervision` no tiene ninguna de las dos cosas.
- **RT-DETR-l** usa el mismo tracker que YOLO y aun así crea casi el doble de IDs de coches y 9 veces más de camiones. Con estos datos no se puede saber si se debe a que detecta más vehículos (por ejemplo, aparcados o lejanos), a que sus detecciones son más inestables entre fotogramas o a que confunde clases.
- **Matrículas:** aunque los RT-DETR crean muchos más IDs de matrícula, el número de matrículas leídas es casi el mismo (27-28). Los dos RT-DETR confirman alguna más (13 y 16 frente a 12). Entre las tres ejecuciones salen 19 textos confirmados distintos y 9 aparecen en las tres.
- **Confirmada no significa correcta:** el vídeo no tiene etiquetas, pero hay matrículas confirmadas con un carácter distinto según el detector (`3713LKC` / `3713LXC`, `9786LCF` / `9785LCF`, `6713HCR` / `6713HCP`), así que al menos una de las dos lecturas está mal.

### Extra: flujo por direcciones

`ContadorFlujo` guarda la posición de cada ID la primera y la última vez que se ve. Cuando un ID lleva 1 s (25 fotogramas) sin verse, se decide por dónde ha salido:

1. Si su último centro estaba cerca de un borde (8 % del fotograma), sale por ese borde.
2. Si se perdió en medio de la imagen, cuenta la dirección dominante de su desplazamiento.
3. Si apenas se ha movido, cuenta como *sin determinar*.

Los totales se dibujan en la esquina superior derecha del vídeo y se guardan en `results/<ejecución>_flujo.csv`. Resultado con YOLO ([`yolo_paddle_flujo.csv`](results/yolo_paddle_flujo.csv)):

| Clase | Izquierda | Derecha | Arriba | Abajo | Sin determinar |
|---|---|---|---|---|---|
| Persona | 8 | 12 | 0 | 0 | 34 |
| Coche | 20 | 4 | 0 | 0 | 103 |
| Bicicleta | 1 | 0 | 0 | 0 | 0 |
| Moto | 3 | 0 | 0 | 0 | 0 |
| Autobús | 2 | 0 | 0 | 0 | 0 |
| Camión | 0 | 0 | 0 | 0 | 2 |

- En este vídeo todo el movimiento es lateral: nadie sale por arriba ni por abajo. Los coches salen sobre todo por la izquierda (20 frente a 4) y las personas algo más por la derecha (12 frente a 8).
- La mayoría de los coches (103 de 127) quedan como *sin determinar*. Son coches que apenas se mueven en el vídeo (por ejemplo, aparcados) o IDs que se rompen y se pierden sin desplazarse, el mismo problema que infla los totales.
- La suma del flujo no coincide exactamente con los totales (127 coches frente a 129 y 54 personas frente a 53): el flujo se calcula aparte del conteo y no se ha investigado de dónde sale la diferencia.
- Con el RT-DETR nativo ([`rtdetr_nativo_paddle_flujo.csv`](results/rtdetr_nativo_paddle_flujo.csv)) el reparto de salidas es parecido (26 coches por la izquierda y 6 por la derecha), pero con muchos más *sin determinar* (286 coches), por lo explicado en el apartado anterior.

### Extra: anonimización

Se genera un segundo vídeo en el que las personas y todas las matrículas se tapan con la imagen de un mandril:
- La imagen se escala para cubrir toda la caja sin deformarse.
- La caja se amplía un 5 % para que no asome nada por los bordes.
- Se trabaja sobre una copia del fotograma sin anotar, así que el vídeo principal no cambia.

![Vídeo anonimizado](results/yolo_paddle_anonimo.gif)

Vídeos completos en el Drive:
[YOLO + PaddleOCR-VL anonimizado](https://drive.google.com/file/d/1QU52L5NZKGi3XgVmrgquNmBv4lOe9jGm/view) ·
[RT-DETR nativo + PaddleOCR-VL anonimizado](https://drive.google.com/file/d/1qNS5pzf1DKzJua6ahfP1IhadR-XlDtyX/view)

------

## Tarea 2 — Reconocimiento de caracteres en las matrículas

### OCR

Se usan dos OCR (sección 4), uno de cada tipo:

- **Tesseract** (clásico), en modo de una sola línea y limitado a letras mayúsculas y números. A cada matrícula se le pasan hasta cuatro preparaciones del recorte, porque acierta con unas y falla con otras: gris con contraste mejorado (CLAHE), umbral de Otsu, umbral adaptativo y una versión en la que solo quedan las manchas con forma de carácter (sin marco, tornillos ni rejilla). Gana la lectura con formato español más repetida. Las versiones se leen en paralelo.
- **PaddleOCR-VL-1.5** (VLM de 0,9 B parámetros especializado en OCR), con la instrucción `OCR:`. Se le pasa la matrícula en color y ampliada a 96 px de alto, porque el modelo está entrenado con fotos reales.

En los dos casos se quita antes la franja azul europea, y la lectura se corrige con el formato de las matrículas españolas: la actual (`1234BCD`) y las provinciales de Canarias (`GC1234AB`, `TF123456`). Según la posición se corrigen las confusiones típicas, como `O`→`0` en la parte numérica u `8`→`B` en la de letras.

En el vídeo, `LectorMatriculas` decide qué leer y cuándo:
- Cada matrícula se lee como mucho cada 3 fotogramas, y solo si se ve al menos tan grande como en su mejor lectura.
- Las lecturas de cada ID se votan, y cada voto pesa según el tamaño de la matrícula.
- Una lectura con formato español que se repite 3 veces y pesa el doble que la siguiente queda confirmada, y esa matrícula ya no vuelve a pasar por el OCR.

### Comparativa: Tesseract vs PaddleOCR-VL

La sección 7 compara los dos OCR con las matrículas del dataset, cuyo texto real viene en las anotaciones:

- **Qué matrículas:** de todas las imágenes, porque los OCR no se entrenan y el reparto entrenamiento/validación no les afecta. Se descartan las matrículas sin texto real (`None`, `"None"` o vacío) y las de menos de 40 px de ancho, que son ilegibles.
- **Mismas condiciones:** los dos OCR reciben los mismos recortes, con el mismo preprocesado que en el vídeo y 3 lecturas de calentamiento. El recorte queda fuera de la medida de tiempo.
- **Acierto:** lectura exacta, lecturas con como mucho un carácter mal y CER (errores de carácter divididos entre la longitud de la matrícula).
- **Tiempo en lote:** todas las matrículas en lotes de 16, como en el vídeo. Es la velocidad real del pipeline. Tesseract procesa el lote en paralelo y PaddleOCR-VL de una en una.
- **Latencia:** 30 matrículas (las mismas para los dos) leídas de una en una, que es lo que tarda en leerse una matrícula suelta.

**Tasa de acierto** (726 matrículas de 500 imágenes):

| OCR | Acierto exacto (%) | ≤ 1 error (%) | CER medio | Sin lectura (%) |
|---|---|---|---|---|
| Tesseract (clásico) | 34.0 | 46.1 | 0.42 | 26.7 |
| PaddleOCR-VL (VLM) | **75.3** | **85.8** | **0.09** | **1.9** |

**Tiempo por matrícula:**

| OCR | En lote (ms) | Matrículas/s (lote) | Latencia media (ms) | Latencia std (ms) | Latencia mediana (ms) |
|---|---|---|---|---|---|
| Tesseract (clásico) | **181.2** | **5.5** | **271.3** | 26.2 | 267.8 |
| PaddleOCR-VL (VLM) | 680.5 | 1.5 | 703.3 | 246.4 | 667.7 |

**Acierto exacto según el ancho de la matrícula:**

| OCR | 40-70 px (95 matrículas) | 70-110 px (110) | > 110 px (521) |
|---|---|---|---|
| Tesseract (clásico) | 10.5 % | 24.5 % | 40.3 % |
| PaddleOCR-VL (VLM) | **52.6 %** | **69.1 %** | **80.8 %** |

Tablas generadas por el cuaderno: [`comparativa_ocr.md`](results/comparativa_ocr/comparativa_ocr.md). La lectura de cada matrícula está en [`lecturas_tesseract.csv`](results/comparativa_ocr/lecturas_tesseract.csv) y [`lecturas_paddle.csv`](results/comparativa_ocr/lecturas_paddle.csv).

![Acierto de los OCR](results/comparativa_ocr/acierto.png)

![Tiempos de los OCR](results/comparativa_ocr/tiempos.png)

**Ejemplos de lectura:**

![Ejemplos de lectura](results/comparativa_ocr/ejemplos.png)

**Conclusiones:**
- **PaddleOCR-VL acierta mucho más:** lee bien el 75,3 % de las matrículas frente al 34,0 % de Tesseract, y su CER es casi 5 veces menor (0,09 frente a 0,42). De las 726 matrículas, 240 las aciertan los dos, 307 solo PaddleOCR-VL y solo 7 únicamente Tesseract.
- **Tesseract falla sobre todo por no leer nada:** en el 26,7 % de las matrículas no devuelve ninguna lectura válida, frente al 1,9 % de PaddleOCR-VL. Cuando lee algo, el error típico es cambiar caracteres parecidos (`6089GLG` → `6836LGJ`) o dejarse algunos.
- **El error típico de PaddleOCR-VL es comerse caracteres:** en 100 de sus 165 lecturas erróneas le faltan caracteres, normalmente los primeros dígitos (`6174HTD` → `174HTD`). En recortes pequeños y borrosos a veces se inventa una matrícula que no existe (`2908MDL` → `1000DDD` con 59 px de ancho), lo mismo que ocurre en el vídeo.
- **El tamaño importa a los dos,** pero PaddleOCR-VL aguanta mucho mejor: con matrículas de 40-70 px acierta el 52,6 %, más que Tesseract con las grandes (40,3 %).
- **Matrículas que no siguen el formato español actual** (48 de 726, como las provinciales o las extranjeras): bajan el acierto de los dos (PaddleOCR-VL 60,4 %, Tesseract 20,8 %). 2 de los 7 aciertos en solitario de Tesseract son de este tipo: la francesa `EQ572GS` del ejemplo y la provincial `GC0040BH`.
- **Velocidad:** PaddleOCR-VL es unas 3,8 veces más lento en lote (680 frente a 181 ms por matrícula) y 2,6 veces en latencia. Además, su latencia varía mucho más (desviación de 246 ms frente a 26), porque genera el texto carácter a carácter. Tesseract tampoco es rápido (271 ms por matrícula suelta), porque lee hasta cuatro versiones de cada recorte.

### Los dos OCR en el vídeo

Con el mismo detector el *tracking* es idéntico, así que las dos ejecuciones tienen las mismas matrículas y se pueden comparar ID a ID:

| Detector | OCR | Matrículas leídas | Confirmadas | Confirmadas por los dos OCR con el mismo texto |
|---|---|---|---|---|
| YOLO | PaddleOCR-VL | 27 | **12** | 5 |
| YOLO | Tesseract | 23 | 6 | 5 |
| RT-DETR (Ultralytics) | PaddleOCR-VL | 27 | **13** | 5 |
| RT-DETR (Ultralytics) | Tesseract | 25 | 7 | 5 |

- **PaddleOCR-VL confirma el doble de matrículas** con cualquiera de los dos detectores. Casi todo lo que confirma Tesseract (5 de 6 y 5 de 7) coincide con lo que lee PaddleOCR-VL.
- **Su coste no frena el vídeo:** `LectorMatriculas` lee en segundo plano, como mucho cada 3 fotogramas, y deja de leer una matrícula en cuanto está confirmada. Una lectura más lenta solo retrasa el momento en que aparece el texto.
- **Conclusión:** para este problema PaddleOCR-VL compensa su coste. Tesseract solo tendría sentido sin GPU o cuando el tiempo por matrícula importe más que el acierto.

------

## Referencias

- [Guía de la Práctica 4](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4)
- [Guía de la Práctica 4b](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4b)
- [Ultralytics: YOLO y RT-DETR](https://docs.ultralytics.com/)
- [Repositorio oficial de RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- [supervision (ByteTrack)](https://github.com/roboflow/supervision)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
