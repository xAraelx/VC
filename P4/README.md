# Práctica 4 y 4b

Este repositorio contiene el cuaderno `VC_P4.ipynb`, que reúne las dos partes de la práctica 4. La primera (P4) es un prototipo que procesa vídeo para detectar y seguir personas y vehículos y localizar sus matrículas. La segunda (P4b) lee el texto de esas matrículas con OCR y compara un OCR clásico con un modelo de lenguaje visual (VLM).

### Autores:
- [Sergio Rodriguez Rubio](https://github.com/SergioRodriguezEnt)
- [Arael Jesús Almeida González](https://github.com/xAraelx)

------

## Estructura de la carpeta

```
P4
|- VC_P4.ipynb                     #Cuaderno jupyter con las tareas desarrolladas
|- README.md                       #Descripción de la solución a las tareas
|- mandril.jpg                     #Imagen con la que se tapan personas y matrículas al anonimizar
|- tracker_general.yaml            #Configuración del tracker de personas y vehículos
|- tracker_matriculas.yaml         #Configuración del tracker de matrículas
|- datos/                          #Dataset de matrículas (imágenes en el enlace del apartado Dataset)
|   |- annotations/                #Anotaciones originales en JSON (caja + texto de la matrícula)
|   |- data.yaml, train.txt, val.txt
|- runs/detect/
|   |- train-3/                    #Entrenamiento de YOLO11n (pesos best.pt y curvas)
|   |- rtdetr-2/                   #Entrenamiento de RT-DETR-l (pesos best.pt y curvas)
|- results/
    |- yolo_paddle_detecciones.csv #Detecciones y seguimiento de cada fotograma del vídeo de test
    |- yolo_paddle_totales.csv     #Total de objetos únicos por clase
    |- yolo_paddle_matriculas.csv  #Texto leído de cada matrícula
    |- yolo_paddle_flujo.csv       #Salidas por dirección de cada clase
    |- comparativa_detectores/     #Tablas y gráficas de YOLO vs RT-DETR
    |- comparativa_ocr/            #Tablas y gráficas de Tesseract vs PaddleOCR-VL
```

El vídeo de test (`C0142.MP4`) y los vídeos de resultado no se incluyen por su tamaño; están en los enlaces de cada apartado.

## Recursos necesarios

Se ha usado un *environment* `VC_P4` con Python 3.11. Además de Ultralytics hacen falta las librerías del tracking, de los dos OCR y de las tablas:

- `pip install ultralytics lap pandas psutil`
- `pip install pytesseract` y el programa [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (en Windows se busca en `C:\Program Files\Tesseract-OCR`)
- `pip install "transformers>=5.0.0" pillow` para PaddleOCR-VL
- PyTorch con CUDA ([instrucciones](https://pytorch.org/get-started/locally/)): sin GPU funciona, pero PaddleOCR-VL y RT-DETR son mucho más lentos.

Los pesos preentrenados (`yolo11s.pt`, `rtdetr-l.pt`, `yolo11n.pt`) los descarga Ultralytics la primera vez que se usan.

### Equipo de pruebas

Todos los tiempos de este README se han medido en este equipo:

| Componente | Valor |
|---|---|
| SO | Windows [PENDIENTE: versión] |
| CPU | [PENDIENTE] |
| RAM | [PENDIENTE] |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| Python | 3.11.18 |
| PyTorch | 2.6.0+cu124 |
| Ultralytics | 8.4.190 |

La celda de utilidades del cuaderno (sección 0) imprime esta tabla completa.

------

## Tarea 1 — Detección y seguimiento de personas, vehículos y matrículas

### Dataset de matrículas

Los modelos de Ultralytics no detectan matrículas, así que se ha entrenado un detector propio. El dataset tiene **1976 imágenes** anotadas: cada imagen tiene un JSON con la caja de cada matrícula (centro y tamaño normalizados) y su texto (campo `plate`), que se usa después en la comparativa de OCR.

La sección 1 del cuaderno convierte las anotaciones al formato de YOLO y reparte las imágenes al azar (semilla 42) en **1581 de entrenamiento y 395 de validación** (80/20).

Enlace al dataset: [PENDIENTE: enlace]

### Entrenamiento

Se han entrenado dos detectores de matrículas con el mismo dataset, a 640 px y durante 100 épocas:

- **YOLO11n**, partiendo de `yolo11n.pt`. Respecto a los valores por defecto se ha cambiado el *data augmentation*: más variación de saturación y brillo (día, noche, sombras), rotación de ±5°, cizalla y perspectiva leves, y sin volteo horizontal, porque el texto de la matrícula no debe salir invertido.
- **RT-DETR-l**, partiendo de `rtdetr-l.pt`. Se ha entrenado a 640 px, la resolución de su preentrenamiento, con AdamW y `lr0 = 0.0001` (valores habituales para modelos tipo DETR) y una paciencia de 30 épocas, porque estos modelos convergen más despacio que YOLO.

**Evolución del entrenamiento de YOLO11n:**

![Entrenamiento YOLO](runs/detect/train-3/results.png)
<!-- PENDIENTE: comprobar que la imagen existe en esa ruta -->

**Evolución del entrenamiento de RT-DETR-l:**

![Entrenamiento RT-DETR](runs/detect/rtdetr-2/results.png)
<!-- PENDIENTE: comprobar que la imagen existe en esa ruta -->

[PENDIENTE: comentar las curvas, por ejemplo si la pérdida de validación deja de bajar o si hay sobreajuste]

### Comparativa YOLO vs RT-DETR (extra)

La sección 3 evalúa los dos detectores en las mismas condiciones: las 395 imágenes de validación, a 640 px, en el mismo equipo y con un *batch* de 1 para los tiempos. Además de las métricas de detección, calcula el umbral de confianza que maximiza F1, que se usa como `conf_matricula` en el pipeline.

Resultados de la validación al final de cada entrenamiento (395 imágenes, 927 matrículas):

| Modelo | Precision | Recall | mAP50 | mAP50-95 | Inferencia (ms/img) | Parámetros (M) |
|---|---|---|---|---|---|---|
| YOLO11n | 0.886 | 0.645 | 0.674 | 0.351 | 2.6 | 2.6 |
| RT-DETR-l | 0.901 | 0.672 | 0.721 | 0.399 | 11.1 | 32.0 |

<!-- PENDIENTE: sustituir por las tablas de results/comparativa_detectores/comparativa.md al ejecutar la sección 3 -->

![Métricas de detección](results/comparativa_detectores/metricas.png)
<!-- PENDIENTE: gráfica generada por la sección 3 -->

![Tiempos de los detectores](results/comparativa_detectores/tiempos.png)
<!-- PENDIENTE: gráfica generada por la sección 3 -->

**Análisis:**
- RT-DETR-l es algo mejor en todas las métricas (+0.05 de mAP50 y +0.05 de mAP50-95), pero tiene unas 12 veces más parámetros y su inferencia es unas 4 veces más lenta.
- Los dos tienen una precisión alta (~0.9) y un recall bajo (~0.65): casi todo lo que detectan es una matrícula, pero se les escapa una de cada tres.
- [PENDIENTE: completar con los resultados de la sección 3]

### Pipeline de vídeo

La sección 6 procesa el vídeo de test con cualquiera de los dos detectores (`main(detector="yolo" | "rtdetr", ocr=...)`):

1. **Personas y vehículos**: un detector general preentrenado en COCO (`yolo11s.pt` a 960 px, o `rtdetr-l.pt` a 640 px) con *tracking*, filtrado a persona, bicicleta, coche, moto, autobús y camión. El tracker recibe también las detecciones dudosas (confianza ≥ 0.1) para no perder objetos, pero solo se dibujan y guardan las de confianza ≥ 0.3.
2. **Matrículas**: el detector entrenado, con su propio tracker. Cada matrícula se asocia al vehículo cuya caja contiene su centro; si un vehículo tiene varias, se queda con la más fiable.
3. **Identificadores**: un objeto solo recibe un ID visible cuando su track lleva 5 fotogramas, y los IDs se numeran de forma consecutiva. Así las detecciones falsas de un solo fotograma no cuentan en los totales.
4. **Conteo**: cada ID cuenta una vez, con la clase que más veces se le ha asignado.
5. **OCR** (Tarea 2): las matrículas se leen en segundo plano para no frenar el vídeo.
6. **Salidas**: el vídeo anotado, el CSV de detecciones, los totales, las matrículas leídas, el flujo y el vídeo anonimizado.

El CSV de detecciones tiene una fila por objeto y fotograma, con el formato propuesto en el enunciado:

```
fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matricula_en_su_caso, confianza_matricula, mx1, my1, mx2, my2, texto_matricula
```

**Resultado con el vídeo de test** (YOLO + PaddleOCR-VL):

![Vídeo resultado](results/yolo_paddle.gif)
<!-- PENDIENTE: GIF de un fragmento del vídeo resultado -->

Vídeo completo: [PENDIENTE: enlace]

| Clase | Total |
|---|---|
| Persona | 53 |
| Coche | 129 |
| Bicicleta | 1 |
| Moto | 2 |
| Autobús | 2 |
| Camión | 2 |
| Matrícula | 38 |

Se han procesado los 2832 fotogramas en 483 s (5.9 fps) y se han leído 29 matrículas, 12 de ellas confirmadas por votación.

**Problemas observados:**
- El total de coches es más alto que los coches reales: cuando un coche queda tapado o sale y vuelve a entrar, el tracker le da un ID nuevo y cuenta dos veces.
- [PENDIENTE: otros problemas vistos en el vídeo]

### Extra: flujo por direcciones

`ContadorFlujo` guarda la posición de cada ID la primera y la última vez que se ve. Cuando un ID lleva 1 s (25 fotogramas) sin verse, se decide por dónde ha salido:

1. Si su último centro estaba cerca de un borde (8 % del fotograma), sale por ese borde.
2. Si se perdió en medio de la imagen, cuenta la dirección dominante de su desplazamiento.
3. Si apenas se ha movido, cuenta como *sin determinar*.

Los totales se dibujan en la esquina superior derecha del vídeo y se guardan en `results/yolo_paddle_flujo.csv`.

| Clase | Izquierda | Derecha | Arriba | Abajo | Sin determinar |
|---|---|---|---|---|---|
| Persona | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Coche | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

### Extra: anonimización

Se genera un segundo vídeo en el que las personas y todas las matrículas se tapan con la imagen de un mandril. La imagen se escala para cubrir toda la caja sin deformarse, y la caja se amplía un 5 % para que no asome nada por los bordes. Se trabaja sobre una copia del fotograma sin anotar, así que el vídeo principal no cambia.

![Vídeo anonimizado](results/yolo_paddle_anonimo.gif)
<!-- PENDIENTE: GIF de un fragmento del vídeo anonimizado -->

Vídeo completo: [PENDIENTE: enlace]

------

## Tarea 2 — Reconocimiento de caracteres en las matrículas

### OCR

Se usan dos OCR (sección 4), uno de cada tipo:

- **Tesseract** (clásico), en modo de una sola línea y limitado a letras mayúsculas y números. Como acierta con unas preparaciones del recorte y falla con otras, a cada matrícula se le pasan hasta cuatro versiones: gris con contraste mejorado (CLAHE), umbral de Otsu, umbral adaptativo y una versión en la que solo quedan las manchas con forma de carácter, sin marco, tornillos ni rejilla. Gana la lectura con formato español más repetida. Las versiones se leen en paralelo.
- **PaddleOCR-VL-1.5** (VLM de 0.9B parámetros especializado en OCR), con la instrucción `OCR:`. Se le pasa la matrícula en color, ampliada a 96 px de alto, porque el modelo está entrenado con fotos reales.

En los dos casos se quita antes la franja azul europea y la lectura se corrige con el formato de las matrículas españolas (`limpiar`): la actual (`1234BCD`) y las provinciales de Canarias (`GC1234AB`, `TF123456`). Según la posición se corrigen las confusiones típicas, como `O`→`0` en la parte numérica u `8`→`B` en la de letras.

En el vídeo, `LectorMatriculas` decide qué leer y cuándo:
- Cada matrícula se lee como mucho cada 3 fotogramas, y solo si se ve al menos tan grande como en su mejor lectura anterior.
- Las lecturas de cada ID se votan, y cada voto pesa según el tamaño de la matrícula.
- Una lectura con formato español que se repite 3 veces y duplica en peso a la siguiente queda confirmada, y esa matrícula ya no vuelve a pasar por el OCR.

### Comparativa: Tesseract vs PaddleOCR-VL

La sección 7 compara los dos OCR con las matrículas del dataset, cuyo texto real viene en las anotaciones:

- Se usan matrículas de todas las imágenes (los OCR no se entrenan, así que el reparto entre entrenamiento y validación no importa).
- Se descartan las matrículas sin texto real (`None`, `"None"` o vacío) y las de menos de 40 px de ancho, que son ilegibles. El resto de matrículas de esas imágenes sí se usan.
- Los dos OCR reciben los mismos recortes, con el mismo preprocesado que en el vídeo.
- **Acierto**: lectura exacta, lecturas con como mucho un carácter mal y CER (errores de carácter divididos entre la longitud de la matrícula).
- **Tiempo en lote**: todas las matrículas en lotes de 16, como en el vídeo. Es la velocidad real del pipeline.
- **Latencia**: 30 matrículas leídas de una en una, lo que tarda en leerse una matrícula suelta.

**Tasa de acierto** ([PENDIENTE: nº] matrículas):

| OCR | Acierto exacto (%) | ≤ 1 error (%) | CER medio | Sin lectura (%) |
|---|---|---|---|---|
| Tesseract (clásico) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| PaddleOCR-VL (VLM) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

**Tiempo por matrícula:**

| OCR | En lote (ms) | Latencia (ms) | Matrículas/s (lote) |
|---|---|---|---|
| Tesseract (clásico) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| PaddleOCR-VL (VLM) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

<!-- PENDIENTE: copiar las tablas de results/comparativa_ocr/comparativa_ocr.md (incluye también el acierto por ancho de matrícula) -->

![Acierto de los OCR](results/comparativa_ocr/acierto.png)
<!-- PENDIENTE: gráfica generada por la sección 7 -->

![Tiempos de los OCR](results/comparativa_ocr/tiempos.png)
<!-- PENDIENTE: gráfica generada por la sección 7 -->

**Conclusiones:**
- [PENDIENTE: cuál acierta más y con qué tipo de matrículas falla cada uno]
- [PENDIENTE: diferencia de velocidad y si PaddleOCR-VL compensa su coste en el vídeo]

------

## Referencias

- [Guía de la Práctica 4](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4)
- [Guía de la Práctica 4b](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4b)
- [Ultralytics: YOLO y RT-DETR](https://docs.ultralytics.com/)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
