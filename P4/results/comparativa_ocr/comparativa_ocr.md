726 matrículas de 500 imágenes del dataset (ancho ≥ 40 px, texto real del campo `plate` de las anotaciones; excluidas las matrículas sin texto real), mismo recorte y preprocesado para ambos OCR y 3 lecturas de calentamiento. El recorte queda fuera de las medidas de tiempo.

- **En lote**: todas las matrículas, en lotes de 16 (tiempo del lote / tamaño). Es el rendimiento en el vídeo; Tesseract procesa el lote en paralelo, PaddleOCR-VL de una en una.
- **Latencia**: 30 matrículas (subconjunto fijo, el mismo para los dos), leídas de una en una.

Equipo: SO: Windows 10 | CPU: AMD Ryzen 7 7735HS with Radeon Graphics | RAM: 31 GB | GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) | Python: 3.11.18 | torch: 2.6.0+cu124 | CUDA: 12.4 | ultralytics: 8.4.190

**Tasa de acierto**

| OCR | Acierto exacto (%) | ≤ 1 error (%) | CER medio | Sin lectura (%) | Errores del OCR (%) |
|---|---|---|---|---|---|
| Tesseract (clásico) | 34.02 | 46.14 | 0.42 | 26.72 | 0.00 |
| PaddleOCR-VL (VLM) | 75.34 | 85.81 | 0.09 | 1.93 | 0.00 |

**Tiempo de inferencia por matrícula**

| OCR | En lote (ms/matrícula) | Matrículas/s (lote) | Latencia media (ms) | Latencia std (ms) | Latencia mediana (ms) |
|---|---|---|---|---|---|
| Tesseract (clásico) | 181.2 | 5.5 | 271.3 | 26.2 | 267.8 |
| PaddleOCR-VL (VLM) | 680.5 | 1.5 | 703.3 | 246.4 | 667.7 |

**Acierto exacto según el ancho de la matrícula** (n = {'40-70 px': 95, '70-110 px': 110, '> 110 px': 521})

| OCR | Acierto 40-70 px (%) | Acierto 70-110 px (%) | Acierto > 110 px (%) |
|---|---|---|---|
| Tesseract (clásico) | 10.5 | 24.5 | 40.3 |
| PaddleOCR-VL (VLM) | 52.6 | 69.1 | 80.8 |