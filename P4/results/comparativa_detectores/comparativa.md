Validación: 395 imágenes, imgsz=640, batch 1 para los tiempos. Equipo: SO: Windows 10 | CPU: AMD Ryzen 7 7735HS with Radeon Graphics | RAM: 31 GB | GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) | Python: 3.11.18 | torch: 2.6.0+cu124 | CUDA: 12.4 | ultralytics: 8.4.190

**Detección**

| Modelo | Precision | Recall | F1 max | mAP50 | mAP50-95 | Conf. F1 max |
|---|---|---|---|---|---|---|
| YOLO11n | 0.879 | 0.643 | 0.743 | 0.672 | 0.352 | 0.344 |
| RT-DETR-l | 0.904 | 0.672 | 0.777 | 0.720 | 0.402 | 0.642 |
| RT-DETRv2-L (nativo) | 0.887 | 0.727 | 0.799 | 0.781 | 0.419 | 0.581 |

**Velocidad y tamaño**

| Modelo | Preproceso (ms) | Inferencia (ms) | Postproceso (ms) | Total (ms) | FPS | Parámetros (M) | GFLOPs | Tamaño (MB) |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | 3.4 | 26.1 | 2.1 | 32.2 | 31.1 | 2.6 | 6.6 | 5.5 |
| RT-DETR-l | 3.1 | 54.5 | 1.0 | 59.2 | 16.9 | 32.8 | 109.9 | 66.2 |
| RT-DETRv2-L (nativo) | 59.6 | 64.0 | 1.5 | 125.4 | 8.0 | 49.8 | 133.7 | 86.0 |