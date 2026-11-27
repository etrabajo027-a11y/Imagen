# Evaluación del modelo DeepLabV3+

## Alcance del experimento

Se evaluó el checkpoint obtenido en el workflow de entrenamiento reducido, sin volver a entrenar el modelo ni modificar sus pesos.

- Arquitectura: DeepLabV3+ con backbone ResNet-50 preentrenado en ImageNet.
- Dataset: BDD100K, máscaras semánticas en train IDs.
- Muestra de entrenamiento original: 400 imágenes.
- Muestra de validación: 100 imágenes.
- Épocas completadas: 15.
- Resolución: 512 × 288.
- Batch size: 1.
- Semilla: 42.
- Tiempo de entrenamiento original: 2 h 30 min 20 s.
- Checkpoint evaluado: `deeplabv3plus_best.pth`, época 15.

El script de evaluación reconstruyó la misma muestra de validación mediante la misma semilla y el mismo procedimiento de selección usado durante el entrenamiento. Se verificaron 7.000 pares TRAIN y 1.000 pares VAL disponibles en los archivos BDD100K, y se prepararon los mismos 100 pares de validación.

## Resultados globales

| Métrica | Resultado |
|---|---:|
| Pérdida de validación | 0.57482 |
| mIoU en clases con píxeles reales o predichos (unión > 0) | 36.57 % |
| mIoU incluyendo las 19 clases, asignando cero a clases sin unión | 34.65 % |
| Precisión por píxel | 87.34 % |
| Píxeles válidos evaluados | 12.554.600 |
| Imágenes evaluadas | 100 |

La mIoU de 36.57 % coincide con la métrica guardada durante el entrenamiento. La diferencia con el promedio de las 19 clases se debe a que la clase `train` (ID 16) no aparece en esta muestra de validación y, por tanto, no tiene unión para calcular IoU.

## IoU por clase

| ID | Clase | IoU |
|---:|---|---:|
| 0 | road | 89.53 % |
| 1 | sidewalk | 45.26 % |
| 2 | building | 71.67 % |
| 3 | wall | 0.19 % |
| 4 | fence | 8.26 % |
| 5 | pole | 21.18 % |
| 6 | traffic light | 27.71 % |
| 7 | traffic sign | 19.68 % |
| 8 | vegetation | 79.40 % |
| 9 | terrain | 36.39 % |
| 10 | sky | 93.95 % |
| 11 | person | 38.21 % |
| 12 | rider | 0.00 % |
| 13 | car | 77.35 % |
| 14 | truck | 14.52 % |
| 15 | bus | 15.35 % |
| 16 | train | Sin muestras reales en VAL |
| 17 | motorcycle | 19.68 % |
| 18 | bicycle | 0.00 % |

## Comprobaciones técnicas

- El artefacto de entrenamiento se descargó desde el run original y se verificaron los cinco archivos esperados.
- El checkpoint `best` se cargó con coincidencia estricta de parámetros y se evaluó sobre las 100 imágenes.
- El checkpoint `last` también se cargó correctamente.
- El checkpoint `last` conserva `optimizer_state_dict`, por lo que se puede reanudar el entrenamiento.
- Se generaron 12 paneles con imagen original, predicción y máscara real.
- La ejecución completa de evaluación terminó en aproximadamente 68 segundos, sin ejecutar entrenamiento.

## Interpretación y limitaciones

El modelo funciona mejor en clases frecuentes y extensas (carretera, cielo, vegetación, edificios y coches). Su rendimiento es bajo en varias clases menos frecuentes o pequeñas, especialmente muro, valla, ciclista, camión, autobús, motocicleta y bicicleta.

La precisión por píxel de 87.34 % no debe interpretarse como una calidad uniforme entre clases: las clases grandes dominan esa métrica. Para segmentación multiclase, la mIoU y el análisis por clase son más informativos.

Este resultado corresponde a un experimento reducido de 400/100 imágenes, no a un entrenamiento completo con las 7.000 imágenes TRAIN y 1.000 VAL. No se declara el modelo como definitivo. Una ampliación futura debería priorizar más datos y representación de clases raras, conservar una validación independiente y reanudar desde el checkpoint existente, no entrenar desde cero.

## Ejecuciones y artefactos

- Entrenamiento: [GitHub Actions run 45887532742](https://github.com/etrabajo027-a11y/Imagen/actions/runs/45887532742).
- Artefacto del modelo (847 MB, retención hasta 26-12-2026): [descargar checkpoint y métricas](https://github.com/etrabajo027-a11y/Imagen/actions/runs/45887532742/artifacts/15157038640).
- Evaluación final: [GitHub Actions run 45914313778](https://github.com/etrabajo027-a11y/Imagen/actions/runs/45914313778).
- Artefacto de evaluación (métricas, matriz de confusión y 12 ejemplos visuales; retención hasta 27-12-2026): [descargar resultados](https://github.com/etrabajo027-a11y/Imagen/actions/runs/45914313778/artifacts/15161761604).

Los artefactos de Actions son temporales. Deben descargarse y conservarse externamente antes de sus fechas de expiración.
