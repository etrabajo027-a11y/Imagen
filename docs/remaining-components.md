# Componentes pendientes: Mask R-CNN y análisis ambiental

## Estado y alcance

DeepLabV3+ ya está entrenado y evaluado en una muestra reducida de BDD100K. Esta fase incorpora pruebas técnicas para los componentes restantes, sin repetir ese entrenamiento.

## Segmentación por instancias

BDD100K publica un paquete independiente de Instance Segmentation (111 MB; MD5 oficial `651b41f229d7327d8c4af97772de4390`). Sus máscaras son PNG RGBA: R identifica la categoría, G codifica atributos (truncado, ocluido, crowd e ignore) y B contiene el identificador de instancia para esta tarea. Las ocho categorías evaluadas son pedestrian, rider, car, truck, bus, train, motorcycle y bicycle.

No se deben reutilizar las máscaras semánticas como etiquetas de instancias. El flujo debe descargar el paquete de instancia, emparejar cada máscara con la imagen por nombre, excluir regiones crowd/ignore y construir máscaras binarias, cajas y etiquetas para Mask R-CNN.

## Metadatos ambientales

Los atributos oficiales son `weather`, `scene` y `timeofday`. Las imágenes 10K no son un subconjunto exacto de las 100K; por ello el análisis ambiental debe cruzar los nombres/IDs y reportar:
- cobertura total y porcentaje de coincidencia;
- número de imágenes por clima y hora del día;
- resultados por subconjunto únicamente cuando exista etiqueta real;
- categoría `unknown` para imágenes sin correspondencia.

No se debe etiquetar lluvia, niebla o noche mediante inferencia visual o el nombre del archivo.

## Prueba automatizada

El workflow `Remaining components readiness` comprueba:
1. decodificación de máscaras RGBA y separación de instancias;
2. forward, cálculo de pérdidas, backward e inferencia de Mask R-CNN con nueve salidas (fondo + ocho categorías);
3. unión de metadatos ambientales por nombre y conservación de valores desconocidos.

Esta prueba es sintética: verifica el código y las formas de los tensores, pero **no constituye entrenamiento ni evaluación real en BDD100K**. La fase de entrenamiento real debe ejecutarse después de validar la estructura del ZIP oficial y sus emparejamientos.

## Comparación de modelos

DeepLabV3+ y Mask R-CNN resuelven tareas diferentes. Para la comparación se reportarán por separado:
- DeepLabV3+: mIoU y Pixel Accuracy en máscaras semánticas.
- Mask R-CNN: mAP@[.50:.95] y AP50 en instancias.

No se deben comparar directamente los valores mIoU y mAP como si midieran lo mismo. La comparación cualitativa utilizará imágenes comunes, con predicción semántica y máscaras individuales superpuestas.

## Fuente oficial

- [Descarga y tamaños/checksums BDD100K](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst)
- [Formato de etiquetas, categorías y canales RGBA](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)
