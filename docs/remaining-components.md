# Componentes pendientes: Mask R-CNN y análisis ambiental

## Entrenamiento Mask R-CNN

El workflow `Mask R-CNN BDD100K real-data training (CPU)` entrena una muestra controlada con imágenes y bitmasks oficiales de BDD100K. Parámetros iniciales: 100 imágenes train, 30 validation, 2 épocas, resolución 512×288, batch 1 y backbone ResNet-50-FPN con pesos COCO preentrenados. La finalidad de esta primera ejecución es obtener un modelo real reproducible dentro del límite de GitHub Actions; no equivale al entrenamiento completo de las 7.000 imágenes.

El lector remoto usa HTTP Range/RemoteZip para leer únicamente las imágenes seleccionadas del ZIP de 10K y las máscaras correspondientes. No descarga ni almacena el ZIP completo. Las imágenes se leen desde el directorio Berkeley histórico y las máscaras desde el archivo oficial `bdd100k_ins_seg_labels_trainval.zip` del endpoint de publicación ETH/S3. El ZIP de máscaras se verifica por estructura: debe contener bitmasks train y val.

## Formato de las máscaras de instancia

BDD100K almacena las máscaras como PNG RGBA:
- R: categoría (1–8).
- G: atributos; bit 0 = ignore, bit 1 = crowd, bit 2 = occluded, bit 3 = truncated.
- B y A: identificador de instancia de 16 bits, calculado como `(B << 8) | A`.

Se excluyen regiones crowd e ignore. Las ocho categorías son pedestrian, rider, car, truck, bus, train, motorcycle y bicycle. Las cajas se calculan a partir de las máscaras binarias.

## Métricas

El entrenamiento reporta AP50 y mAP@[.50:.95] mediante AP interpolada a 101 puntos sobre IoU de máscaras, por categoría y umbral. Esta implementación permite una evaluación local reproducible del subconjunto; no debe presentarse como resultado oficial del servidor BDD100K.

## Comparación con DeepLabV3+

- DeepLabV3+: mIoU y Pixel Accuracy en segmentación semántica.
- Mask R-CNN: AP50 y mAP@[.50:.95] en segmentación por instancias.

No se comparan mIoU y mAP como si fueran la misma métrica.

## Metadatos ambientales

Los atributos oficiales `weather`, `scene` y `timeofday` deben cruzarse por nombre/ID. Las imágenes del paquete 10K no son un subconjunto exacto de 100K; se informará la cobertura y se conservarán las imágenes sin correspondencia como `unknown`. Los análisis por lluvia, niebla, noche y día solo se realizarán con etiquetas reales coincidentes.

## Fuentes oficiales

- [Descarga, tamaños y checksums BDD100K](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst)
- [Formato de etiquetas y canales RGBA](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)
- [Codificador oficial de bitmasks](https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/to_mask.py)
