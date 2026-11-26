# Datos — BDD100K

## Dataset seleccionado

BDD100K es el conjunto principal del proyecto. Fuente oficial: https://dl.cv.ethz.ch/bdd100k/data/  
Documentación de descargas, tamaños, checksums y estructura: https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst

Al descargar se aceptan las condiciones oficiales de licencia/uso. No se redistribuyen los archivos de datos en este repositorio.

## Paquetes para el primer flujo

Para **segmentación semántica** e **instancias** se requieren:

- `10k_images_train.zip`
- `10k_images_val.zip`
- `10k_images_test.zip`
- `bdd100k_sem_seg_labels_trainval.zip` (MD5 documentado: `9a2968dde3345eeb689cffb1e26f9c78`)
- `bdd100k_ins_seg_labels_trainval.zip` (MD5 documentado: `651b41f229d7327d8c4af97772de4390`)

Los ZIP de etiquetas contienen máscaras y anotaciones para train/val; test no necesariamente tiene etiquetas públicas para estas tareas. No descargar los videos de 1.8 TB para este flujo.

## Diferencia importante entre tareas

BDD100K especifica que:

- Semantic segmentation: 10K Images + Semantic Segmentation labels.
- Instance segmentation: 10K Images + Instance Segmentation labels.
- Lane marking y drivable area: 100K Images + sus paquetes de anotaciones específicos.

Por tanto, no se debe tratar la clase `lane` como si viniera automáticamente en las máscaras semánticas/instancias de 10K. Si se incorpora carril, deberá prepararse un flujo separado con 100K Images y lane marking labels.

## Metadatos ambientales

La documentación indica atributos `weather`, `scene` y `timeofday` en los JSON de anotación de detección. Sin embargo, las imágenes 10K de segmentación no son un subconjunto exacto de las imágenes 100K. Se debe cruzar por nombre/ID, calcular la intersección y reportar cobertura; las imágenes sin correspondencia deben conservar una categoría `unknown`. No inferir lluvia, niebla, día o noche a partir del nombre del archivo.

## Notebook

Abrir en Colab: https://colab.research.google.com/github/etrabajo027-a11y/Imagen/blob/main/notebooks/01_exploracion_datos.ipynb

El notebook descarga, verifica los checksums de etiquetas, extrae y explora la estructura. Los datos se guardan en Google Drive y no se versionan en Git.
