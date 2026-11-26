# Tema 2 — Detección y segmentación en escenas urbanas

Proyecto académico para desarrollar un sistema de percepción visual en escenas urbanas mediante **segmentación semántica multiclase** y **segmentación por instancias**.

## Dataset principal: BDD100K

Se selecciona **BDD100K** como conjunto principal. Para segmentación semántica y por instancias se utilizarán las **10K Images** y sus paquetes de anotaciones específicos. Cityscapes queda como posible conjunto complementario; KITTI no se incorpora en esta primera fase para mantener el alcance controlado.

**Importante sobre las tareas:** las anotaciones semánticas y de instancias de BDD100K corresponden al conjunto 10K. Las anotaciones de carriles (*lane marking*) y área transitable (*drivable area*) están asociadas al conjunto de 100K Images. No se asumirá que esas máscaras están disponibles para las mismas imágenes 10K. Además, las imágenes 10K no son un subconjunto exacto de las 100K, aunque existe solapamiento significativo. Los atributos ambientales `weather`, `scene` y `timeofday` deben cruzarse por ID/nombre y su cobertura real debe medirse; no se asignarán etiquetas ambientales por inferencia.

Documentación oficial: [BDD100K — descarga y estructura](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst). Los datos se descargan desde [el servidor oficial ETH](https://dl.cv.ethz.ch/bdd100k/data/). Al descargar, se deben aceptar y cumplir las condiciones oficiales de uso.

## Ejecutar en Google Colab

1. Abre el notebook de exploración: **[01_exploracion_datos.ipynb en Google Colab](https://colab.research.google.com/github/etrabajo027-a11y/Imagen/blob/main/notebooks/01_exploracion_datos.ipynb)**.
2. En Colab, selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU** si hay una disponible.
3. Ejecuta las celdas en orden y autoriza el montaje de Google Drive.
4. El notebook descarga imágenes 10K y anotaciones semánticas/de instancias, verifica los MD5 de los paquetes de etiquetas, extrae los archivos y genera un informe inicial en `MyDrive/Imagenes_Tema2`.

El notebook no descarga videos (1.8 TB), no entrena modelos y no sube datos al repositorio. Los recursos GPU y los límites de Colab pueden variar; guarda datos y checkpoints en Drive y comienza con subconjuntos pequeños.

## Objetivo

Identificar y segmentar elementos relevantes de escenas urbanas y estudiar el comportamiento del sistema bajo distintas condiciones ambientales (día, noche, lluvia y niebla), siempre que existan metadatos verificables para las imágenes evaluadas.

## Metodología prevista

1. Exploración y validación de imágenes y anotaciones.
2. Preprocesamiento: exposición, gamma y aumentos realistas cuando proceda.
3. Segmentación semántica con DeepLabv3+ como arquitectura de referencia.
4. Segmentación por instancias con Mask R-CNN como arquitectura de referencia.
5. Evaluación con mIoU y mAP, más análisis por condición ambiental cuando haya metadatos suficientes.
6. Análisis de errores, limitaciones y reproducibilidad.

## Estructura del repositorio

```text
.
├── config/
├── data/                  # Instrucciones; no contiene datos pesados
├── docs/
├── notebooks/             # Notebooks reproducibles de Colab
├── results/
└── src/
    ├── preprocessing/
    ├── semantic_segmentation/
    ├── instance_segmentation/
    └── evaluation/
```

## Reproducibilidad y alcance

Registrar dataset, partición, clases, resolución, transformaciones, arquitectura, pesos iniciales, hiperparámetros, semilla, métricas y condiciones ambientales. No reportar métricas hasta ejecutar los experimentos. Los datos originales y pesos grandes no deben subirse a GitHub.

## Estado

**Fase 1 — exploración:** notebook inicial creado; pendiente ejecutarlo en Colab y validar la estructura real descargada. Los resultados se incorporarán después de ejecutar los experimentos.
