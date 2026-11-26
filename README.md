# Tema 2 — Detección y segmentación en escenas urbanas

Proyecto académico para desarrollar un sistema de percepción visual en escenas urbanas mediante **segmentación semántica multiclase** y **segmentación por instancias**.

## Objetivo

Identificar y segmentar elementos relevantes de escenas urbanas —por ejemplo, peatones, vehículos, ciclistas, señales de tránsito y carriles— y estudiar el comportamiento del sistema bajo distintas condiciones ambientales, como día, noche, lluvia y niebla.

## Alcance de trabajo

1. **Datos y exploración:** seleccionar y documentar un conjunto de datos compatible con las clases y anotaciones requeridas.
2. **Preprocesamiento:** analizar las imágenes y aplicar, cuando corresponda, ajustes de exposición y gamma, además de aumentos de datos realistas.
3. **Segmentación semántica:** preparar y evaluar un modelo de referencia, tomando DeepLabv3+ como arquitectura recomendada para esta tarea.
4. **Segmentación por instancias:** preparar y evaluar un modelo de referencia, tomando Mask R-CNN como arquitectura recomendada.
5. **Evaluación:** medir la segmentación semántica con mIoU y la detección/segmentación de instancias con mAP, además de comparar resultados por condición ambiental.
6. **Análisis:** documentar limitaciones, errores frecuentes y oportunidades de mejora.

## Conjuntos de datos candidatos

- **Cityscapes**
- **BDD100K**
- **KITTI**

La selección final se realizará después de revisar las clases disponibles, el tipo de anotaciones, las condiciones ambientales y los requisitos de acceso/licencia. No se incluyen datos de entrenamiento en este repositorio.

## Estructura del repositorio

```text
.
├── config/                 # Configuración de experimentos
├── data/                   # Instrucciones y metadatos de los datos (no datos pesados)
├── docs/                   # Diseño, metodología y documentación
├── notebooks/              # Exploración y experimentos reproducibles
├── results/                # Métricas, tablas y visualizaciones
├── src/
│   ├── preprocessing/      # Preparación y aumentos de imágenes
│   ├── semantic_segmentation/ # Segmentación semántica
│   ├── instance_segmentation/ # Segmentación por instancias
│   └── evaluation/         # Métricas y análisis
├── .gitignore
└── requirements.txt
```

## Instalación inicial

Se recomienda crear un entorno virtual e instalar las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Estado del proyecto

**Fase inicial:** estructura y documentación base. Los modelos, experimentos y resultados se incorporarán y validarán progresivamente. No se reportarán métricas hasta ejecutar los experimentos correspondientes.

## Reproducibilidad

Cada experimento deberá registrar como mínimo: conjunto de datos y partición, clases utilizadas, resolución de entrada, transformaciones, arquitectura y pesos iniciales, hiperparámetros, semilla aleatoria, métricas y condición ambiental.

## Licencia y uso de datos

Antes de descargar o redistribuir un conjunto de datos, se revisarán sus términos oficiales de acceso, uso y atribución. Los datos originales y los pesos grandes de modelos no deben subirse al repositorio sin comprobar previamente sus condiciones.
