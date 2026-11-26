# Datos

Esta carpeta está reservada para los datos y sus metadatos.

## Conjuntos candidatos

- Cityscapes
- BDD100K
- KITTI

La elección se hará tras verificar que las anotaciones permitan cubrir las clases y tareas del proyecto. Consultar siempre las condiciones oficiales de descarga, licencia y uso.

## Organización sugerida

```text
data/
├── raw/        # Datos originales (no versionar)
├── processed/  # Datos preparados (no versionar)
└── README.md
```

No subir imágenes, etiquetas ni archivos comprimidos pesados a Git sin comprobar antes las condiciones de distribución y el tamaño. Documentar aquí la fuente, versión, particiones y clases del conjunto finalmente seleccionado.
