# Set de evaluación revisado

`val.json` es el split de validación **con cajas revisadas a mano** que se usó para entrenar y calibrar
el modelo `model-v1` en [vit-billboard-detector](https://github.com/ChristianConchari/vit-billboard-detector)
(`data/annotations/reviewed/val.json`, sha256 `48076fa5…`): 95 imágenes, 130 cajas, clase única `billboard`.

El DAG `registrar_modelo` evalúa contra este archivo y no contra `annotations/val.json` del export de origen, por dos motivos:

- el split por video del export no coincide con el del entrenamiento: 138 de las 169 imágenes de su `val` eran de `train` del modelo;
- 102 de las 169 cajas de ese `val` son pre-etiquetas `pending`, sin revisar.

Las imágenes son las del dataset `billboards-2025-09-14-sur-lp` (todas están en el zip). El DAG las busca por nombre
de archivo en `raw/images/<dataset>/`, sin importar el split del export en el que hayan caído.
