# Sets revisados

Splits con **cajas revisadas a mano** del repo del modelo
[vit-billboard-detector](https://github.com/ChristianConchari/vit-billboard-detector)
(`data/annotations/reviewed/`), con el mismo split por video con el que se entrenó `model-v1`.
Clase única `billboard`. `val` y `test` tienen videos disjuntos (y disjuntos del `train` del modelo).

| Archivo | Imágenes | Cajas | Videos | sha256 | Uso en este repo |
|---|---|---|---|---|---|
| `test.json` | 95 | 126 | 7 | `11bd28e3…` | **gate** de `registrar_modelo`: no intervino ni en el entrenamiento ni en la calibración del umbral |
| `val.json` | 95 | 130 | 7 | `48076fa5…` | **referencia de drift**. Se usó para elegir el checkpoint y calibrar el umbral, así que no sirve como gate |

No se usan las anotaciones del export de origen (`data/<dataset>/annotations/`) para evaluar:
su split por video no coincide con el del entrenamiento (138 de las 169 imágenes de su `val` eran
de `train` del modelo) y 102 de las 169 cajas de su `val` son pre-etiquetas `pending`, sin revisar.

Las imágenes son las del dataset `billboards-2025-09-14-sur-lp` (todas están en el zip). Los DAGs
las buscan por nombre de archivo en `raw/images/<dataset>/`, sin importar el split del export en
el que hayan caído (y fallan si un nombre se repite entre splits).
