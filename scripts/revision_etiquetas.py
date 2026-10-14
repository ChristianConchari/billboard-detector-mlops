"""Revisión de las cajas `pending` del dataset COCO en Label Studio.

Flujo:
    docker compose --profile labeling up -d
    python scripts/revision_etiquetas.py importar   # crea el proyecto y carga las tareas
    # revisar en http://localhost:8090: dejar la caja (confirmar), moverla (ajustar)
    # o borrarla (rechazar) y pulsar Submit
    python scripts/revision_etiquetas.py exportar   # escribe annotations_revisadas/

Solo se cargan las imágenes con al menos una caja `pending`; las cajas ya revisadas
de esas imágenes también se muestran. Al exportar, cada caja original recibe un
veredicto comparando su geometría con la enviada:
    sin cambios (≤ 1 px) -> confirmed · movida -> adjusted · borrada -> rejected
Las cajas nuevas se agregan con veredicto `added`. Las tareas sin enviar (o con
Skip) quedan como estaban.

Solo usa la biblioteca estándar, para correrlo desde el host sin instalar nada.
"""

import argparse
import http.cookiejar
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
RAIZ_CONTENEDOR = "/datasets"  # LOCAL_FILES_DOCUMENT_ROOT del servicio label-studio
SPLITS = ("train", "val", "test")
DATASET_POR_DEFECTO = "billboards-2025-09-14-sur-lp"
VERSION_MODELO = "origen"  # nombre de las pre-anotaciones importadas
TOLERANCIA_PX = 1.0
IOU_MINIMO = 0.5

CONFIG_ETIQUETADO = """
<View>
  <Header value="$titulo"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="box" toName="image">
    <Label value="billboard" background="#e8590c"/>
  </RectangleLabels>
</View>
"""


def leer_env():
    """Lee el .env del repo sin pisar variables ya definidas en el entorno."""
    env = {}
    ruta = RAIZ / ".env"
    if ruta.exists():
        for linea in ruta.read_text().splitlines():
            if "=" in linea and not linea.lstrip().startswith("#"):
                clave, valor = linea.split("=", 1)
                env[clave.strip()] = valor.strip()
    env.update(os.environ)
    return env


class LabelStudio:
    """Cliente mínimo de la API usando sesión (los tokens legacy vienen deshabilitados)."""

    def __init__(self, url, usuario, clave):
        self.url = url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        self.http = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookies))
        self.http.open(f"{self.url}/user/login/")
        datos = urllib.parse.urlencode(
            {"email": usuario, "password": clave, "csrfmiddlewaretoken": self._csrf()}
        ).encode()
        req = urllib.request.Request(f"{self.url}/user/login/", data=datos,
                                     headers={"Referer": f"{self.url}/user/login/"})
        self.http.open(req)
        if not any(c.name == "sessionid" for c in self.cookies):
            sys.exit(f"No se pudo iniciar sesión en {self.url} como {usuario}")

    def _csrf(self):
        return next((c.value for c in self.cookies if c.name == "csrftoken"), "")

    def api(self, metodo, ruta, cuerpo=None):
        req = urllib.request.Request(
            f"{self.url}/api/{ruta}",
            method=metodo,
            data=None if cuerpo is None else json.dumps(cuerpo).encode(),
            headers={"Content-Type": "application/json", "X-CSRFToken": self._csrf(),
                     "Referer": self.url},
        )
        try:
            with self.http.open(req) as resp:
                contenido = resp.read()
        except urllib.error.HTTPError as e:
            sys.exit(f"{metodo} /api/{ruta} -> {e.code}: {e.read().decode()[:500]}")
        return json.loads(contenido) if contenido else None

    def proyecto(self, titulo):
        respuesta = self.api("GET", f"projects?title={urllib.parse.quote(titulo)}&page_size=100")
        proyectos = respuesta["results"] if isinstance(respuesta, dict) else respuesta
        return next((p for p in proyectos if p["title"] == titulo), None)


def conectar():
    env = leer_env()
    url = env.get("LABEL_STUDIO_URL", f"http://localhost:{env.get('LABEL_STUDIO_PORT', '8090')}")
    return LabelStudio(url, env.get("LABEL_STUDIO_USERNAME", "admin@vallavision.local"),
                       env.get("LABEL_STUDIO_PASSWORD", "labelstudio"))


def titulo_proyecto(dataset):
    return f"Revisión {dataset}"


def cargar_coco(dataset, carpeta="annotations"):
    return {s: json.loads((DATOS / dataset / carpeta / f"{s}.json").read_text()) for s in SPLITS}


def cajas_por_imagen(coco):
    cajas = {}
    for a in coco["annotations"]:
        cajas.setdefault(a["image_id"], []).append(a)
    return cajas


def id_region(split, ann_id):
    return f"{split}-{ann_id}"


def a_porcentaje(bbox, ancho, alto):
    x, y, w, h = bbox
    return {"x": x / ancho * 100, "y": y / alto * 100,
            "width": w / ancho * 100, "height": h / alto * 100, "rotation": 0}


def a_pixeles(valor, ancho, alto):
    return [valor["x"] / 100 * ancho, valor["y"] / 100 * alto,
            valor["width"] / 100 * ancho, valor["height"] / 100 * alto]


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def importar(args):
    coco = cargar_coco(args.dataset)
    tareas, total_pendientes = [], 0
    for split in SPLITS:
        cajas = cajas_por_imagen(coco[split])
        imagenes = sorted(coco[split]["images"], key=lambda i: (i["video_id"], i["frame_number"]))
        for img in imagenes:
            visibles = [a for a in cajas.get(img["id"], []) if a["verdict"] != "rejected"]
            pendientes = sum(a["verdict"] == "pending" for a in visibles)
            if not pendientes:
                continue
            total_pendientes += pendientes
            resultado = [{
                "id": id_region(split, a["id"]),
                "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                "original_width": img["width"], "original_height": img["height"],
                "value": {**a_porcentaje(a["bbox"], img["width"], img["height"]),
                          "rectanglelabels": ["billboard"]},
            } for a in visibles]
            tareas.append({
                "data": {
                    "image": f"/data/local-files/?d={args.dataset}/images/{img['file_name']}",
                    "titulo": f"{split} · {img['file_name']} · {pendientes} de {len(visibles)} cajas pendientes",
                    "split": split, "image_id": img["id"], "file_name": img["file_name"],
                },
                "predictions": [{"model_version": VERSION_MODELO, "result": resultado}],
            })

    ls = conectar()
    titulo = titulo_proyecto(args.dataset)
    proyecto = ls.proyecto(titulo)
    if proyecto is None:
        proyecto = ls.api("POST", "projects", {
            "title": titulo,
            "description": "Confirmar (dejar), ajustar (mover) o rechazar (borrar) cada caja y pulsar Submit.",
            "label_config": CONFIG_ETIQUETADO,
            "show_collab_predictions": True,  # las cajas existentes aparecen pre-dibujadas
            "model_version": VERSION_MODELO,
        })
        ls.api("POST", "storages/localfiles", {
            "project": proyecto["id"], "title": args.dataset,
            "path": f"{RAIZ_CONTENEDOR}/{args.dataset}/images", "use_blob_urls": False,
        })
        print(f"Proyecto creado: {titulo} (id {proyecto['id']})")
    elif proyecto.get("task_number"):
        print(f"El proyecto {titulo} ya tiene {proyecto['task_number']} tareas; no se importa de nuevo.")
        print(f"Abrir: {ls.url}/projects/{proyecto['id']}/data")
        return

    ls.api("POST", f"projects/{proyecto['id']}/import", tareas)
    print(f"Importadas {len(tareas)} imágenes con {total_pendientes} cajas pendientes.")
    print(f"Abrir: {ls.url}/projects/{proyecto['id']}/data")


def emparejar(originales, enviadas):
    """Empareja cajas originales con las enviadas: primero por id de región, luego por IoU."""
    pares, libres = {}, dict(enviadas)
    for ann_id, (region, _) in originales.items():
        if region in libres:
            pares[ann_id] = libres.pop(region)
    candidatos = sorted(
        ((iou(caja, libres[r]), ann_id, r) for ann_id, (_, caja) in originales.items()
         if ann_id not in pares for r in libres),
        reverse=True,
    )
    for valor, ann_id, region in candidatos:
        if valor < IOU_MINIMO:
            break
        if ann_id not in pares and region in libres:
            pares[ann_id] = libres.pop(region)
    return pares, list(libres.values())


def exportar(args):
    ls = conectar()
    titulo = titulo_proyecto(args.dataset)
    proyecto = ls.proyecto(titulo)
    if proyecto is None:
        sys.exit(f"No existe el proyecto {titulo}; correr primero `importar`.")
    tareas = ls.api("GET", f"projects/{proyecto['id']}/export?exportType=JSON&download_all_tasks=true")

    revisadas = {}  # (split, image_id) -> {region: bbox en px}
    for tarea in tareas:
        enviadas = [a for a in tarea.get("annotations", []) if not a.get("was_cancelled")]
        if not enviadas:
            continue
        ultima = max(enviadas, key=lambda a: (a.get("updated_at") or "", a["id"]))
        cajas = {}
        for i, r in enumerate(ultima["result"]):
            if r.get("type") == "rectanglelabels":
                cajas[r.get("id") or f"nueva-{i}"] = a_pixeles(r["value"], r["original_width"], r["original_height"])
        revisadas[(tarea["data"]["split"], tarea["data"]["image_id"])] = cajas

    coco = cargar_coco(args.dataset)
    resumen = {}
    for split in SPLITS:
        datos = coco[split]
        cajas = cajas_por_imagen(datos)
        siguiente_id = max((a["id"] for s in SPLITS for a in coco[s]["annotations"]), default=0) + 1
        conteo = {"imagenes_revisadas": 0, "confirmed": 0, "adjusted": 0, "rejected": 0, "added": 0}
        for image_id, enviadas in revisadas.items():
            if image_id[0] != split:
                continue
            image_id = image_id[1]
            conteo["imagenes_revisadas"] += 1
            originales = {a["id"]: (id_region(split, a["id"]), a["bbox"])
                          for a in cajas.get(image_id, []) if a["verdict"] != "rejected"}
            pares, nuevas = emparejar(originales, enviadas)
            for a in cajas.get(image_id, []):
                if a["id"] not in originales:
                    continue
                if a["id"] not in pares:
                    a.update(verdict="rejected", iscrowd=1, reject_reason="other")
                elif max(abs(p - q) for p, q in zip(a["bbox"], pares[a["id"]])) > TOLERANCIA_PX:
                    a.update(verdict="adjusted", bbox=pares[a["id"]],
                             area=pares[a["id"]][2] * pares[a["id"]][3])
                elif a["verdict"] == "pending":
                    a["verdict"] = "confirmed"
                else:
                    continue  # ya revisada y sin cambios
                conteo[a["verdict"]] += 1
            for bbox in nuevas:
                datos["annotations"].append({
                    "id": siguiente_id, "image_id": image_id, "category_id": 1, "bbox": bbox,
                    "area": bbox[2] * bbox[3], "iscrowd": 0, "registry_version_id": None,
                    "verdict": "added", "reject_reason": None, "origin_type": "VALLAS",
                    "original_bbox": None,
                })
                siguiente_id += 1
                conteo["added"] += 1
        conteo["pending_restantes"] = sum(a["verdict"] == "pending" for a in datos["annotations"])
        resumen[split] = conteo

    salida = DATOS / args.dataset / "annotations_revisadas"
    salida.mkdir(exist_ok=True)
    for split in SPLITS:
        (salida / f"{split}.json").write_text(json.dumps(coco[split], ensure_ascii=False))
    (salida / "revision.json").write_text(json.dumps({
        "exportado_en": datetime.now(timezone.utc).isoformat(),
        "label_studio_proyecto": proyecto["id"],
        "resumen": resumen,
    }, indent=2, ensure_ascii=False))

    print(f"{'split':<6} {'imgs':>5} {'confirm':>8} {'ajust':>6} {'rechaz':>7} {'nuevas':>7} {'pend':>5}")
    for split, c in resumen.items():
        print(f"{split:<6} {c['imagenes_revisadas']:>5} {c['confirmed']:>8} {c['adjusted']:>6} "
              f"{c['rejected']:>7} {c['added']:>7} {c['pending_restantes']:>5}")
    print(f"Escrito en {salida.relative_to(RAIZ)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("accion", choices=["importar", "exportar"])
    parser.add_argument("--dataset", default=DATASET_POR_DEFECTO,
                        help=f"carpeta dentro de data/ (por defecto {DATASET_POR_DEFECTO})")
    args = parser.parse_args()
    {"importar": importar, "exportar": exportar}[args.accion](args)


if __name__ == "__main__":
    main()
