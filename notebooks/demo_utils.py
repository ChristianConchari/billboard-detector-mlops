"""Funciones de la demo en notebook (notebooks/demo_vallavision.ipynb).

La lógica vive acá para que las celdas del notebook sean cortas y se lean en el video. Los secretos
se leen de `.env` y nunca se imprimen. Requiere el stack arriba y con champion.
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import camara_productor  # noqa: E402
import ver_detecciones  # noqa: E402

if not (ROOT / "vallavision" / "grpc_gen" / "detector_pb2_grpc.py").exists():
    subprocess.run(["sh", str(ROOT / "scripts" / "gen_proto.sh")], check=True,
                   env={**os.environ, "PYTHON": sys.executable})

import grpc  # noqa: E402

from vallavision import drift  # noqa: E402
from vallavision.grpc_gen import detector_pb2 as pb  # noqa: E402
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc  # noqa: E402

IMAGENES = ROOT / "data" / "billboards-2025-09-14-sur-lp" / "images"
GATEWAY = f"http://localhost:{ver_detecciones.env('FASTAPI_PORT') or 8800}"
GRPC = "localhost:50051"
BUCKET = "data"
VENTANA_MS = 15000  # la del consumidor (STREAM_WINDOW_MS)

# Paleta validada (la de las figuras del README)
AZUL, VERDE, NARANJA = "#2a78d6", "#1baf7a", "#eb6834"
TINTA, GRIS, GRILLA, FONDO = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "axes.edgecolor": GRILLA,
                     "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _clave():
    return (ver_detecciones.env("GATEWAY_API_KEYS") or "").split(",")[0]


def _token():
    return ver_detecciones.env("INFERENCE_TOKEN") or ""


def imagen(nombre):
    return IMAGENES / nombre


# --- 0 · estado ------------------------------------------------------------------------

def estado():
    """El modelo que está sirviendo el gateway, como tabla."""
    salud = json.loads(urllib.request.urlopen(f"{GATEWAY}/health", timeout=5).read())
    req = urllib.request.Request(f"{GATEWAY}/v1/model", headers={"X-API-Key": _clave()})
    m = json.loads(urllib.request.urlopen(req, timeout=5).read())
    met = m["metrics"]
    filas = {
        "Modelo": f"{m['name']} v{m['version']} ({m['alias']})",
        "Cargado": "sí" if salud["model_loaded"] else "no",
        "Umbral calibrado": m["score_threshold"],
        "Gate: mAP@[.5:.95]": f"{met['gate_map']:.3f}  (IC 95 % {met['gate_map_ci_low']:.3f}–{met['gate_map_ci_high']:.3f})",
        "Gate: AP50": f"{met['gate_map_50']:.3f}",
        "Set del gate": m["eval_set"],
    }
    return pd.DataFrame(filas.items(), columns=["", "valor"]).set_index("")


# --- 1 · inferencia REST ---------------------------------------------------------------

def detectar(nombre, umbral=None):
    """POST /v1/detect por el gateway; devuelve el JSON del contrato."""
    return ver_detecciones.detect(GATEWAY, _clave(), imagen(nombre), umbral)


def mostrar(nombre, resultado, ancho=11):
    """La imagen con las cajas dibujadas, inline."""
    fig, ax = plt.subplots(figsize=(ancho, ancho * 9 / 16))
    ax.imshow(ver_detecciones.draw(imagen(nombre), resultado))
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()


def tabla(resultado):
    """Las detecciones como tabla."""
    filas = [{"valla": i + 1, "score": d["score"], "caja (x0, y0, x1, y1)": d["box"]}
             for i, d in enumerate(resultado["detections"])]
    return pd.DataFrame(filas).set_index("valla")


def mosaico(nombres):
    """Varias calles en una grilla de 2 columnas, cada una con su resultado."""
    filas = (len(nombres) + 1) // 2
    fig, axes = plt.subplots(filas, 2, figsize=(14, 4.3 * filas))
    for ax, nombre in zip(np.ravel(axes), nombres):
        r = detectar(nombre)
        ax.imshow(ver_detecciones.draw(imagen(nombre), r))
        ax.set_title(f"recorrido {nombre[:8]} · {len(r['detections'])} vallas · {r['latency_ms']:.0f} ms",
                     loc="left", color=TINTA, fontsize=11)
        ax.axis("off")
    for ax in np.ravel(axes)[len(nombres):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# --- 2 · gRPC --------------------------------------------------------------------------

def comparar_protocolos(nombre, repeticiones=5):
    """La misma imagen por REST (gateway) y por gRPC directo a inference: bytes y latencia."""
    data = imagen(nombre).read_bytes()
    stub = pb_grpc.DetectorStub(grpc.insecure_channel(GRPC))
    md = [("authorization", f"Bearer {_token()}")]
    request = pb.DetectRequest(image=data)

    def medir(fn):
        fn()  # calentamiento
        ms = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ms))

    rest_ms = medir(lambda: detectar(nombre))
    grpc_ms = medir(lambda: stub.Detect(request, timeout=30, metadata=md))
    b64 = len(json.dumps({"image": base64.b64encode(data).decode()}).encode())
    df = pd.DataFrame([
        {"camino": "REST multipart (gateway → gRPC → inference)", "bytes del request": len(data), "latencia p50 (ms)": rest_ms},
        {"camino": "gRPC directo (bytes)", "bytes del request": request.ByteSize(), "latencia p50 (ms)": grpc_ms},
        {"camino": "JSON + base64 (hipotético)", "bytes del request": b64, "latencia p50 (ms)": None},
    ]).set_index("camino")

    fig, ax = plt.subplots(figsize=(9, 2.6))
    colores = [AZUL, VERDE, NARANJA]
    ax.barh(df.index[::-1], df["bytes del request"][::-1] / 1024, color=colores[::-1], height=0.55)
    for y, v in enumerate(df["bytes del request"][::-1] / 1024):
        ax.text(v + 4, y, f"{v:,.0f} KB", va="center", color=TINTA, fontsize=10)
    ax.set_xlabel("tamaño del request (KB)")
    ax.set_title(f"base64 agrega {b64 / len(data) - 1:+.0%}: gRPC y multipart mandan los bytes tal cual",
                 loc="left", color=TINTA, fontsize=12, fontweight="bold")
    ax.grid(axis="x", color=GRILLA)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()
    return df.style.format({"bytes del request": "{:,.0f}", "latencia p50 (ms)": lambda v: "—" if pd.isna(v) else f"{v:.0f}"})


# --- 3 · seguridad ---------------------------------------------------------------------

def _post(path, data=None, ctype="image/jpeg", clave=None, metodo="POST"):
    headers = {}
    body = None
    if data is not None:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    if clave is not None:
        headers["X-API-Key"] = clave
    req = urllib.request.Request(f"{GATEWAY}{path}", data=body, method=metodo, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}").get("detail", "")


def pruebas_de_seguridad(nombre):
    """Cada ataque contra el gateway y contra inference, con el código esperado y el obtenido."""
    buena = imagen(nombre).read_bytes()
    gif = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(gif, "GIF")
    bomba = io.BytesIO()
    Image.new("1", (20000, 20000)).save(bomba, "PNG")
    clave = _clave()
    casos = [
        ("imagen válida, con API key", 200, _post("/v1/detect", buena, clave=clave)),
        ("sin API key", 401, _post("/v1/detect", buena)),
        ("API key inválida", 403, _post("/v1/detect", buena, clave="no-es-mia")),
        ("GIF renombrado y declarado image/jpeg", 415, _post("/v1/detect", gif.getvalue(), clave=clave)),
        (f"PNG de 20000×20000 ({len(bomba.getvalue()) // 1024} KB): bomba de descompresión", 422,
         _post("/v1/detect", bomba.getvalue(), "image/png", clave=clave)),
        ("imagen de más de 5 MB", 413, _post("/v1/detect", b"\xff\xd8" + b"0" * (5 * 1024 * 1024 + 10), clave=clave)),
        ("recargar el modelo con clave de usuario", 403, _post("/v1/reload", clave=clave)),
    ]
    stub = pb_grpc.DetectorStub(grpc.insecure_channel(GRPC))
    try:
        stub.GetModel(pb.GetModelRequest(), timeout=5)
        grpc_res = ("OK", "")
    except grpc.RpcError as e:
        grpc_res = (e.code().name, e.details())
    casos.append(("gRPC directo a inference sin token", "UNAUTHENTICATED", grpc_res))
    filas = [{"caso": c, "esperado": esp, "obtenido": got[0], "": "✔" if got[0] == esp else "✘",
              "detalle": got[1]} for c, esp, got in casos]
    return pd.DataFrame(filas).set_index("caso")


# --- 4 y 5 · streaming -----------------------------------------------------------------

def _s3():
    import boto3

    return boto3.client("s3", endpoint_url="http://localhost:9000",
                        aws_access_key_id=ver_detecciones.env("MINIO_ACCESS_KEY") or "minio",
                        aws_secret_access_key=ver_detecciones.env("MINIO_SECRET_ACCESS_KEY"))


def transmitir(degradar=None, camaras=3, por_segundo=2, vueltas=1):
    """Simula vehículos de relevamiento: publica sus frames en Redpanda (clave = recorrido, encabezado event_time_ms).

    Devuelve (inicio, fin) en UTC para buscar después lo que produjo este stream.
    """
    from kafka import KafkaProducer

    cams = camara_productor.cameras(ROOT / "annotations" / "reviewed" / "test.json", IMAGENES, camaras)
    eventos = camara_productor.schedule(cams, fps=2, loops=vueltas, late=0)
    # El primer frame arranca una ventana de 15 s del consumidor (si no, la primera ventana queda con 1 frame)
    primero = min(e[1] for e in eventos)
    corrimiento = (VENTANA_MS - primero % VENTANA_MS) % VENTANA_MS + 500
    eventos = [(c, t + corrimiento, f, r) for c, t, f, r in eventos]
    productor = KafkaProducer(bootstrap_servers="localhost:19092", max_request_size=8 * 1024 * 1024)
    inicio = datetime.now(timezone.utc)
    print(f"{len(eventos)} frames de {len(cams)} recorridos ({', '.join(cams)})"
          f"{' · ' + degradar if degradar else ''}", flush=True)
    for i, (cam, t_evento, frame_id, ruta) in enumerate(eventos, 1):
        t0 = time.perf_counter()
        productor.send("frames", key=cam.encode(), value=camara_productor.encode(ruta, degradar),
                       headers=[("event_time_ms", str(t_evento).encode()), ("frame_id", frame_id.encode())])
        if i % 20 == 0 or i == len(eventos):
            print(f"  {i}/{len(eventos)} publicados", flush=True)
        time.sleep(max(0.0, 1 / por_segundo - (time.perf_counter() - t0)))
    productor.flush()
    return inicio, datetime.now(timezone.utc)


def ventanas(stream, espera_max=45):
    """Las ventanas que cerró el consumidor para este stream (espera el cierre por inactividad)."""
    inicio, fin = stream
    s3 = _s3()
    vistas, quieto_desde = {}, time.monotonic()
    limite = time.monotonic() + espera_max
    while time.monotonic() < limite:
        nuevas = 0
        dia = inicio.date()
        while dia <= fin.date():
            prefijo = f"curated/stream_windows/date={dia:%Y-%m-%d}/"
            for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefijo):
                for o in pag.get("Contents", []):
                    if o["LastModified"] >= inicio and o["Key"] not in vistas:
                        vistas[o["Key"]] = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
                        nuevas += 1
            dia += timedelta(days=1)
        if nuevas:
            quieto_desde = time.monotonic()
        elif vistas and time.monotonic() - quieto_desde > 15 and datetime.now(timezone.utc) > fin + timedelta(seconds=12):
            break
        time.sleep(3)
    filas = [{"recorrido": w["camera"],
              "ventana": f"{datetime.fromtimestamp(w['window'][0] / 1000):%H:%M:%S}–{datetime.fromtimestamp(w['window'][1] / 1000):%H:%M:%S}",
              "frames": w["frames"], "tardíos": w["late_frames"], "fps": w["throughput_fps"],
              "p50 ms": w["latency_ms_p50"], "p95 ms": w["latency_ms_p95"],
              "vallas/frame": w["detections_per_frame"], "alertas": "; ".join(w["alerts"]) or "—"}
             for w in vistas.values()]
    df = pd.DataFrame(filas).sort_values(["recorrido", "ventana"]).set_index(["recorrido", "ventana"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.2))
    etiquetas = [f"{c}\n{v.split('–')[0]}" for c, v in df.index]
    a1.bar(etiquetas, df["p95 ms"], color=AZUL, width=0.6)
    a1.axhline(2000, color=NARANJA, lw=1.5, ls="--")
    a1.text(len(etiquetas) - 0.5, 2000, "SLO 2 s", ha="right", va="bottom", color=TINTA, fontsize=9)
    a1.set_title("latencia p95 por ventana (ms)", loc="left", color=TINTA, fontsize=11)
    a2.bar(etiquetas, df["vallas/frame"], color=VERDE, width=0.6)
    a2.set_title("vallas detectadas por frame", loc="left", color=TINTA, fontsize=11)
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=8)
        a.grid(axis="y", color=GRILLA)
        a.set_axisbelow(True)
    fig.tight_layout()
    plt.show()
    return df


def particiones():
    """Particiones del topic y offsets confirmados por el grupo de consumidores (rpk en el contenedor)."""
    salida = subprocess.run(["docker", "exec", "redpanda", "rpk", "group", "describe", "detector"],
                            capture_output=True, text=True, check=True).stdout
    filas = []
    for linea in salida.splitlines():
        partes = linea.split()
        if partes and partes[0] == "frames":
            filas.append({"partición": int(partes[1]), "offset confirmado": partes[2],
                          "fin del log": partes[4], "lag": partes[5]})
    return pd.DataFrame(filas).set_index("partición")


# --- 6 · drift -------------------------------------------------------------------------

def _brillos(desde, hasta, version):
    s3 = _s3()
    valores = []
    dia = desde.date()
    while dia <= hasta.date():
        for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"predictions/date={dia:%Y-%m-%d}/"):
            for o in pag.get("Contents", []):
                if desde <= o["LastModified"] <= hasta:
                    r = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
                    if r["model"]["version"] == version and drift.usable_for_drift(r):
                        valores.append(r["image_stats"]["brightness"])
        dia += timedelta(days=1)
    return np.array(valores)


def comparar_brillo(normal, noche, version="1"):
    """Brillo de las predicciones de cada stream contra la referencia de drift (KS)."""
    from scipy.stats import gaussian_kde, ks_2samp

    s3 = _s3()
    refs = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="curated/reference/billboard-detector/")["Contents"])
    referencia = np.array(json.loads(s3.get_object(Bucket=BUCKET, Key=refs[-1])["Body"].read())["samples"]["brightness"])
    margen = timedelta(seconds=30)  # el consumidor infiere unos segundos después de publicar
    fin_normal = min(normal[1] + margen, noche[0])  # sin pisar el comienzo del recorrido de noche
    series = [("Referencia (val revisado)", referencia, AZUL, None),
              ("Recorrido normal", _brillos(normal[0], fin_normal, version), VERDE, (40, 4)),
              ("Recorrido de noche", _brillos(noche[0], noche[1] + margen, version), NARANJA, (0, 8))]

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.linspace(0, 0.8, 400)
    filas = []
    for nombre, v, color, desplazamiento in series:
        y = gaussian_kde(v)(x)
        ax.plot(x, y, color=color, lw=2, label=f"{nombre} ({len(v)} frames)")
        ax.fill_between(x, y, color=color, alpha=0.08, lw=0)
        if desplazamiento:
            p = ks_2samp(referencia, v).pvalue
            filas.append({"stream": nombre, "frames": len(v), "brillo medio": round(float(v.mean()), 3),
                          "p-valor KS vs referencia": f"{p:.2g}", "drift (α = 0.01)": "sí" if p < 0.01 else "no"})
            k = int(np.argmax(y))
            ax.annotate(f"p = {p:.2g}" + (" · drift" if p < 0.01 else " · sin drift"), (x[k], y[k]),
                        xytext=desplazamiento, textcoords="offset points",
                        ha="left" if desplazamiento[0] else "center", color=TINTA, fontsize=10)
    ax.set_xlabel("brillo medio del frame (luminancia, 0–1)")
    ax.set_yticks([])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.set_xlim(0, 0.8)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRILLA)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("Brillo del tráfico recién enviado contra la referencia de drift", loc="left",
                 color=TINTA, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(filas).set_index("stream")
