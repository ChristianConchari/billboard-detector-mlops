"""Llama a POST /v1/detect y dibuja las cajas sobre la imagen (para la demo).

    python scripts/ver_detecciones.py data/billboards-2025-09-14-sur-lp/images/<imagen>.jpg
    python scripts/ver_detecciones.py <imagen> --umbral 0.5 --salida /tmp/out.jpg --no-abrir

Lee la API key de `.env` (GATEWAY_API_KEYS) sin mostrarla. Guarda la imagen anotada y la abre
con el visor del sistema. Solo necesita Pillow (el HTTP es de la biblioteca estándar).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
COLOR = (232, 89, 12)  # naranja: contrasta con cielo y asfalto


def env(name):
    if os.getenv(name):
        return os.getenv(name)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


def detect(url, api_key, image_path, threshold):
    data = Path(image_path).read_bytes()
    ctype = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{Path(image_path).name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    query = f"?score_threshold={threshold}" if threshold is not None else ""
    req = urllib.request.Request(f"{url}/v1/detect{query}", data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "X-API-Key": api_key or ""})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}")


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(image_path, result):
    # Las cajas vienen sobre la imagen tal como se ve (EXIF ya aplicado por el servicio)
    image = ImageOps.exif_transpose(Image.open(image_path))
    image = image.convert("RGB")
    d = ImageDraw.Draw(image)
    width = max(3, image.width // 400)
    f = font(max(18, image.width // 60))
    for det in result["detections"]:
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=COLOR, width=width)
        label = f"{det['label']} {det['score']:.2f}"
        tw, th = d.textbbox((0, 0), label, font=f)[2:]
        ty = y0 - th - 2 * width if y0 - th - 2 * width > 0 else y1
        tx = min(x0, image.width - tw - 2 * width)  # que la etiqueta no se corte en el borde
        d.rectangle([tx, ty, tx + tw + 2 * width, ty + th + 2 * width], fill=COLOR)
        d.text((tx + width, ty + width // 2), label, fill="white", font=f)
    m = result["model"]
    banner = (f"{m['name']} v{m['version']} ({m['alias']}) · {len(result['detections'])} vallas · "
              f"{result['latency_ms']:.0f} ms")
    bf = font(max(16, image.width // 75))
    bw, bh = d.textbbox((0, 0), banner, font=bf)[2:]
    d.rectangle([0, 0, bw + 24, bh + 16], fill=(0, 0, 0))
    d.text((12, 8), banner, fill="white", font=bf)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen")
    ap.add_argument("--umbral", type=float, help="score_threshold (por defecto el calibrado del modelo)")
    ap.add_argument("--url", default=f"http://localhost:{env('FASTAPI_PORT') or 8800}")
    ap.add_argument("--salida", help="ruta de la imagen anotada (por defecto un temporal)")
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el visor")
    args = ap.parse_args()

    result = detect(args.url, (env("GATEWAY_API_KEYS") or "").split(",")[0], args.imagen, args.umbral)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = args.salida or os.path.join(tempfile.gettempdir(), f"vallavision-{result['request_id'][:8]}.jpg")
    draw(args.imagen, result).save(out, quality=92)
    print(f"\nimagen anotada: {out}", file=sys.stderr)
    if not args.no_abrir:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
