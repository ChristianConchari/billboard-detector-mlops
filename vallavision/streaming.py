"""Ventanas tumbling por event time con watermark (Sesión 4), sin dependencias.

El consumidor de Redpanda agrupa los frames por cámara en ventanas fijas de
`size_ms` según el instante en que la cámara los capturó (event time), no según cuándo
llegaron. El watermark es el event time más alto visto menos `allowed_lateness_ms`:
cuando pasa el fin de una ventana, la ventana se cierra y se emite. Un frame que llega
para una ventana ya cerrada se cuenta como tardío y se descarta.
"""

from collections import defaultdict


class TumblingWindows:
    def __init__(self, size_ms, allowed_lateness_ms=0):
        self.size_ms = size_ms
        self.allowed_lateness_ms = allowed_lateness_ms
        self._open = defaultdict(list)  # (clave, inicio) -> eventos
        self._watermark = {}  # clave -> watermark
        self.late = defaultdict(int)  # clave -> frames descartados por tardíos

    def window_start(self, event_time_ms):
        return event_time_ms - event_time_ms % self.size_ms

    def watermark(self, key):
        return self._watermark.get(key)

    def add(self, key, event_time_ms, event):
        """Agrega un evento; devuelve las ventanas que quedaron cerradas: [(clave, inicio, fin, eventos)]."""
        start = self.window_start(event_time_ms)
        wm = self._watermark.get(key)
        if wm is not None and start + self.size_ms <= wm:
            self.late[key] += 1
            return []
        self._open[(key, start)].append(event)
        new_wm = event_time_ms - self.allowed_lateness_ms
        if wm is None or new_wm > wm:
            self._watermark[key] = new_wm
        return self._close(key)

    def _close(self, key):
        wm = self._watermark[key]
        closed = []
        for (k, start) in sorted(self._open):
            if k == key and start + self.size_ms <= wm:
                closed.append((k, start, start + self.size_ms, self._open.pop((k, start))))
        return closed

    def flush(self):
        """Cierra todas las ventanas abiertas (fin del stream)."""
        closed = [(k, s, s + self.size_ms, ev) for (k, s), ev in sorted(self._open.items())]
        self._open.clear()
        return closed
