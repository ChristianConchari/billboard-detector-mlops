"""Event-time tumbling windows with a watermark (Session 4), no dependencies.

The Redpanda consumer groups frames per camera into fixed `size_ms` windows according to
the moment the camera captured them (event time), not when they arrived. The watermark is
the highest event time seen minus `allowed_lateness_ms`: once it passes the end of a
window, the window is closed and emitted. A frame arriving for a window that is already
closed is counted as late and dropped.
"""

from collections import defaultdict


class TumblingWindows:
    def __init__(self, size_ms, allowed_lateness_ms=0):
        self.size_ms = size_ms
        self.allowed_lateness_ms = allowed_lateness_ms
        self._open = defaultdict(list)  # (key, start) -> events
        self._watermark = {}  # key -> watermark
        self.late = defaultdict(int)  # key -> frames dropped for being late

    def window_start(self, event_time_ms):
        return event_time_ms - event_time_ms % self.size_ms

    def watermark(self, key):
        return self._watermark.get(key)

    def add(self, key, event_time_ms, event):
        """Adds an event; returns the windows it closed: [(key, start, end, events)]."""
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

    def flush_key(self, key):
        """Closes the open windows of one key (the camera stopped sending frames).

        The watermark only moves with new events: without this, the last windows of a
        camera that goes quiet would stay open until it transmits again. After the flush,
        anything arriving for those windows counts as late.
        """
        closed = [(k, s, s + self.size_ms, ev) for (k, s), ev in sorted(self._open.items()) if k == key]
        for k, s, _, _ in closed:
            del self._open[(k, s)]
        if closed:
            self._watermark[key] = max(self._watermark.get(key, 0), closed[-1][2])
        return closed

    def flush(self):
        """Closes every open window (end of the stream)."""
        closed = [(k, s, s + self.size_ms, ev) for (k, s), ev in sorted(self._open.items())]
        self._open.clear()
        return closed
