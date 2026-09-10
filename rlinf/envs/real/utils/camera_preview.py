# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np


class CameraPreview:
    """Localhost JPEG preview for the latest RGB camera frames."""

    def __init__(self, port: int = 8080) -> None:
        self._latest: tuple[Mapping[str, object] | None, float | None] = (None, None)
        self._stopped = False
        self._stop_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(1.0)

            def do_GET(self) -> None:
                owner._handle_get(self)

            def log_message(self, fmt: str, *args: object) -> None:
                del fmt, args

        self._server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}/"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"CameraPreview:{self.port}",
            daemon=True,
        )
        self._thread.start()

    def put_frame(self, frames: Mapping[str, object]) -> None:
        self._latest = (frames, time.monotonic())

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2.0)

    def _handle_get(self, request: BaseHTTPRequestHandler) -> None:
        path = request.path.split("?", 1)[0]
        if path == "/frame.jpg":
            payload = self._render_jpeg()
            content_type = "image/jpeg"
        else:
            payload = self._render_page()
            content_type = "text/html; charset=utf-8"
        request.send_response(200)
        request.send_header("Content-Type", content_type)
        request.send_header("Cache-Control", "no-store")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)

    def _render_page(self) -> bytes:
        return b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Camera Preview</title>
<style>body{margin:0;background:#111;color:#eee;font:14px/1.4 system-ui,sans-serif}
main{min-height:100vh;display:grid;place-items:center;padding:24px;box-sizing:border-box}
img{max-width:100%;height:auto;border:1px solid #333;background:#222}
#status{margin-bottom:12px;color:#bbb}</style></head><body><main><div><div id="status">Connecting...</div>
<img id="preview" alt="camera preview"></div></main>
<script>
const img=document.getElementById('preview');
const status=document.getElementById('status');
function next(){setTimeout(()=>{img.src='/frame.jpg?t='+Date.now()},100)}
img.onload=()=>{status.textContent='Connected';next()};
img.onerror=()=>{status.textContent='Connection interrupted';next()};
next();
</script></body></html>"""

    def _render_jpeg(self) -> bytes:
        frames, timestamp = self._latest
        age = None if timestamp is None else time.monotonic() - timestamp
        names = list(frames.keys())[:3] if frames else []
        canvas = np.full((320, 960, 3), 18, dtype=np.uint8)
        for index in range(3):
            if index < len(names):
                name = names[index]
                frame = frames[name] if frames is not None else None
                tile = self._tile(name, frame, age)
            else:
                tile = self._placeholder(f"camera_{index + 1}", "no frame")
            canvas[:, index * 320 : (index + 1) * 320] = tile
        ok, encoded = cv2.imencode(
            ".jpg", canvas[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return encoded.tobytes()

    def _tile(self, name: str, frame: object, age: float | None) -> np.ndarray:
        try:
            rgb = np.asarray(frame)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("expected HxWx3 RGB frame")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            resized = self._fit_320(rgb)
            tile = np.full((320, 320, 3), 18, dtype=np.uint8)
            y = (320 - resized.shape[0]) // 2
            x = (320 - resized.shape[1]) // 2
            tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            age_label = "waiting" if age is None else f"Obs age {age:.2f}s"
            return self._label(tile, f"{name}  {age_label}")
        except (ValueError, TypeError, cv2.error):
            return self._placeholder(str(name), "invalid frame")

    def _fit_320(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        scale = min(320.0 / max(width, 1), 320.0 / max(height, 1))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    def _placeholder(self, name: str, text: str) -> np.ndarray:
        return self._label(
            np.full((320, 320, 3), 28, dtype=np.uint8), f"{name}  {text}"
        )

    def _label(self, image: np.ndarray, text: str) -> np.ndarray:
        label = str(text)[:80]
        cv2.rectangle(image, (0, 0), (319, 24), (0, 0, 0), thickness=-1)
        cv2.putText(
            image,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image
