from __future__ import annotations

import socket
import time


def wait_for_http_server(host: str, port: int, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
    raise RuntimeError(f"HTTP test server did not become ready on {host}:{port}") from last_error
