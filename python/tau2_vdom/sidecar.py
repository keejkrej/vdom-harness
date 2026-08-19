"""Long-lived Node sidecar that walks the vdom AgentGraph for one τ² turn."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR = REPO_ROOT / "src" / "eval" / "tau2-sidecar.ts"


class SidecarError(RuntimeError):
    pass


class VdomSidecar:
    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._n = 0

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        npx = shutil.which("npx")
        if npx is None:
            raise SidecarError("npx not found; install Node.js >= 18")
        script = self.repo_root / "src" / "eval" / "tau2-sidecar.ts"
        if not script.is_file():
            raise SidecarError(f"sidecar missing: {script}")
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [npx, "--yes", "tsx", str(script)],
            cwd=str(self.repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        # Drain stderr so a full pipe cannot deadlock the child.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for _line in proc.stderr:
            pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def request(self, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise SidecarError("sidecar not started")
        with self._lock:
            self._n += 1
            payload = {**payload, "id": payload.get("id") or f"t{self._n}"}
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        if not line:
            raise SidecarError("sidecar closed stdout")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SidecarError(f"sidecar returned non-json: {line!r}") from exc
        if data.get("op") == "error":
            raise SidecarError(data.get("error") or "sidecar error")
        return data


_default: VdomSidecar | None = None
_default_lock = threading.Lock()


def default_sidecar() -> VdomSidecar:
    global _default
    with _default_lock:
        if _default is None:
            _default = VdomSidecar()
        return _default
