#!/usr/bin/env python3
"""Cheap one-shot probe: isolated process + 8 min timeout. Output path via VDOM_PROBE_OUT."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "tau2"
OUT = Path(os.environ.get("VDOM_PROBE_OUT", str(EVAL / "probe-airline-0731.json")))
TASKS = sys.argv[1:] or ["0", "1", "2"]
TIMEOUT = int(os.environ.get("VDOM_TRIAL_TIMEOUT", "480"))
MAX_STEPS = int(os.environ.get("VDOM_MAX_STEPS", "80"))
DOMAIN = os.environ.get("VDOM_PROBE_DOMAIN", "airline")
VENV_PY = ROOT / ".tau2-venv" / "bin" / "python"


def main() -> int:
    EVAL.mkdir(parents=True, exist_ok=True)
    results = []
    for tid in TASKS:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        env["VDOM_TAU2_TECHNIQUE"] = "one-shot"
        cmd = [
            str(VENV_PY),
            "-m",
            "tau2_vdom",
            "--domain",
            DOMAIN,
            "--task-ids",
            tid,
            "--num-trials",
            "1",
            "--max-steps",
            str(MAX_STEPS),
            "--model",
            env.get("OPENAI_MODEL", "deepseek/deepseek-v4-flash-0731"),
        ]
        print(f"[probe] start domain={DOMAIN} task={tid} timeout={TIMEOUT}s max_steps={MAX_STEPS}", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                timeout=TIMEOUT,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            latest = EVAL / "latest.json"
            reward = None
            termination = None
            model = None
            if latest.is_file():
                payload = json.loads(latest.read_text())
                model = payload.get("model")
                sims = payload.get("simulations") or []
                if sims:
                    reward = sims[0].get("reward")
                    termination = sims[0].get("termination")
            rec = {
                "taskId": tid,
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "elapsed_s": round(elapsed, 1),
                "reward": reward,
                "termination": termination,
                "model": model,
                "stdout_tail": (proc.stdout or "")[-800:],
                "stderr_tail": (proc.stderr or "")[-800:],
            }
            print(
                f"[probe] done task={tid} rc={proc.returncode} reward={reward} "
                f"term={termination} elapsed={elapsed:.1f}s",
                flush=True,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - t0
            rec = {
                "taskId": tid,
                "ok": False,
                "returncode": None,
                "elapsed_s": round(elapsed, 1),
                "reward": None,
                "termination": "timeout",
                "model": None,
                "stdout_tail": (exc.stdout or "")[-400:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-400:] if isinstance(exc.stderr, str) else "",
            }
            print(f"[probe] SKIP task={tid} hung > {TIMEOUT}s", flush=True)
        results.append(rec)
        OUT.write_text(json.dumps({"domain": DOMAIN, "tasks": results, "timeout_s": TIMEOUT, "max_steps": MAX_STEPS}, indent=2) + "\n")

    fails = [r["taskId"] for r in results if r.get("reward") is None or r.get("reward", 1) < 1 - 1e-6]
    hits = [r["taskId"] for r in results if r.get("reward") is not None and r["reward"] >= 1 - 1e-6]
    print(json.dumps({"hits": hits, "fails": fails, "n": len(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
