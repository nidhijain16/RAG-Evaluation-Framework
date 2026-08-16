# ------------------------------------------------------------
# latency_helper.py
# Bridge between the FastAPI server and the AgentEval-Latency-Suite submodule.
# ------------------------------------------------------------
import os
import sys
import subprocess
import pathlib
import json
import datetime
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at: task/latency_helper.py
# The submodule lives at: task/latency_suite/
_THIS_DIR    = pathlib.Path(__file__).parent.resolve()
LATENCY_ROOT = _THIS_DIR / "latency_suite"

# Cache file is stored alongside this script in the task/ directory
CACHE_FILE = _THIS_DIR / "latency_results.json"


def _run_benchmark() -> str:
    """Executes `python -m benchmarks.latency_test` inside the latency_suite
    submodule using the *current* Python interpreter (the active venv).

    Returns the raw stdout string from the benchmark process.
    """
    # Use sys.executable so we always inherit the active virtual environment.
    python_exe = sys.executable

    # Pass NVIDIA_API_KEY through to the subprocess so it can read it.
    env = os.environ.copy()

    result = subprocess.run(
        [python_exe, "-m", "benchmarks.latency_test"],
        cwd=str(LATENCY_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Latency benchmark failed (exit code {result.returncode}).\n"
            f"Stderr:\n{result.stderr}"
        )

    return result.stdout


def parse_benchmark_output(raw: str) -> Dict[str, Any]:
    """Parse the benchmark stdout into a structured dictionary.

    Expected table format printed by LatencyBenchmarker.print_summary():

        ======================================================================
        User ID    | TTFT (ms)   | Total (ms)  | TPS      | Status
        ----------------------------------------------------------------------
        0          |     202.22 |    1428.53 |    16.31 | mock_success
        ...
        ----------------------------------------------------------------------
        Mean       |     202.22 |            |    16.31 |
        P50 (Medi… |     202.22 | (Tail Latency Analysis)
        P95        |     220.10 | (Production Threshold)
        P99        |     240.00 | (Extreme Case)
        ======================================================================

    Returns a dict with ``per_user`` (list) and ``summary`` (dict).
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Locate the header row
    try:
        header_idx = next(
            i for i, l in enumerate(lines) if l.startswith("User ID")
        )
    except StopIteration:
        # No header found — return raw text for debugging
        return {"per_user": [], "summary": {}, "raw": raw}

    data_rows = lines[header_idx + 2:]

    per_user: list = []
    summary: Dict[str, Any] = {}

    for line in data_rows:
        # Stop at separator lines
        if line.startswith("-") or line.startswith("="):
            continue

        parts = [p.strip() for p in line.split("|")]

        # Per-user data rows have exactly 5 pipe-separated columns
        if len(parts) >= 5 and parts[0].isdigit():
            try:
                per_user.append({
                    "user":     int(parts[0]),
                    "ttft_ms":  float(parts[1]),
                    "total_ms": float(parts[2]),
                    "tps":      float(parts[3]),
                    "status":   parts[4],
                })
            except (ValueError, IndexError):
                continue

        # Summary rows
        elif line.startswith("Mean") and len(parts) >= 4:
            try:
                summary["mean_ttft_ms"] = float(parts[1])
                summary["mean_tps"]     = float(parts[3])
            except (ValueError, IndexError):
                pass

        elif line.startswith("P50") and len(parts) >= 2:
            try:
                summary["p50_ttft_ms"] = float(parts[1])
            except (ValueError, IndexError):
                pass

        elif line.startswith("P95") and len(parts) >= 2:
            try:
                summary["p95_ttft_ms"] = float(parts[1])
            except (ValueError, IndexError):
                pass

        elif line.startswith("P99") and len(parts) >= 2:
            try:
                summary["p99_ttft_ms"] = float(parts[1])
            except (ValueError, IndexError):
                pass

    return {"per_user": per_user, "summary": summary}


def _store_results(data: Dict[str, Any]) -> None:
    """Write benchmark result dict to CACHE_FILE with a UTC timestamp."""
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    try:
        CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Latency Helper] Results saved to {CACHE_FILE}")
    except Exception as exc:
        print(f"[Latency Helper] Failed to write cache: {exc}")


def load_cached_results() -> Dict[str, Any]:
    """Load cached benchmark results.

    Returns the ``data`` portion of the stored JSON.
    Raises ``FileNotFoundError`` if no cache exists yet.
    """
    if CACHE_FILE.exists():
        try:
            content = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            return content.get("data", {})
        except Exception as exc:
            print(f"[Latency Helper] Failed to read cache: {exc}")
            raise
    raise FileNotFoundError(f"Cache file not found: {CACHE_FILE}")


def run_latency_benchmark() -> Dict[str, Any]:
    """Public API: run the benchmark, store the result, return the structured dict.

    Called by the FastAPI /latency endpoint and the UI Refresh button.
    """
    raw    = _run_benchmark()
    result = parse_benchmark_output(raw)
    _store_results(result)
    return result
