"""
Step 4.4 — regression tracking: each `python -m evals` run writes its
aggregate metrics to evals/results/<timestamp>.json (tagged with the current
git commit when available), and compares against the previous run so a
regression is visible immediately instead of silently drifting.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Metrics where lower is better — everything else in a results record is
# treated as "higher is better" when checking for regressions.
LOWER_IS_BETTER = {"review_rate"}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def write_results(metrics: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    record = {
        "timestamp": timestamp.isoformat(),
        "git_sha": _git_sha(),
        **metrics,
    }
    # Microsecond precision — plain second-level timestamps collide (and
    # silently overwrite each other) on fast successive runs.
    path = results_dir / f"{timestamp.strftime('%Y%m%dT%H%M%S%f')}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def latest_results(results_dir: Path = RESULTS_DIR, exclude: Path | None = None) -> dict | None:
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("*.json"))
    if exclude is not None:
        files = [f for f in files if f.resolve() != exclude.resolve()]
    if not files:
        return None
    return json.loads(files[-1].read_text())


def compare(current: dict, previous: dict | None, tolerance: float = 0.02) -> dict:
    """
    Returns {metric: {"current", "previous", "delta", "regressed"}} for every
    numeric metric both records share. `tolerance` is the minimum drop (or
    rise, for LOWER_IS_BETTER metrics) before it counts as a regression.
    """
    if previous is None:
        return {}
    comparison = {}
    for key, value in current.items():
        prior = previous.get(key)
        if not isinstance(value, (int, float)) or not isinstance(prior, (int, float)):
            continue
        delta = round(value - prior, 4)
        regressed = delta < -tolerance if key not in LOWER_IS_BETTER else delta > tolerance
        comparison[key] = {"current": value, "previous": prior, "delta": delta, "regressed": regressed}
    return comparison
