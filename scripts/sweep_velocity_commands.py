"""Sweep velocity commands through a trained walking policy and report tracking.

    uv run scripts/sweep_velocity_commands.py out.onnx --motor hd1910
    uv run scripts/sweep_velocity_commands.py old.onnx new.onnx --motor hd1910   # A/B

Runs `scripts/infer_policy.py` once per command point (CPU MuJoCo, no GPU — but it
does open a viewer window), parses its `[vel 1s avg]` lines, throws away the first
`--drop` samples as spawn transient, and averages the rest. `ratio = achieved /
commanded`, so 1.00 is perfect tracking.

WHY THIS EXISTS — trust this, not the training curves.
`Episode_Reward/track_*` and `Metrics/twist/error_vel_*` are averages over a mixed
command distribution, and they can move the OPPOSITE way from real per-command
tracking. Measured 2026-10-10: tightening the angular tracking std scored
`error_vel_yaw` 2x *worse* and mean reward 131 -> 98, while full-stick turning in
this sweep collapsed from 1.29x to 0.26x. The sweep said "it stopped turning";
the aggregate said "everything got worse". Only one of them is actionable.

`--motor` has no default on purpose: rehearsing an HD-1910 policy with the xl330
actuator model produces numbers that look fine and compare the wrong robot.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INFER = REPO_ROOT / "scripts" / "infer_policy.py"

# One line per command point, mirroring the training command ranges
# (vx +-0.4, vy +-0.3, wz +-1.0 — see microduck_velocity_env_cfg.py).
# The extra vx points bracket the small-command dead zone.
POINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zero", ()),
    ("vx +0.10", ("--lin-vel-x", "0.10")),
    ("vx +0.20", ("--lin-vel-x", "0.20")),
    ("vx +0.30", ("--lin-vel-x", "0.30")),
    ("vx +0.40", ("--lin-vel-x", "0.40")),
    ("vx +0.50", ("--lin-vel-x", "0.50")),
    ("vx -0.20", ("--lin-vel-x", "-0.20")),
    ("vy +0.20", ("--lin-vel-y", "0.20")),
    ("vy -0.20", ("--lin-vel-y", "-0.20")),
    ("wz +0.50", ("--ang-vel-z", "0.50")),
    ("wz -0.50", ("--ang-vel-z", "-0.50")),
    ("wz +1.00", ("--ang-vel-z", "1.00")),
    ("wz -1.00", ("--ang-vel-z", "-1.00")),
)

# `[vel 1s avg] achieved/cmd  fwd=+0.18/+0.20  lat=+0.00/+0.00 m/s  yaw=-0.03/+0.00 rad/s   trunk_z=117.9 mm`
LINE = re.compile(
    r"fwd=(?P<fwd>[+-][\d.]+)/(?P<fwd_cmd>[+-][\d.]+)\s+"
    r"lat=(?P<lat>[+-][\d.]+)/(?P<lat_cmd>[+-][\d.]+)\s*m/s\s+"
    r"yaw=(?P<yaw>[+-][\d.]+)/(?P<yaw_cmd>[+-][\d.]+)\s*rad/s\s+"
    r"trunk_z=(?P<tz>[\d.]+)\s*mm"
)

# trunk_z under this reads as fallen, not crouched (nominal is ~115-120 mm).
FALLEN_MM = 60.0
TZ_CMD = 0.01  # below this a command slot counts as un-commanded


def _launcher() -> list[str]:
    """Prefer `uv run` so this works under a bare `python3` too."""
    uv = shutil.which("uv")
    return [uv, "run", str(INFER)] if uv else [sys.executable, str(INFER)]


def run_point(
    onnx: Path,
    args: tuple[str, ...],
    motor: str,
    seconds: float,
    drop: int,
    extra: list[str],
) -> tuple[dict[str, float] | None, str]:
    """Run one command point; return (averaged samples or None, diagnostic tail).

    The policy runs until it is *killed*, exactly like `timeout <n> infer_policy.py`.
    `subprocess.run(timeout=...)` is wrong here: it re-raises and throws away the
    output, so every point comes back empty. Hence Popen + communicate.

    `start_new_session` + killpg mirrors GNU `timeout`: `uv run` forks the real
    python, so killing just uv leaves the grandchild holding the stdout pipe and
    the final communicate() blocks forever.
    """
    cmd = [
        *_launcher(),
        "--walking", str(onnx),
        "--new-cmd-obs",
        "--motor", motor,
        *extra,
        *args,
    ]
    # PYTHONUNBUFFERED: with stdout on a pipe Python block-buffers, so a killed
    # run can lose everything it printed.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=REPO_ROOT, env=env, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=seconds + 20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            return None, "killed but stdout never closed"

    rows = [m.groupdict() for m in LINE.finditer(out or "")][drop:]
    if not rows:
        tail = "\n".join((err or "").strip().splitlines()[-6:])
        return None, tail

    def avg(key: str) -> float:
        return statistics.fmean(float(r[key]) for r in rows)

    return {
        "fwd": avg("fwd"), "fwd_cmd": avg("fwd_cmd"),
        "lat": avg("lat"), "lat_cmd": avg("lat_cmd"),
        "yaw": avg("yaw"), "yaw_cmd": avg("yaw_cmd"),
        "tz": avg("tz"), "tz_min": min(float(r["tz"]) for r in rows),
        "n": len(rows),
    }


def ratio_of(r: dict[str, float]) -> tuple[float, float, float]:
    """(ratio, achieved, commanded) along whichever axis this point commands."""
    for achieved, commanded in (
        ("yaw", "yaw_cmd"), ("fwd", "fwd_cmd"), ("lat", "lat_cmd"),
    ):
        if abs(r[commanded]) > TZ_CMD:
            return r[achieved] / r[commanded], r[achieved], r[commanded]
    return float("nan"), 0.0, 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("onnx", nargs="+", type=Path, help="walking policy ONNX (1+ to compare)")
    parser.add_argument(
        "--motor", required=True, choices=("xl330", "hd1910"),
        help="actuator preset the policy was TRAINED against (no default on purpose)",
    )
    parser.add_argument(
        "--seconds", type=float, default=26.0,
        help="wall-clock seconds to let each point run before killing it (~2 sim-s per "
             "wall-s, so ~2 samples/s; 1 sample = 1 s of sim)",
    )
    parser.add_argument(
        "--drop", type=int, default=4,
        help="discard the first N samples per point as spawn transient (1 sample ~ 1 s)",
    )
    parser.add_argument(
        "--infer-arg", action="append", default=[], metavar="FLAG",
        help="extra flag passed through to infer_policy.py, repeatable",
    )
    parser.add_argument(
        "--points", default=None,
        help="comma-separated point labels to run a subset, e.g. 'wz +1.00,zero'",
    )
    args = parser.parse_args()

    missing = [p for p in args.onnx if not p.is_file()]
    if missing:
        parser.error("not a file: " + ", ".join(map(str, missing)))

    points = POINTS
    if args.points:
        wanted = {p.strip() for p in args.points.split(",")}
        points = tuple(p for p in POINTS if p[0] in wanted)
        unknown = wanted - {p[0] for p in points}
        if unknown:
            parser.error(f"unknown point(s): {', '.join(sorted(unknown))}")

    print(f"motor = {args.motor}   ({len(points)} points x {args.seconds:.0f}s"
          f" x {len(args.onnx)} model(s))", flush=True)
    for p in args.onnx:
        print(f"  {p}", flush=True)
    print(flush=True)

    results: dict[tuple[str, str], dict[str, float]] = {}
    for label, point_args in points:
        for onnx in args.onnx:
            r = run_point(onnx, point_args, args.motor, args.seconds, args.drop, args.infer_arg)
            state = "fell?" if r and r["tz_min"] < FALLEN_MM else "ok"
            if r is None:
                state = "NO DATA"
            elif state == "ok":
                state = f"ok ({r['n']})"
            print(f"  {label:<10} {onnx.name:<28} {state}", flush=True)
            if r is not None:
                results[(label, onnx.name)] = r
        print(flush=True)

    width = max(len(p.name) for p in args.onnx)
    print("=" * (30 + width + 34))
    print(f"{'command':<11} {'model':<{width}} {'ratio':>7} {'achieved':>9} {'cmd':>7} "
          f"{'trunk_z mm':>14} {'n':>4}")
    print("-" * (30 + width + 34))
    for label, _ in points:
        for onnx in args.onnx:
            r = results.get((label, onnx.name))
            if r is None:
                print(f"{label:<11} {onnx.name:<{width}} {'—':>7}")
                continue
            ratio, achieved, commanded = ratio_of(r)
            ratio_s = "—" if ratio != ratio else f"{ratio:>7.2f}"
            cmd_s = "—" if abs(commanded) <= TZ_CMD else f"{commanded:>+7.2f}"
            flag = " !" if r["tz_min"] < FALLEN_MM else ""
            print(f"{label:<11} {onnx.name:<{width}} {ratio_s} {achieved:>+9.3f} {cmd_s} "
                  f"{r['tz']:>6.1f}/{r['tz_min']:<6.1f} {r['n']:>4}{flag}")
    print("=" * (30 + width + 34))
    print("ratio = achieved / commanded; 1.00 is perfect tracking. "
          f"trunk_z nominal ~115-120 mm; under {FALLEN_MM:.0f} = fallen (!).")
    print("A 0.00 ratio at a non-zero command is a dead zone: the policy stood still.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
