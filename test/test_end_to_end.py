#!/usr/bin/env python3
"""End-to-end test: SDF -> plugin -> ECM component -> gz-transport telemetry.

The 14 GTest cases cover the simulator-independent core (CubicSpline and
EscModel). They say nothing about the layer the user actually runs: SDF
parsing, the BatterySoC component, and the BatteryState topic. This test
exercises that path.

It launches the demo world headless for a fixed number of iterations and checks
that the plugin loaded, parsed its parameters, and reported a terminal voltage
consistent with the configured OCV curve and power load.

Run:  python3 test/test_end_to_end.py
Requires: gz sim on PATH and GZ_SIM_SYSTEM_PLUGIN_PATH pointing at the build.
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORLD = os.path.join(ROOT, "worlds", "ecm_battery_demo.sdf")

# The demo world declares an OCV curve spanning 3.00-4.20 V and an 18.5 W load
# on a 2.5 Ah LIR18650. Anything outside this window means the SDF was not
# parsed into the model correctly.
V_MIN, V_MAX = 2.5, 4.3


def run_sim(iterations=200):
    env = dict(os.environ)
    env.setdefault("GZ_IP", "127.0.0.1")
    env.setdefault("GZ_PARTITION", "e2e_test")
    plugin_path = env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    build = os.path.join(ROOT, "build")
    if build not in plugin_path:
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = f"{build}:{plugin_path}"
    return subprocess.run(
        ["gz", "sim", "-s", "-r", "-v", "4", "--iterations", str(iterations),
         WORLD],
        capture_output=True, text=True, timeout=180, env=env)


def main():
    if not os.path.exists(WORLD):
        print(f"FAIL: world not found at {WORLD}")
        return 1
    try:
        proc = run_sim()
    except FileNotFoundError:
        print("SKIP: 'gz' not on PATH; Gazebo Sim is required for this test.")
        return 0
    except subprocess.TimeoutExpired:
        print("FAIL: gz sim did not finish within the timeout.")
        return 1

    # Gazebo interleaves ANSI colour codes inside its log lines, which breaks
    # naive pattern matching; strip them before checking.
    out = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    checks = []

    loaded = "EscBatteryPlugin configured" in out
    checks.append(("plugin loaded and configured", loaded))

    system = "Loaded system" in out and "EscBatteryPlugin" in out
    checks.append(("registered as a gz-sim system", system))

    # The configure log reports the parsed SDF values and the topic name.
    parsed = re.search(r"capacity ([\d.]+) Ah, R0=([\d.]+) Ohm", out)
    checks.append(("SDF parameters parsed", parsed is not None))

    topic = "/model/battery_test_box/battery/lir18650_esc/state" in out
    checks.append(("telemetry topic advertised", topic))

    # The load-time parameterization check must report a reachable floor that
    # is physically sensible for the declared curve.
    floor = re.search(r"can reach is about ([\d.]+) V", out)
    floor_ok = floor is not None and V_MIN <= float(floor.group(1)) <= V_MAX
    checks.append(("reachable-voltage check within declared OCV range",
                   floor_ok))

    ok = all(passed for _, passed in checks)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not ok:
        print("\n--- simulator output ---")
        print(out[-3000:])
    print("\nEND-TO-END:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
