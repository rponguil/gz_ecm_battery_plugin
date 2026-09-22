#!/usr/bin/env python3
"""Build and run benchmark_step_cost.cc, and record its figures.

Kept separate from the C++ so the manifest stays the single place a number
enters the manuscript.

Run:  python3 record_step_cost.py
"""
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_numbers import record  # noqa: E402

BIN = os.path.join(HERE, "benchmark_step_cost")
SRC = os.path.join(HERE, "benchmark_step_cost.cc")


def main():
    subprocess.run(["g++", "-O2", "-std=c++17", "-I../include", SRC, "-o", BIN],
                   cwd=HERE, check=True)
    out = subprocess.run([BIN], cwd=HERE, capture_output=True, text=True,
                         check=True).stdout
    print(out, end="")
    m = re.search(r"^JSON (\{.*\})$", out, re.M)
    if not m:
        sys.exit("benchmark produced no JSON line")
    d = json.loads(m.group(1))

    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    cpu = re.sub(r"\s+with\s+.*$", "", cpu)

    record("runtime.step_ns_1rc", d["step_ns_1rc"], "ns",
           f"EscModel::Step, one RC branch, scalar R0/R1, on {cpu}")
    record("runtime.step_ns_2rc", d["step_ns_2rc"], "ns",
           "EscModel::Step, two RC branches, scalar R0/R1")
    record("runtime.step_ns_curves", d["step_ns_curves"], "ns",
           "EscModel::Step, two RC branches, R0(z) and R1(z) splines")
    record("runtime.model_bytes", d["sizeof_model"], "bytes",
           "sizeof(EscModel); the OCV spline's samples are heap allocated")
    # What the number means for a user: fraction of a 1 kHz physics step.
    record("runtime.duty_1khz_pct", 100.0 * d["step_ns_curves"] * 1e-9 / 1e-3,
           "%", "worst configuration as a share of a 1 ms physics step")
    # The manifest stores numbers; the machine goes in the note of the
    # figure it qualifies, above.


if __name__ == "__main__":
    main()
