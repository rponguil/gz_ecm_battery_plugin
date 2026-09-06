#!/usr/bin/env python3
"""Baseline comparison: Gazebo Sim's LinearBatteryPlugin against this model.

The stock plugin models the cell as

    V = e0 + e1 * (1 - q/c) - r*i

with three constant coefficients. This script fits those three coefficients to
each measured trace by least squares and reports the resulting RMSE, so that
the comparison in the paper's validation table is reproducible rather than
asserted.

Two protocols are reported, because they answer different questions:

  in-sample  e0, e1, r are fitted on the whole trace and evaluated on the whole
             trace. This is deliberately generous to the baseline: it minimizes
             the reported metric directly, while the ECM parameters come from
             pulse-relaxation sub-experiments that never see it. It is the
             harder number for this work to beat.

  held-out   e0, e1, r are fitted once on the first half of the trace and
             evaluated on the second half. This is closer to how a simulator is
             actually used -- coefficients are chosen once, then the vehicle
             flies a profile nobody fitted to.

The state of charge used in the (1 - q/c) term is the dataset's own reference
where the file provides one (Panasonic), and coulomb-counted from the measured
current otherwise.

Run:  python3 validate_vs_linear_baseline.py
Requires: the result CSVs produced by the three per-cell validators, plus the
LG raw file. Run those first if they are missing; this script says which.
"""

import csv
import os
import sys

import numpy as np
from paper_numbers import record

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)


def rmse_mv(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * 1000.0)


def coulomb_soc(t, i):
    """SOC from the measured current, normalized to the charge actually drawn."""
    q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600.0
    return 1.0 - q / (q[-1] if q[-1] > 0 else 1.0)


def fit_linear(soc, current, voltage, fit_slice=None):
    """Least-squares fit of V = e0 + e1*(1 - q/c) - r*i.

    Returns (e0, e1, r) and the design matrix, so the caller can evaluate the
    fitted model on a different slice than the one it was fitted on.
    """
    design = np.column_stack([np.ones_like(voltage), soc, -current])
    sel = slice(None) if fit_slice is None else fit_slice
    coef, *_ = np.linalg.lstsq(design[sel], voltage[sel], rcond=None)
    return coef, design


def evaluate(name, t, current, voltage, soc, ecm_rmse=None, key=None):
    coef, design = fit_linear(soc, current, voltage)
    in_sample = rmse_mv(design @ coef, voltage)

    half = len(voltage) // 2
    coef_h, _ = fit_linear(soc, current, voltage, fit_slice=slice(0, half))
    held_out = rmse_mv(design[half:] @ coef_h, voltage[half:])

    print(f"\n{name}  ({len(voltage)} samples)")
    print(f"  baseline coefficients (full trace): "
          f"e0={coef[0]:.3f} V, e1={coef[1]:.3f} V, r={coef[2] * 1000:.1f} mOhm")
    print(f"  baseline, in-sample : {in_sample:6.1f} mV")
    print(f"  baseline, held-out  : {held_out:6.1f} mV")
    if ecm_rmse is not None:
        print(f"  this work (ECM)     : {ecm_rmse:6.1f} mV"
              f"   -> {in_sample / ecm_rmse:.2f}x in-sample")
    if key:
        record(f"{key}.baseline_in_sample_mv", in_sample, "mV",
               "Eq. (1) fitted to the whole trace by least squares")
        record(f"{key}.baseline_held_out_mv", held_out, "mV",
               "Eq. (1) fitted on the first half, evaluated on the second")
    return in_sample, held_out


def load_results_csv(filename, soc_column=None):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t_s"]) for r in rows])
    i = np.array([float(r["current_a"]) for r in rows])
    v = np.array([float(r["v_measured"]) for r in rows])
    sim = np.array([float(r["v_simulated"]) for r in rows])
    soc = (np.array([float(r[soc_column]) for r in rows])
           if soc_column and soc_column in rows[0] else coulomb_soc(t, i))
    return t, i, v, soc, sim


def main():
    print("Linear baseline (Gazebo LinearBatteryPlugin form) vs. this model")
    print("=" * 66)
    missing = []

    data = load_results_csv("validation_vs_panasonic18650pf.csv", "soc_reference")
    if data:
        t, i, v, soc, _ = data
        # The ECM figure quoted for this cell is the same-session-OCV run of
        # validate_ocv_provenance.py, not the cross-session run stored here.
        evaluate("Panasonic 18650PF (NCA, 18650)", t, i, v, soc, ecm_rmse=34.34, key="panasonic")
    else:
        missing.append("validate_against_panasonic18650pf.py")

    try:
        from validate_against_lg_hg2 import load_digatron
        lg_path = os.path.join(HERE, "lg_hg2_25degC_HPPC.csv")
        if os.path.exists(lg_path):
            d = load_digatron(lg_path)
            # This loader reports discharge as negative; the other two cells
            # use discharge-positive. Normalize so the fitted r is physical.
            lg_i = -np.asarray(d["i"])
            evaluate("LG 18650HG2 (NMC, 18650)", d["t"], lg_i, d["v"],
                     coulomb_soc(d["t"], lg_i), ecm_rmse=34.04, key="lg_hg2")
        else:
            missing.append("lg_hg2_25degC_HPPC.csv (see validate_against_lg_hg2.py)")
    except ImportError:
        missing.append("validate_against_lg_hg2.py")

    data = load_results_csv("validation_vs_osf_molicel.csv")
    if data:
        t, i, v, soc, _ = data
        evaluate("Molicel INR-21700-P42A (NMC, 21700)", t, i, v, soc,
                 ecm_rmse=33.42, key="molicel")
    else:
        missing.append("validate_against_osf_molicel.py")

    if missing:
        print("\nSkipped -- run these first:")
        for m in missing:
            print(f"  - {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
