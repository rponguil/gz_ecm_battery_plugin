#!/usr/bin/env python3
"""Third independent cell: LG 18650HG2 (NMC/graphite, 3.0 Ah), 25 degC HPPC.

Dataset: P. Kollmeyer, M. Naguib et al., "LG 18650HG2 Li-ion Battery Data and
Example Deep Neural Network xEV SOC Estimator Script", Mendeley Data,
doi:10.17632/cp3473x7xv.3. Tested at McMaster University on a Digatron Firing
Circuits tester. Not redistributed here -- download the 25 degC HPPC file from
the DOI and place it as lg_hg2_25degC_HPPC.csv in this directory.

Same procedure as the other two cells, no per-cell tuning:
  * OCV(SOC) from the rest periods of this file (same-session method)
  * R0, R1, C1 from this file's own pulses
  * SOC anchored to this file's measured Ah

The point of a third cell is generalization. Two cells can agree by luck; three
chemistries from two laboratories converging on the same accuracy is evidence
that the model, and the parameterization recipe, transfer.

Run:  python3 validate_against_lg_hg2.py
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from validate_against_panasonic18650pf import (  # noqa: E402
    find_pulses, fit_r0_r1_c1_per_pulse, build_soc_curve)
from validate_against_osf_molicel import extract_ocv_from_rests  # noqa: E402
from esc_battery_model import ESCModel, ESCParams  # noqa: E402

DATA_FILE = "lg_hg2_25degC_HPPC.csv"


def load_digatron(path):
    """Digatron CSV reader for this dataset.

    Positional columns (the file has two irregular header blocks):
    3 = elapsed time HH:MM:SS.mmm, 8 = V, 9 = A, 10 = degC, 11 = Ah.
    """
    df = pd.read_csv(path, skiprows=30, header=None,
                     usecols=[3, 8, 9, 10, 11],
                     names=["tstr", "v", "i", "temp", "ah"],
                     on_bad_lines="skip").dropna()

    def to_sec(s):
        h, m, rest = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    t = np.array([to_sec(s) for s in df["tstr"]], dtype=float)
    return {"t": t, "v": df["v"].to_numpy(float),
            "i": df["i"].to_numpy(float), "ah": df["ah"].to_numpy(float)}


def main():
    path = os.path.join(HERE, DATA_FILE)
    if not os.path.exists(path):
        print(f"Missing {DATA_FILE}. Download it from "
              "doi:10.17632/cp3473x7xv.3 (25 degC HPPC file).")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    d = load_digatron(path)
    t, v = d["t"], d["v"]
    i = -d["i"]                            # project convention: + = discharge
    ah_disch = np.abs(d["ah"] - d["ah"][0])
    capacity_ah = float(ah_disch.max())
    soc_ref = np.clip(1.0 - ah_disch / capacity_ah, 0.0, 1.0)

    print(f"Samples {len(t)}, duration {t[-1]/3600:.2f} h, "
          f"V [{v.min():.3f}, {v.max():.3f}] V, capacity {capacity_ah:.3f} Ah")

    soc_pts, v_pts, _sg = extract_ocv_from_rests(t, v, i, ah_disch,
                                                  min_rest_s=90.0)
    soc_frac = 1.0 - (soc_pts - ah_disch.min()) / capacity_ah
    o = np.argsort(soc_frac)
    g, vg = build_soc_curve(soc_frac[o], v_pts[o], bin_width=0.05,
                            min_points=2)
    ocv = interp1d(g, vg, kind="linear", bounds_error=False,
                   fill_value=(vg[0], vg[-1]))
    print(f"OCV curve from {len(soc_pts)} rest periods, {len(g)} bins, "
          f"floor {vg.min():.3f} V (measured minimum {v.min():.3f} V)")

    dd = {"t": t, "v": v, "i": -i, "ah": -ah_disch}
    _s0, r0v, _s1, r1v, c1 = fit_r0_r1_c1_per_pulse(
        dd, find_pulses(dd), capacity_ah)
    r0, r1 = float(np.median(r0v)), float(np.median(r1v))
    print(f"Fitted from {len(r0v)} pulses: R0={r0*1000:.1f} mOhm, "
          f"R1={r1*1000:.1f} mOhm, C1={c1:.0f} F")

    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1
    model = ESCModel(ESCParams(capacity_ah=capacity_ah, r0_ohm=r0, r1_ohm=r1,
                               c1_farad=c1, hyst_m=0.0, hyst_m0=0.0,
                               coulombic_eff=1.0, ocv_func=ocv))
    v_sim = np.empty_like(v)
    for k in range(len(t)):
        model.z = float(soc_ref[k])
        v_sim[k] = model.step(float(i[k]), float(dt[k]))

    err = (v_sim - v) * 1000.0
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"\nRMSE vs. measured cell: {rmse:.2f} mV | "
          f"max error {np.max(np.abs(err)):.1f} mV")
    print("Reference: 34.34 mV (Panasonic 18650PF), 33.42 mV (Molicel P42A), "
          "both same-session parameterized.")

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t / 3600, i, "C1", lw=0.5)
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("ESC model vs. measured LG 18650HG2 cell "
                    "(Mendeley doi:10.17632/cp3473x7xv.3, HPPC, 25 degC)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t / 3600, v, "k", lw=0.8, label="Measured (real cell)")
    ax[1].plot(t / 3600, v_sim, "steelblue", lw=0.6, ls="--", label="ESC model")
    ax[1].set_ylabel("Voltage [V]")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)
    ax[2].plot(t / 3600, err, "purple", lw=0.4)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("Error [mV]")
    ax[2].set_xlabel("Time [h]")
    ax[2].set_title(f"RMSE = {rmse:.1f} mV | "
                    f"max. error = {np.max(np.abs(err)):.1f} mV")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "validation_vs_lg_hg2.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
