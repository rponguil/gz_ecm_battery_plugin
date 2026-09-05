#!/usr/bin/env python3
"""Figure: the OCV-provenance experiment, both variants side by side.

Same cell, same HPPC test, same fitted R0/R1/C1, same model. The only thing
that differs between the two panels is where the OCV(SOC) curve comes from:
a separate C/20 session recorded two months earlier, versus the rest periods
inside the HPPC file itself.

This is the figure form of validate_ocv_provenance.py, and it uses exactly the
same configuration as the numbers reported in the paper's validation table:
constant R0 and R1 (the medians of the per-pulse fits), no hysteresis.

Run:  python3 make_provenance_figure.py
"""

import os
import sys

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from validate_against_panasonic18650pf import (  # noqa: E402
    load_meas, extract_ocv_curve, find_pulses, fit_r0_r1_c1_per_pulse,
    build_soc_curve)
from validate_against_osf_molicel import extract_ocv_from_rests  # noqa: E402
from esc_battery_model import ESCModel, ESCParams  # noqa: E402


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    c20 = load_meas(os.path.join(HERE, "c20_ocv_25degC.mat"))
    hppc = load_meas(os.path.join(HERE, "hppc_25degC.mat"))

    capacity_ah = float(abs(hppc["ah"].min()))
    t, v = hppc["t"], hppc["v"]
    i = -hppc["i"]
    soc_ref = np.clip(1.0 + hppc["ah"] / capacity_ah, 0.0, 1.0)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    # Resistances: fitted once from the HPPC pulses, shared by both variants.
    _s0, r0v, _s1, r1v, c1 = fit_r0_r1_c1_per_pulse(
        hppc, find_pulses(hppc), capacity_ah)
    r0, r1 = float(np.median(r0v)), float(np.median(r1v))

    # (a) OCV from the separate C/20 session
    ga, va, _ = extract_ocv_curve(c20)
    ocv_a = interp1d(ga, va, kind="cubic", bounds_error=False,
                     fill_value=(va[0], va[-1]))
    # (b) OCV from this file's own rest periods
    ah_d = -hppc["ah"]
    sp, vp, _sg = extract_ocv_from_rests(t, v, i, ah_d, min_rest_s=90.0)
    sf = 1.0 - (sp - ah_d.min()) / capacity_ah
    o = np.argsort(sf)
    gb, vb = build_soc_curve(sf[o], vp[o], bin_width=0.05, min_points=2)
    ocv_b = interp1d(gb, vb, kind="linear", bounds_error=False,
                     fill_value=(vb[0], vb[-1]))

    def simulate(ocv):
        m = ESCModel(ESCParams(capacity_ah=capacity_ah, r0_ohm=r0, r1_ohm=r1,
                               c1_farad=c1, hyst_m=0.0, hyst_m0=0.0,
                               coulombic_eff=1.0, ocv_func=ocv))
        out = np.empty_like(v)
        for k in range(len(t)):
            m.z = float(soc_ref[k])
            out[k] = m.step(float(i[k]), float(dt[k]))
        return out

    v_a, v_b = simulate(ocv_a), simulate(ocv_b)
    rms = lambda x: float(np.sqrt(np.mean((x - v) ** 2))) * 1000
    ra, rb = rms(v_a), rms(v_b)
    print(f"OCV from separate C/20 session : RMSE {ra:.2f} mV")
    print(f"OCV from same-session rests    : RMSE {rb:.2f} mV  ({ra/rb:.2f}x)")

    th = t / 3600.0
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey="row")
    for col, (vs, r, title) in enumerate([
            (v_a, ra, "(a) OCV from a separate test session (C/20, 2 months earlier)"),
            (v_b, rb, "(b) OCV from the rest periods of this same file")]):
        ax[0, col].plot(th, v, "k", lw=0.8, label="Measured")
        ax[0, col].plot(th, vs, "steelblue", lw=0.6, ls="--", label="ESC model")
        ax[0, col].set_title(f"{title}\nRMSE = {r:.1f} mV", fontsize=10)
        ax[0, col].grid(alpha=0.3)
        if col == 0:
            ax[0, col].set_ylabel("Voltage [V]")
            ax[0, col].legend(fontsize=9)
        ax[1, col].plot(th, (vs - v) * 1000, "purple", lw=0.4)
        ax[1, col].axhline(0, color="k", lw=0.5)
        ax[1, col].set_xlabel("Time [h]")
        ax[1, col].grid(alpha=0.3)
        if col == 0:
            ax[1, col].set_ylabel("Error [mV]")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ocv_provenance_two_panel.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
