#!/usr/bin/env python3
"""Controlled experiment: how much does the PROVENANCE of the OCV curve matter?

Reproduces the two RMSE figures reported for the Panasonic 18650PF cell in the
paper (Table "Validation datasets and results"):

  * OCV taken from a SEPARATE test session (the C/20 slow discharge, recorded
    ~2 months before the HPPC file, when the cell still held 6.6% more charge)
  * OCV extracted from the rest periods WITHIN the HPPC file itself

Everything else is held fixed: same cell, same HPPC test, same fitted R0, R1
and C1, same model, same SOC reference. The only variable is where the OCV
curve comes from.

This is the evidence behind the practical guidance in the README and the paper:
extract the OCV curve from the same test session as the pulse data used to fit
the resistances, rather than from a separately recorded slow discharge.

Run:  python3 validate_ocv_provenance.py
Needs the two .mat files documented in validate_against_panasonic18650pf.py
(download from Mendeley doi:10.17632/wykht8y7tg.1).
"""

import os
import sys

import numpy as np
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from validate_against_panasonic18650pf import (  # noqa: E402
    load_meas, extract_ocv_curve, find_pulses, fit_r0_r1_c1_per_pulse,
    build_soc_curve)
from validate_against_osf_molicel import extract_ocv_from_rests  # noqa: E402
from esc_battery_model import ESCModel, ESCParams  # noqa: E402


def simulate(t, i, soc_ref, dt_arr, capacity_ah, r0, r1, c1, ocv_func):
    model = ESCModel(ESCParams(
        capacity_ah=capacity_ah, r0_ohm=r0, r1_ohm=r1, c1_farad=c1,
        hyst_m=0.0, hyst_m0=0.0, coulombic_eff=1.0, ocv_func=ocv_func))
    v_sim = np.empty(len(t))
    for k in range(len(t)):
        model.z = float(soc_ref[k])
        v_sim[k] = model.step(float(i[k]), float(dt_arr[k]))
    return v_sim


def main():
    c20 = load_meas(os.path.join(HERE, "c20_ocv_25degC.mat"))
    hppc = load_meas(os.path.join(HERE, "hppc_25degC.mat"))

    # Capacity from the HPPC file itself (the C/20 session measured 2.968 Ah,
    # this one 2.773 Ah -- the cell degraded between sessions).
    capacity_ah = float(abs(hppc["ah"].min()))

    t, v = hppc["t"], hppc["v"]
    i = -hppc["i"]                       # dataset: negative = discharge
    soc_ref = np.clip(1.0 + hppc["ah"] / capacity_ah, 0.0, 1.0)
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 0.1

    # Resistances fitted from the HPPC pulses -- identical for both variants.
    pulses = find_pulses(hppc)
    _s0, r0_vals, _s1, r1_vals, c1 = fit_r0_r1_c1_per_pulse(
        hppc, pulses, capacity_ah)
    r0, r1 = float(np.median(r0_vals)), float(np.median(r1_vals))
    print(f"Fitted from HPPC pulses (shared by both variants): "
          f"R0={r0*1000:.2f} mOhm, R1={r1*1000:.2f} mOhm, C1={c1:.0f} F")
    print(f"Capacity (HPPC file's own): {capacity_ah:.3f} Ah\n")

    # --- Variant A: OCV from the separate C/20 session ---
    g_a, v_a, cap_c20 = extract_ocv_curve(c20)
    ocv_a = interp1d(g_a, v_a, kind="cubic", bounds_error=False,
                     fill_value=(v_a[0], v_a[-1]))

    # --- Variant B: OCV from the HPPC file's own rest periods ---
    ah_disch = -hppc["ah"]               # increasing = Ah discharged
    soc_pts, v_pts, _sign = extract_ocv_from_rests(t, v, i, ah_disch,
                                                    min_rest_s=90.0)
    soc_frac = 1.0 - (soc_pts - ah_disch.min()) / capacity_ah
    order = np.argsort(soc_frac)
    curve_b = build_soc_curve(soc_frac[order], v_pts[order],
                              bin_width=0.05, min_points=2)
    if curve_b is None:
        print("Not enough rest periods in the HPPC file to build an OCV curve.")
        return
    g_b, v_b = curve_b
    ocv_b = interp1d(g_b, v_b, kind="linear", bounds_error=False,
                     fill_value=(v_b[0], v_b[-1]))

    variants = [
        (f"separate session (C/20, {cap_c20:.3f} Ah, {len(g_a)} points)", ocv_a),
        (f"same session (HPPC rest periods, {len(g_b)} bins)", ocv_b),
    ]

    results = []
    for label, ocv in variants:
        v_sim = simulate(t, i, soc_ref, dt_arr, capacity_ah, r0, r1, c1, ocv)
        err = (v_sim - v) * 1000.0
        rmse = float(np.sqrt(np.mean(err ** 2)))
        max_err = float(np.max(np.abs(err)))
        results.append((label, rmse, max_err))
        print(f"OCV from {label}")
        print(f"    RMSE = {rmse:7.2f} mV   |  max error = {max_err:7.1f} mV")

    ra, rb = results[0][1], results[1][1]
    print(f"\nChanging ONLY the OCV provenance: {ra:.2f} -> {rb:.2f} mV "
          f"({ra/rb:.2f}x)")
    print("Note the worst-case error does NOT improve -- the end-of-discharge "
          "residual has a different cause (see the paper's Limitations).")


if __name__ == "__main__":
    main()
