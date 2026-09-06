#!/usr/bin/env python3
"""Validation across temperature: -20, -10, 0, 10 and 25 degC.

Why this matters for users of the plugin: equivalent-circuit parameters are
strongly temperature dependent (a cell is far more resistive when cold), and a
robot that flies outdoors does not operate at a comfortable 25 degC. A model
validated only at room temperature says little about the conditions the
simulation is actually meant to cover.

Each temperature is treated as a self-contained parameterization exercise,
using the method the README recommends:

  * OCV(SOC) extracted from the rest periods inside that temperature's own
    HPPC file
  * R0, R1, C1 fitted from that same file's pulses
  * SOC anchored to that same file's measured Ah

Note this is not a stylistic choice. The dataset contains a dedicated C/20 OCV
campaign only at 25 degC, so at every other temperature the same-session
extraction is the only option available -- which is precisely the situation a
practitioner parameterizing a cell from a partial dataset will face.

Data: Kollmeyer, "Panasonic 18650PF Li-ion Battery Data", Mendeley Data V1,
doi:10.17632/wykht8y7tg.1. The .mat files are not redistributed here; fetch
them from the DOI (or the mirror noted in validate_against_panasonic18650pf.py)
and place them in this directory as hppc_<T>degC.mat.

Run:  python3 validate_across_temperature.py
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
    load_meas, find_pulses, fit_r0_r1_c1_per_pulse, build_soc_curve)
from validate_against_osf_molicel import extract_ocv_from_rests  # noqa: E402
from esc_battery_model import ESCModel, ESCParams  # noqa: E402
from paper_numbers import record

# (etiqueta, archivo, temperatura en degC)
CASES = [
    ("-20 degC", "hppc_n20degC.mat", -20),
    ("-10 degC", "hppc_n10degC.mat", -10),
    ("  0 degC", "hppc_0degC.mat", 0),
    (" 10 degC", "hppc_10degC.mat", 10),
    (" 25 degC", "hppc_25degC.mat", 25),
]


def run_one(path):
    """Parameterize and evaluate the ESC model on one temperature's HPPC file."""
    d = load_meas(path)
    capacity_ah = float(abs(d["ah"].min()))

    t, v = d["t"], d["v"]
    i = -d["i"]                       # dataset: negative = discharge
    soc_ref = np.clip(1.0 + d["ah"] / capacity_ah, 0.0, 1.0)

    # OCV from this file's own rest periods (same-session method).
    ah_disch = -d["ah"]
    soc_pts, v_pts, _sign = extract_ocv_from_rests(t, v, i, ah_disch,
                                                    min_rest_s=90.0)
    if len(soc_pts) < 4:
        return None
    soc_frac = 1.0 - (soc_pts - ah_disch.min()) / capacity_ah
    order = np.argsort(soc_frac)
    curve = build_soc_curve(soc_frac[order], v_pts[order],
                            bin_width=0.05, min_points=2)
    if curve is None:
        return None
    g, vg = curve
    ocv = interp1d(g, vg, kind="linear", bounds_error=False,
                   fill_value=(vg[0], vg[-1]))

    # R0/R1/C1 from this file's own pulses.
    pulses = find_pulses(d)
    fit = fit_r0_r1_c1_per_pulse(d, pulses, capacity_ah)
    _s0, r0_vals, _s1, r1_vals, c1 = fit
    if len(r0_vals) == 0 or len(r1_vals) == 0:
        return None
    r0, r1 = float(np.median(r0_vals)), float(np.median(r1_vals))

    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    model = ESCModel(ESCParams(
        capacity_ah=capacity_ah, r0_ohm=r0, r1_ohm=r1, c1_farad=c1,
        hyst_m=0.0, hyst_m0=0.0, coulombic_eff=1.0, ocv_func=ocv))
    v_sim = np.empty_like(v)
    for k in range(len(t)):
        model.z = float(soc_ref[k])
        v_sim[k] = model.step(float(i[k]), float(dt[k]))

    err = (v_sim - v) * 1000.0
    return {
        "rmse_mv": float(np.sqrt(np.mean(err ** 2))),
        "max_mv": float(np.max(np.abs(err))),
        "r0_mohm": r0 * 1000.0,
        "r1_mohm": r1 * 1000.0,
        "c1_f": c1,
        "capacity_ah": capacity_ah,
        "n_rest": len(soc_pts),
        "n_pulses": len(r0_vals),
        "ocv_floor": float(vg.min()),
        "v_floor": float(v.min()),
        "t": t, "v": v, "v_sim": v_sim, "ocv": ocv,
    }


def fit_only(path):
    """Return the parameters fitted from one temperature's file, nothing else."""
    r = run_one(path)
    if r is None:
        return None
    return {"r0": r["r0_mohm"] / 1000.0, "r1": r["r1_mohm"] / 1000.0,
            "c1": r["c1_f"], "capacity_ah": r["capacity_ah"], "ocv": r["ocv"]}


def run_transfer(src_path, dst_path):
    """Parameterize at one temperature, evaluate at another.

    The paper states that parameters do not transfer across temperature. This
    puts a number on it instead of asserting it: how much worse is a model
    carrying 25 C parameters when the cell is actually at -20 C?
    """
    src = fit_only(src_path)
    if src is None:
        return None
    d = load_meas(dst_path)
    capacity_ah = float(abs(d["ah"].min()))
    t, v = d["t"], d["v"]
    i = -d["i"]
    soc_ref = np.clip(1.0 + d["ah"] / capacity_ah, 0.0, 1.0)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    # Everything comes from the source temperature except the trace itself.
    model = ESCModel(ESCParams(
        capacity_ah=src["capacity_ah"], r0_ohm=src["r0"], r1_ohm=src["r1"],
        c1_farad=src["c1"], hyst_m=0.0, hyst_m0=0.0, coulombic_eff=1.0,
        ocv_func=src["ocv"]))
    v_sim = np.empty_like(v)
    for k in range(len(t)):
        model.z = float(soc_ref[k])
        v_sim[k] = model.step(float(i[k]), float(dt[k]))
    err = (v_sim - v) * 1000.0
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    print(f"{'T':>9} | {'RMSE':>8} | {'max err':>8} | {'R0':>7} | {'R1':>7} |"
          f" {'Cap':>6} | {'pulses':>6} | {'OCV floor vs Vmin':>18}")
    print("-" * 96)
    for label, fname, temp in CASES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"{label:>9} | (falta {fname} -- ver docstring)")
            continue
        r = run_one(path)
        if r is None:
            print(f"{label:>9} | (no se pudo parametrizar con este archivo)")
            continue
        rows.append((label, temp, r))
        print(f"{label:>9} | {r['rmse_mv']:6.2f}mV | {r['max_mv']:6.1f}mV |"
              f" {r['r0_mohm']:5.1f}mO | {r['r1_mohm']:5.1f}mO |"
              f" {r['capacity_ah']:5.3f}Ah | {r['n_pulses']:6d} |"
              f" {r['ocv_floor']:.3f}V vs {r['v_floor']:.3f}V")

    if not rows:
        print("\nNo hay archivos para evaluar.")
        return

    rm = [r["rmse_mv"] for _, _, r in rows]
    by_t = {t: r for _, t, r in rows}
    record("temperature.rmse_min_mv", min(rm), "mV", "best over the sweep")
    record("temperature.rmse_max_mv", max(rm), "mV", "worst over the sweep")
    if 25 in by_t and -20 in by_t:
        record("temperature.r0_25c_mohm", by_t[25]["r0_mohm"], "mOhm", "fitted at 25 C")
        record("temperature.r0_n20c_mohm", by_t[-20]["r0_mohm"], "mOhm", "fitted at -20 C")
        record("temperature.capacity_25c_ah", by_t[25]["capacity_ah"], "Ah", "usable at 25 C")
        record("temperature.capacity_n20c_ah", by_t[-20]["capacity_ah"], "Ah", "usable at -20 C")
        record("temperature.capacity_drop_pct",
               100.0 * (1.0 - by_t[-20]["capacity_ah"] / by_t[25]["capacity_ah"]),
               "%", "usable capacity lost between 25 and -20 C")

    # Cuanto cuesta NO re-parametrizar: 25 degC aplicado a -20 degC.
    src, dst = os.path.join(HERE, "hppc_25degC.mat"), os.path.join(HERE, "hppc_n20degC.mat")
    if os.path.exists(src) and os.path.exists(dst) and -20 in by_t:
        xfer = run_transfer(src, dst)
        if xfer:
            record("temperature.transfer_25c_to_n20c_mv", xfer, "mV",
                   "25 C parameters evaluated on -20 C data")
            record("temperature.transfer_penalty_x", xfer / by_t[-20]["rmse_mv"], "x",
                   "penalty for not re-parameterizing at the operating temperature")
            print(f"\n25 degC parameters on -20 degC data: {xfer:.1f} mV "
                  f"vs {by_t[-20]['rmse_mv']:.1f} mV re-parameterized "
                  f"({xfer / by_t[-20]['rmse_mv']:.1f}x worse)")

    print("\nR0 crece al bajar la temperatura, como debe: la celda es mas")
    print("resistiva en frio. El modelo se re-parametriza por temperatura;")
    print("no se asume que los parametros de 25 degC valgan a -20 degC.")

    # --- figura resumen ---
    temps = [t for _, t, _ in rows]
    rmses = [r["rmse_mv"] for _, _, r in rows]
    r0s = [r["r0_mohm"] for _, _, r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(temps, rmses, "o-", color="steelblue")
    ax[0].set_xlabel("Temperature [°C]")
    ax[0].set_ylabel("RMSE vs. measured cell [mV]")
    ax[0].set_title("ESC model accuracy across temperature")
    ax[0].grid(alpha=0.3)
    ax[1].plot(temps, r0s, "s-", color="crimson")
    ax[1].set_xlabel("Temperature [°C]")
    ax[1].set_ylabel("Fitted $R_0$ [m$\\Omega$]")
    ax[1].set_title("Series resistance rises as the cell gets cold")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "validation_across_temperature.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nFigura: {out}")


if __name__ == "__main__":
    main()
