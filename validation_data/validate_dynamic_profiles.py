#!/usr/bin/env python3
"""Score the ECM on dynamic load profiles it was never fitted to.

The validation reported so far parameterises and scores the model on
characterisation data: HPPC or GITT pulse trains. That answers "does the model
reproduce the cell" but not "does a parameterisation obtained once transfer to
a load the fit never saw". This script answers the second question.

Protocol
--------
Everything the model knows comes from the 25 degC HPPC session: OCV(SOC) from
its rest periods, R0 and R1 from its pulses, C1 from the relaxation time
constants. Nothing is refitted afterwards. The model is then run open loop on
the drive-cycle files of the same dataset -- US06, UDDS, LA92, HWFET and the
mixed Cycle_1/Cycle_2 profiles -- with state of charge obtained by coulomb
counting from a full cell, exactly as the plugin runs inside a simulator. No
SOC anchoring to the file's Ah record is used here: unlike the HPPC file, these
files log the current continuously, so the integration is well posed.

These are automotive profiles, not flight profiles; what they provide is a
dynamic, transient-rich discharge (peaks near 7C with regeneration) recorded on
the same cell, under a protocol disjoint from the one used to fit.

Two capacity choices are reported, because the deployment mismatch is the point:

  characterisation  capacity taken from the HPPC session -- what a user has
                    after characterising the cell once;
  oracle            capacity taken from the profile being scored -- not
                    available in practice, reported to separate the error due
                    to capacity drift from the error due to the model.

Run:  python3 validate_dynamic_profiles.py
"""

import glob
import os
import sys

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CYCLE_DIR = os.path.join(HERE, "drive_cycles_25degC")
sys.path.insert(0, HERE)

from esc_battery_model import ESCModel, ESCParams  # noqa: E402
from paper_numbers import record  # noqa: E402
from validate_against_panasonic18650pf import (  # noqa: E402
    build_soc_curve,
    find_pulses,
    fit_r0_r1_c1_per_pulse,
    load_meas,
)
from validate_against_osf_molicel import extract_ocv_from_rests  # noqa: E402

# Pretty names, in the order they are reported.
PROFILES = [
    ("US06", "US06"),
    ("UDDS", "UDDS"),
    ("LA92", "LA92"),
    ("HWFTa", "HWFET"),
    ("Cycle_1", "Mixed 1"),
    ("Cycle_2", "Mixed 2"),
]


def characterise():
    """Everything the model is allowed to know, from the HPPC session alone."""
    hppc = load_meas(os.path.join(HERE, "hppc_25degC.mat"))
    capacity_ah = float(abs(hppc["ah"].min()))

    t, v = hppc["t"], hppc["v"]
    i = -hppc["i"]  # dataset: negative = discharge; model: positive = discharge

    pulses = find_pulses(hppc)
    _s0, r0_vals, _s1, r1_vals, c1 = fit_r0_r1_c1_per_pulse(
        hppc, pulses, capacity_ah)
    r0, r1 = float(np.median(r0_vals)), float(np.median(r1_vals))

    ah_disch = -hppc["ah"]
    soc_pts, v_pts, _sign = extract_ocv_from_rests(
        t, v, i, ah_disch, min_rest_s=90.0)
    soc_frac = 1.0 - (soc_pts - ah_disch.min()) / capacity_ah
    order = np.argsort(soc_frac)
    curve = build_soc_curve(soc_frac[order], v_pts[order],
                            bin_width=0.05, min_points=2)
    if curve is None:
        sys.exit("Not enough rest periods in the HPPC file to build an OCV curve.")
    g, vg = curve
    ocv = interp1d(g, vg, kind="linear", bounds_error=False,
                   fill_value=(vg[0], vg[-1]))
    return dict(capacity_ah=capacity_ah, r0=r0, r1=r1, c1=c1, ocv=ocv,
                n_ocv_bins=len(g), hppc=hppc)


def run_open_loop(cycle, capacity_ah, r0, r1, c1, ocv):
    """Coulomb counting from a full cell -- no anchoring to the Ah record."""
    t = cycle["t"]
    i = -cycle["i"]
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    model = ESCModel(ESCParams(
        capacity_ah=capacity_ah, r0_ohm=r0, r1_ohm=r1, c1_farad=c1,
        hyst_m=0.0, hyst_m0=0.0, coulombic_eff=1.0, ocv_func=ocv), z0=1.0)
    v_sim = np.empty(len(t))
    for k in range(len(t)):
        v_sim[k] = model.step(float(i[k]), float(dt[k]))
    return v_sim


def linear_baseline(cycle, e0, e1, r, capacity_ah):
    """Eq. (1) of the manuscript, run over the same profile."""
    i = -cycle["i"]
    t = cycle["t"]
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1
    q = capacity_ah * 3600.0
    charge = np.clip(q - np.cumsum(i * dt), 0.0, q)
    return e0 + e1 * (1.0 - charge / q) - r * i


def fit_linear_on_hppc(hppc, capacity_ah):
    """Least squares for e0, e1, r on the characterisation trace."""
    i = -hppc["i"]
    soc = np.clip(1.0 + hppc["ah"] / capacity_ah, 0.0, 1.0)
    A = np.column_stack([np.ones_like(i), 1.0 - soc, -i])
    coef, *_ = np.linalg.lstsq(A, hppc["v"], rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def rmse_mv(v_sim, v_meas):
    e = (v_sim - v_meas) * 1000.0
    return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.isdir(CYCLE_DIR):
        sys.exit(
            f"Missing {CYCLE_DIR}.\n"
            "The drive-cycle files are part of the Panasonic 18650PF dataset\n"
            "(doi:10.17632/wykht8y7tg.1) and are not redistributed here. Fetch\n"
            "the six files named in PROFILES from the '25degC/Drive cycles'\n"
            "folder of the dataset, or of its GitHub mirror\n"
            "https://github.com/fededc88/Panasonic-18650PF-Data, into that\n"
            "directory.")

    p = characterise()
    e0, e1, r_lin = fit_linear_on_hppc(p["hppc"], p["capacity_ah"])
    print(f"Characterisation (25 degC HPPC session only):")
    print(f"  capacity {p['capacity_ah']:.3f} Ah | R0 {p['r0']*1000:.2f} mOhm | "
          f"R1 {p['r1']*1000:.2f} mOhm | C1 {p['c1']:.0f} F | "
          f"OCV from {p['n_ocv_bins']} rest bins")
    print(f"  linear baseline fitted on the same trace: e0={e0:.3f} V, "
          f"e1={e1:.3f} V, r={r_lin*1000:.1f} mOhm\n")

    rows = []
    for tag, nice in PROFILES:
        hits = glob.glob(os.path.join(CYCLE_DIR, f"*{tag}*.mat"))
        if not hits:
            print(f"  (skipping {nice}: file not present)")
            continue
        cyc = load_meas(hits[0])
        cap_oracle = float(abs(cyc["ah"].min()))

        v_char = run_open_loop(cyc, p["capacity_ah"], p["r0"], p["r1"],
                               p["c1"], p["ocv"])
        v_orac = run_open_loop(cyc, cap_oracle, p["r0"], p["r1"],
                               p["c1"], p["ocv"])
        v_lin = linear_baseline(cyc, e0, e1, r_lin, p["capacity_ah"])

        rc, mc = rmse_mv(v_char, cyc["v"])
        ro, mo = rmse_mv(v_orac, cyc["v"])
        rl, ml = rmse_mv(v_lin, cyc["v"])
        peak_c = float(np.max(np.abs(cyc["i"]))) / cap_oracle
        bias = float(np.mean((v_orac - cyc["v"]) * 1000.0))

        # Aggregate RMSE over a long profile is dominated by a static offset,
        # and a fitted series resistance can absorb an offset without
        # representing anything. To separate "compensated" from "followed",
        # the error is also scored with its own mean removed, and again over
        # the samples where the load actually moves -- the decile of largest
        # |di/dt|, which is where the RC branch and the OCV curvature are the
        # only things that can help.
        e_ecm = (v_orac - cyc["v"]) * 1000.0
        e_lin = (v_lin - cyc["v"]) * 1000.0
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        di = np.abs(np.diff(cyc["i"], prepend=cyc["i"][0]))
        fast = di >= np.quantile(di, 0.90)
        ecm_nb, lin_nb = rms(e_ecm - e_ecm.mean()), rms(e_lin - e_lin.mean())
        ecm_tr = rms(e_ecm[fast] - e_ecm[fast].mean())
        lin_tr = rms(e_lin[fast] - e_lin[fast].mean())
        # Diagnostic: the one scalar that would absorb the bias. Reported to
        # locate the error, never applied -- refitting per profile would be
        # exactly the leakage this experiment exists to avoid.
        r0_best, rmse_best = p["r0"], ro
        for r0_try in np.arange(p["r0"], p["r0"] + 0.045, 0.0025):
            rr, _ = rmse_mv(run_open_loop(cyc, cap_oracle, float(r0_try),
                                          p["r1"], p["c1"], p["ocv"]), cyc["v"])
            if rr < rmse_best:
                r0_best, rmse_best = float(r0_try), rr
        rows.append(dict(tag=tag, nice=nice, cyc=cyc, cap=cap_oracle,
                         hours=float(cyc["t"][-1] / 3600.0), peak_c=peak_c,
                         rmse_char=rc, max_char=mc, rmse_orac=ro,
                         max_orac=mo, rmse_lin=rl, max_lin=ml, bias=bias,
                         r0_best=r0_best, rmse_best=rmse_best,
                         ecm_nb=ecm_nb, lin_nb=lin_nb,
                         ecm_tr=ecm_tr, lin_tr=lin_tr,
                         v_char=v_char, v_orac=v_orac))
        print(f"  {nice:<8} {rows[-1]['hours']:4.2f} h  peak {peak_c:4.1f}C  |  "
              f"ECM {rc:6.1f} mV (oracle capacity {ro:6.1f})  |  "
              f"linear {rl:7.1f} mV  |  ratio {rl/rc:4.2f}x  |  "
              f"bias {bias:+5.1f} mV, R0 opt {r0_best*1000:4.1f} mOhm\n"
              f"           transients: ECM {ecm_tr:5.1f} vs linear "
              f"{lin_tr:5.1f} mV  ({lin_tr/ecm_tr:4.2f}x)")

    if not rows:
        sys.exit("No drive-cycle files found.")

    ecm = np.array([r["rmse_char"] for r in rows])
    orac = np.array([r["rmse_orac"] for r in rows])
    lin = np.array([r["rmse_lin"] for r in rows])
    print(f"\n  ECM  {ecm.min():.1f}-{ecm.max():.1f} mV "
          f"(oracle capacity {orac.min():.1f}-{orac.max():.1f})")
    print(f"  Eq.(1) {lin.min():.1f}-{lin.max():.1f} mV | "
          f"advantage {(lin/ecm).min():.2f}-{(lin/ecm).max():.2f}x")

    record("dynamic.n_profiles", len(rows), "profiles",
           "25 degC drive cycles scored with HPPC-only parameters")
    record("dynamic.ecm_rmse_min_mv", float(ecm.min()), "mV",
           "ECM, open loop, capacity from the characterisation session")
    record("dynamic.ecm_rmse_max_mv", float(ecm.max()), "mV",
           "ECM, open loop, capacity from the characterisation session")
    record("dynamic.ecm_oracle_rmse_min_mv", float(orac.min()), "mV",
           "ECM, open loop, capacity taken from the scored profile")
    record("dynamic.ecm_oracle_rmse_max_mv", float(orac.max()), "mV",
           "ECM, open loop, capacity taken from the scored profile")
    record("dynamic.linear_rmse_min_mv", float(lin.min()), "mV",
           "Eq. (1) fitted on the same HPPC session, same profiles")
    record("dynamic.linear_rmse_max_mv", float(lin.max()), "mV",
           "Eq. (1) fitted on the same HPPC session, same profiles")
    record("dynamic.advantage_min_x", float((lin / ecm).min()), "x",
           "Eq. (1) RMSE divided by ECM RMSE, per profile")
    record("dynamic.advantage_max_x", float((lin / ecm).max()), "x",
           "Eq. (1) RMSE divided by ECM RMSE, per profile")
    bias_a = np.array([r["bias"] for r in rows])
    r0b = np.array([r["r0_best"] for r in rows])
    record("dynamic.bias_min_mv", float(bias_a.min()), "mV",
           "mean signed error, oracle capacity; positive = model reads high")
    record("dynamic.bias_max_mv", float(bias_a.max()), "mV",
           "mean signed error, oracle capacity; positive = model reads high")
    record("dynamic.r0_hppc_mohm", float(p["r0"] * 1000), "mOhm",
           "series resistance identified from HPPC pulse onsets")
    record("dynamic.r0_refit_min_mohm", float(r0b.min() * 1000), "mOhm",
           "single R0 minimising RMSE per profile (diagnostic, not applied)")
    record("dynamic.r0_refit_max_mohm", float(r0b.max() * 1000), "mOhm",
           "single R0 minimising RMSE per profile (diagnostic, not applied)")
    record("dynamic.advantage_oracle_min_x", float((lin / orac).min()), "x",
           "Eq. (1) over ECM with oracle capacity, per profile")
    record("dynamic.advantage_oracle_max_x", float((lin / orac).max()), "x",
           "Eq. (1) over ECM with oracle capacity, per profile")
    tr_adv = np.array([r["lin_tr"] / r["ecm_tr"] for r in rows])
    nb_adv = np.array([r["lin_nb"] / r["ecm_nb"] for r in rows])
    record("dynamic.transient_advantage_min_x", float(tr_adv.min()), "x",
           "linear over ECM on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_advantage_max_x", float(tr_adv.max()), "x",
           "linear over ECM on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_ecm_min_mv",
           float(min(r["ecm_tr"] for r in rows)), "mV",
           "ECM on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_ecm_max_mv",
           float(max(r["ecm_tr"] for r in rows)), "mV",
           "ECM on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_linear_min_mv",
           float(min(r["lin_tr"] for r in rows)), "mV",
           "Eq. (1) on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_linear_max_mv",
           float(max(r["lin_tr"] for r in rows)), "mV",
           "Eq. (1) on the top decile of |di/dt|, mean removed")
    record("dynamic.transient_profiles_won", int((tr_adv > 1.0).sum()),
           "profiles", "profiles where the ECM tracks transients better")
    record("dynamic.debiased_advantage_min_x", float(nb_adv.min()), "x",
           "linear over ECM with each model's own mean error removed")
    record("dynamic.debiased_advantage_max_x", float(nb_adv.max()), "x",
           "linear over ECM with each model's own mean error removed")
    record("dynamic.peak_c_max", float(max(r["peak_c"] for r in rows)), "C",
           "largest current in the scored profiles, in C-rate")
    record("dynamic.hours_total", float(sum(r["hours"] for r in rows)), "h",
           "total scored duration")

    csv = os.path.join(RESULTS_DIR, "validation_dynamic_profiles.csv")
    with open(csv, "w") as fh:
        fh.write("profile,hours,peak_C,capacity_Ah,ecm_rmse_mV,ecm_max_mV,"
                 "ecm_oracle_rmse_mV,linear_rmse_mV,linear_max_mV,"
                 "bias_mV,r0_refit_mOhm,rmse_refit_mV,"
                 "ecm_transient_mV,linear_transient_mV\n")
        for r in rows:
            fh.write(f"{r['nice']},{r['hours']:.3f},{r['peak_c']:.2f},"
                     f"{r['cap']:.4f},{r['rmse_char']:.2f},{r['max_char']:.1f},"
                     f"{r['rmse_orac']:.2f},{r['rmse_lin']:.2f},"
                     f"{r['max_lin']:.1f},{r['bias']:.1f},"
                     f"{r['r0_best']*1000:.1f},{r['rmse_best']:.2f},"
                     f"{r['ecm_tr']:.2f},{r['lin_tr']:.2f}\n")
    print(f"\n-> {os.path.relpath(csv, HERE)}")

    # LaTeX table fragment, so the manuscript never retypes these rows.
    tex = os.path.join(RESULTS_DIR, "table_dynamic.tex")
    with open(tex, "w") as fh:
        fh.write("% GENERATED by validation_data/validate_dynamic_profiles.py\n"
                 "% Do not edit: rerun the script.\n")
        for r in rows:
            fh.write(f"{r['nice']} & {r['hours']:.2f} & {r['peak_c']:.1f} & "
                     f"{r['rmse_orac']:.1f} & {r['rmse_lin']:.1f} & "
                     f"{r['bias']:+.0f} & "
                     f"\\textbf{{{r['ecm_tr']:.1f}}} & {r['lin_tr']:.1f} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"Range & {sum(r['hours'] for r in rows):.1f} & "
                 f"{max(r['peak_c'] for r in rows):.1f} & "
                 f"{orac.min():.1f}--{orac.max():.1f} & "
                 f"{lin.min():.1f}--{lin.max():.1f} & "
                 f"{bias_a.min():+.0f} to {bias_a.max():+.0f} & "
                 f"\\textbf{{{min(r['ecm_tr'] for r in rows):.1f}--"
                 f"{max(r['ecm_tr'] for r in rows):.1f}}} & "
                 f"{min(r['lin_tr'] for r in rows):.1f}--"
                 f"{max(r['lin_tr'] for r in rows):.1f} \\\\\n")
        # The closing rule lives inside the fragment: a \\ immediately before
        # end of file leaves a \bottomrule that follows the \input misplaced.
        fh.write("\\bottomrule\n")
    print(f"-> {os.path.relpath(tex, HERE)}")

    # Figure: the most aggressive profile, measured against simulated.
    worst = max(rows, key=lambda r: r["peak_c"])
    cyc = worst["cyc"]
    t_h = cyc["t"] / 3600.0
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [1.0, 2.0, 1.2]})
    ax[0].plot(t_h, -cyc["i"], lw=0.4, color="0.35")
    ax[0].set_ylabel("Current (A)")
    ax[0].set_title(f"{worst['nice']} profile, peak {worst['peak_c']:.1f}C — "
                    f"parameters from the HPPC session only")
    ax[1].plot(t_h, cyc["v"], lw=0.8, label="measured")
    ax[1].plot(t_h, worst["v_orac"], lw=0.8, ls="--", label="ECM, open loop")
    ax[1].set_ylabel("Terminal voltage (V)")
    ax[1].legend(loc="lower left", frameon=False)
    ax[2].plot(t_h, (worst["v_orac"] - cyc["v"]) * 1000.0, lw=0.4, color="tab:red")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("Error (mV)")
    ax[2].set_xlabel("Time (h)")
    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    png = os.path.join(RESULTS_DIR, "validation_dynamic_profiles.png")
    fig.savefig(png, dpi=200)
    print(f"-> {os.path.relpath(png, HERE)}")


if __name__ == "__main__":
    main()
