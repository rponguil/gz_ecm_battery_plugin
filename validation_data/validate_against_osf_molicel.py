#!/usr/bin/env python3
"""Segunda validacion externa, dataset independiente: celda Molicel INR-21700-P42A
(NMC, 4.2Ah nominal, formato 21700), test HCGT ("High C-rate pulse with GITT"),
25 grados C, celda A13. Fuente: dataset abierto en OSF, DOI 10.17605/OSF.IO/9CEAV
("High-power lithium-ion battery characterization dataset for stochastic battery
modeling"), archivo osf_dataset/A13_HCGT_25degC.csv (exportado del .xlsx original,
formato Arbin).

Por que este dataset importa ademas del de Panasonic 18650PF: es una celda,
formato y capacidad DISTINTOS (21700/4.2Ah NMC vs 18650/2.9Ah NMC), de un
laboratorio distinto, con protocolo de pulsos distinto (GITT, no HPPC clasico) --
si el plugin generico (parametrizable por SDF, sin nada de celda hardcodeado en
el codigo) tambien reproduce esta celda con fidelidad razonable, es evidencia de
que el diseño "chemistry-agnostic" funciona de verdad, no solo para la celda
para la que se ajusto originalmente.

Metodologia (self-contained, una sola sesion de test, evita el problema de
desajuste de capacidad entre sesiones ya diagnosticado con el dataset de
Panasonic):
1. OCV(SOC): un test GITT ya incluye periodos de reposo entre pulsos --
   se usa el voltaje al FINAL de cada reposo suficientemente largo como
   punto de la curva OCV, indexado por el SOC acumulado en ese instante
   (coulomb counting propio, mismo metodo que usa el script cell_data.m
   del propio dataset).
2. Capacidad de referencia: el swing neto de Ah dentro de ESTE MISMO archivo
   (no se importa capacidad de otro archivo/sesion).
3. R0/R1/C1: ajustados de los pulsos reales del mismo archivo (reusa
   find_pulses/fit_r0_r1_c1_per_pulse de validate_against_panasonic18650pf.py,
   no se reimplementa la logica de ajuste una segunda vez).
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

# Hysteresis experiment: split rest points by approach direction (post-charge vs.
# post-discharge), measure the OCV gap at equal SOC, and feed half of it back as
# the instantaneous hysteresis amplitude M0.
#
# This is a REPORTED NEGATIVE RESULT, kept off by default. With the rest-point
# coverage this dataset provides (61 points, split 24/36 by direction) the two
# curves are too sparse to separate reliably, and enabling it degrades RMSE from
# 33.4 mV to 84.0 mV. The code is kept so the negative result is reproducible.
# Set to True to reproduce it; leave False for the baseline reported in the paper.
HYSTERESIS_EXPERIMENT = False
sys.path.insert(0, HERE)
from validate_against_panasonic18650pf import (  # noqa: E402
    build_soc_curve, fit_r0_r1_c1_per_pulse, fit_two_rc_branches_per_pulse,
    find_pulses)

from esc_battery_model import ESCModel, ESCParams  # noqa: E402


def load_osf_hcgt(path):
    df = pd.read_csv(path)
    t = df["Test_Time(s)"].to_numpy(dtype=float)
    v = df["Voltage(V)"].to_numpy(dtype=float)
    # Este dataset: corriente POSITIVA = carga (confirmado contra
    # Charge_Capacity creciente) -- opuesto a nuestra convencion
    # (positivo = descarga). Se invierte el signo al cargar, asi el resto
    # del pipeline es identico al del dataset de Panasonic.
    i = -df["Current(A)"].to_numpy(dtype=float)
    return {"t": t, "v": v, "i": i}


def coulomb_count_ah(t, i_discharge_positive):
    """Ah neto descargado acumulado (positivo=descarga), integracion
    trapezoidal -- mismo metodo que cell_data.m/coulomb_counting del
    propio dataset."""
    return np.concatenate([[0.0], np.cumsum(
        0.5 * (i_discharge_positive[1:] + i_discharge_positive[:-1])
        * np.diff(t))]) / 3600.0


def extract_ocv_from_rests(t, v, i, ah, min_rest_s=90.0, i_thresh=0.05):
    """Voltaje al final de cada periodo de reposo suficientemente largo,
    indexado por el SOC (via Ah acumulado) en ese instante, y el signo de
    la corriente que precedio al reposo (+1=veniamos de descarga, -1=de
    carga, 0=no se pudo determinar) -- la tecnica estandar de un test GITT
    para extraer OCV(SOC) de una sola sesion, extendida para poder separar
    "OCV aparente" segun la direccion desde la que se llego, que es
    exactamente como se mide histeresis real (ver literatura citada en el
    chat: la histeresis es un fenomeno de dependencia de trayectoria,
    persiste incluso tras reposos largos)."""
    resting = np.abs(i) < i_thresh
    edges = np.diff(resting.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if resting[0]:
        starts = np.insert(starts, 0, 0)
    if resting[-1]:
        ends = np.append(ends, len(resting))

    soc_pts, v_pts, sign_pts = [], [], []
    for s, e in zip(starts, ends):
        if t[e - 1] - t[s] < min_rest_s:
            continue
        # Signo de la corriente inmediatamente antes de este reposo
        # (ventana corta hacia atras, ignorando otros reposos previos).
        pre = i[max(0, s - 50):s]
        pre_active = pre[np.abs(pre) > i_thresh]
        sign = float(np.sign(pre_active[-1])) if len(pre_active) else 0.0
        soc_pts.append(ah[e - 1])
        v_pts.append(v[e - 1])
        sign_pts.append(sign)
    return np.array(soc_pts), np.array(v_pts), np.array(sign_pts)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_osf_hcgt(os.path.join(HERE, "osf_dataset", "A13_HCGT_25degC.csv"))
    t, v, i = data["t"], data["v"], data["i"]
    print(f"Muestras: {len(t)}, duracion: {t[-1]/3600:.1f} h, "
          f"V=[{v.min():.3f},{v.max():.3f}], I=[{i.min():.2f},{i.max():.2f}] A")

    ah_discharged = coulomb_count_ah(t, i)  # positivo = Ah descargados acumulados
    capacity_ah = float(ah_discharged.max() - ah_discharged.min())
    print(f"Capacidad (swing neto de Ah dentro de este archivo): "
          f"{capacity_ah:.3f} Ah")
    # find_pulses/fit_r0_r1_c1_per_pulse (reusadas de validate_against_
    # panasonic18650pf.py) esperan la convencion original de ese dataset:
    # ah NEGATIVO durante descarga (soc = 1 + ah/capacidad).
    data["ah"] = -(ah_discharged - ah_discharged.min())

    soc_ah_pts, v_ocv_pts, sign_pts = extract_ocv_from_rests(t, v, i, ah_discharged)
    print(f"Puntos OCV extraidos de periodos de reposo: {len(soc_ah_pts)}")
    # SOC fraccional: 1.0 en el Ah maximo alcanzado (mas cargado visto),
    # decreciendo con Ah descargado acumulado.
    soc_frac = 1.0 - (soc_ah_pts - ah_discharged.min()) / capacity_ah
    order = np.argsort(soc_frac)
    soc_frac, v_ocv_pts, sign_pts = soc_frac[order], v_ocv_pts[order], sign_pts[order]
    # Promedio por bins de SOC (igual tecnica que build_soc_curve) -- curva
    # "cruda" (mezcla reposos post-descarga y post-carga, como antes).
    ocv_curve = build_soc_curve(soc_frac, v_ocv_pts, bin_width=0.05, min_points=2)
    if ocv_curve is None:
        print("No hay suficientes puntos de reposo para construir OCV(SOC). Abortando.")
        return
    soc_grid, v_grid = ocv_curve
    print(f"Curva OCV (cruda, sin separar histeresis): {len(soc_grid)} bins, "
          f"rango SOC [{soc_grid.min():.2f},{soc_grid.max():.2f}]")

    # --- Histeresis: separar reposos que vienen de descarga vs. de carga,
    # comparar el voltaje en el MISMO SOC segun la direccion de llegada.
    # Esto es exactamente como se mide histeresis real (fenomeno de
    # dependencia de trayectoria, ver literatura citada en el chat).
    dis_mask, chg_mask = sign_pts > 0, sign_pts < 0
    print(f"Reposos post-descarga: {dis_mask.sum()}, post-carga: {chg_mask.sum()}")
    hyst_m0_fit = 0.0
    ocv_avg_grid, ocv_avg_v = soc_grid, v_grid
    if HYSTERESIS_EXPERIMENT and dis_mask.sum() >= 3 and chg_mask.sum() >= 3:
        dis_curve = build_soc_curve(soc_frac[dis_mask], v_ocv_pts[dis_mask],
                                     bin_width=0.1, min_points=2)
        chg_curve = build_soc_curve(soc_frac[chg_mask], v_ocv_pts[chg_mask],
                                     bin_width=0.1, min_points=2)
        if dis_curve is not None and chg_curve is not None:
            soc_dis, v_dis = dis_curve
            soc_chg, v_chg = chg_curve
            # gap en los bins de SOC que existen en ambas curvas
            common = sorted(set(np.round(soc_dis, 2)) & set(np.round(soc_chg, 2)))
            gaps = []
            for s in common:
                vd = v_dis[np.argmin(np.abs(soc_dis - s))]
                vc = v_chg[np.argmin(np.abs(soc_chg - s))]
                gaps.append(vc - vd)  # cargando deberia leer MAS alto que descargando
            if gaps:
                gap_mean = float(np.mean(gaps))
                hyst_m0_fit = gap_mean / 2.0
                print(f"Gap carga-descarga en {len(gaps)} bins de SOC comunes: "
                      f"media={gap_mean*1000:.1f} mV -> hyst_m0 ajustado = "
                      f"{hyst_m0_fit*1000:.1f} mV")
                # OCV "verdadera" = promedio de ambas direcciones (cancela
                # histeresis), mas preciso que usar solo una direccion.
                v_grid_dis_interp = np.interp(soc_grid, soc_dis, v_dis)
                v_grid_chg_interp = np.interp(soc_grid, soc_chg, v_chg)
                ocv_avg_v = (v_grid_dis_interp + v_grid_chg_interp) / 2.0
                ocv_avg_grid = soc_grid
            else:
                print("Sin bins de SOC comunes entre carga/descarga -- "
                      "no se pudo estimar histeresis, sigue con curva cruda.")
        else:
            print("No hay suficientes puntos para curvas separadas -- "
                  "sigue con curva cruda (sin separar histeresis).")
    else:
        print("Muy pocos reposos de una de las dos direcciones -- "
              "no se pudo estimar histeresis, sigue con curva cruda.")

    soc_reference = 1.0 - (ah_discharged - ah_discharged.min()) / capacity_ah
    soc_reference = np.clip(soc_reference, 0.0, 1.0)

    pulses = find_pulses(data, i_threshold=1.0, min_len=3)
    print(f"Pulsos detectados: {len(pulses)}")
    soc_r0, r0_vals, soc_r1, r1_vals, c1 = fit_r0_r1_c1_per_pulse(
        data, pulses, capacity_ah, rest_after_s=60.0)
    print(f"R0: {len(r0_vals)} pulsos, rango [{r0_vals.min():.4f},{r0_vals.max():.4f}] Ohm")
    print(f"R1: {len(r1_vals)} pulsos, rango [{r1_vals.min():.4f},{r1_vals.max():.4f}] Ohm")
    r0_median, r1_median = float(np.median(r0_vals)), float(np.median(r1_vals))

    # Curva OCV promediada entre carga/descarga (cancela histeresis) si se
    # pudo estimar; si no, la curva cruda de antes (comportamiento previo).
    ocv_func = interp1d(ocv_avg_grid, ocv_avg_v, kind="linear",
                         bounds_error=False,
                         fill_value=(ocv_avg_v[0], ocv_avg_v[-1]))

    params = ESCParams(
        capacity_ah=capacity_ah, r0_ohm=r0_median, r1_ohm=r1_median,
        c1_farad=c1, hyst_m=0.0, hyst_m0=hyst_m0_fit, coulombic_eff=1.0,
        ocv_func=ocv_func)

    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 0.1

    model = ESCModel(params, z0=1.0)
    v_sim = np.zeros_like(v)
    for k in range(len(t)):
        model.z = float(soc_reference[k])
        v_sim[k] = model.step(float(i[k]), float(dt_arr[k]))

    error_mv = (v_sim - v) * 1000.0
    rmse_mv = float(np.sqrt(np.mean(error_mv**2)))
    max_err_mv = float(np.max(np.abs(error_mv)))
    print(f"\nRMSE vs. celda real medida (Molicel P42A, 1 rama RC): {rmse_mv:.2f} mV")
    print(f"Error maximo (1 rama RC): {max_err_mv:.2f} mV")

    # --- Mismo experimento de 2 ramas RC que en el dataset de Panasonic ---
    fit2 = fit_two_rc_branches_per_pulse(data, pulses, rest_after_s=45.0)
    n2 = len(fit2["r0"])
    print(f"\n--- 2 ramas RC: {n2} pulsos con relajacion bi-exponencial resuelta ---")
    if n2 >= 5:
        r0_2, r1_2, r2_2 = (float(np.median(fit2["r0"])),
                             float(np.median(fit2["r1"])),
                             float(np.median(fit2["r2"])))
        c1_2 = float(np.median(fit2["tau1"])) / r1_2
        c2_2 = float(np.median(fit2["tau2"])) / r2_2
        print(f"R0={r0_2:.4f} Ohm, R1={r1_2:.4f} Ohm (tau1_mediana="
              f"{np.median(fit2['tau1']):.1f}s), C1={c1_2:.1f} F, "
              f"R2={r2_2:.4f} Ohm (tau2_mediana={np.median(fit2['tau2']):.1f}s), "
              f"C2={c2_2:.1f} F")
        params_2rc = ESCParams(
            capacity_ah=capacity_ah, r0_ohm=r0_2, r1_ohm=r1_2, c1_farad=c1_2,
            r2_ohm=r2_2, c2_farad=c2_2, hyst_m=0.0, hyst_m0=hyst_m0_fit,
            coulombic_eff=1.0, ocv_func=ocv_func)
        model_2rc = ESCModel(params_2rc, z0=1.0)
        v_sim_2rc = np.zeros_like(v)
        for k in range(len(t)):
            model_2rc.z = float(soc_reference[k])
            v_sim_2rc[k] = model_2rc.step(float(i[k]), float(dt_arr[k]))
        error_2rc_mv = (v_sim_2rc - v) * 1000.0
        rmse_2rc_mv = float(np.sqrt(np.mean(error_2rc_mv**2)))
        max_err_2rc_mv = float(np.max(np.abs(error_2rc_mv)))
        print(f"RMSE vs. celda real medida (2 ramas RC): {rmse_2rc_mv:.2f} mV "
              f"(1 rama: {rmse_mv:.2f} mV)")
        print(f"Error maximo (2 ramas RC): {max_err_2rc_mv:.2f} mV "
              f"(1 rama: {max_err_mv:.2f} mV)")

        fig2b, ax2b = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax2b[0].plot(t / 3600, v, "k", lw=1.0, label="Measured (real cell)")
        ax2b[0].plot(t / 3600, v_sim, "steelblue", lw=0.6, ls="--",
                      label=f"1 RC branch (RMSE={rmse_mv:.1f} mV)")
        ax2b[0].plot(t / 3600, v_sim_2rc, "darkorange", lw=0.6, ls=":",
                      label=f"2 RC branches (RMSE={rmse_2rc_mv:.1f} mV)")
        ax2b[0].set_ylabel("Voltage [V]")
        ax2b[0].legend(fontsize=9)
        ax2b[0].set_title("Molicel INR-21700-P42A: one vs. two RC branches against the measured cell")
        ax2b[0].grid(alpha=0.3)
        ax2b[1].plot(t / 3600, error_mv, "steelblue", lw=0.4, label="1 RC branch")
        ax2b[1].plot(t / 3600, error_2rc_mv, "darkorange", lw=0.4, label="2 RC branches")
        ax2b[1].axhline(0, color="k", lw=0.5)
        ax2b[1].set_ylabel("Error [mV]")
        ax2b[1].set_xlabel("Time [h]")
        ax2b[1].legend(fontsize=9)
        ax2b[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "molicel_1rc_vs_2rc.png"),
                    dpi=130, bbox_inches="tight")
    else:
        print("Muy pocos pulsos con relajacion bi-exponencial resuelta "
              "(<5) -- no se genera comparacion 2-RC para este dataset.")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t / 3600, i, "C1", lw=0.5)
    axes[0].set_ylabel("Current [A]")
    axes[0].set_title(
        "ESC model vs. measured Molicel INR-21700-P42A cell "
        "(OSF dataset, doi:10.17605/OSF.IO/9CEAV, cell A13, HCGT test, 25 degC)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t / 3600, v, "k", lw=0.8, label="Measured (real cell)")
    axes[1].plot(t / 3600, v_sim, "steelblue", lw=0.6, ls="--", label="ESC model")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[2].plot(t / 3600, error_mv, "purple", lw=0.4)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Error [mV]")
    axes[2].set_xlabel("Time [h]")
    axes[2].set_title(f"RMSE = {rmse_mv:.1f} mV | max. error = {max_err_mv:.1f} mV")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "validation_vs_osf_molicel.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")

    csv_path = os.path.join(RESULTS_DIR, "validation_vs_osf_molicel.csv")
    np.savetxt(csv_path,
               np.column_stack([t, i, v, v_sim, error_mv, soc_reference]),
               header="t_s,current_a,v_measured,v_simulated,error_mv,soc_reference",
               delimiter=",", comments="")
    print(f"\nCSV: {csv_path}\nFigura: {fig_path}")


if __name__ == "__main__":
    main()
