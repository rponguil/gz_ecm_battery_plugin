#!/usr/bin/env python3
"""Validación externa del modelo ESC contra datos REALES publicados.

Dataset: Kollmeyer, Phillip (2018), "Panasonic 18650PF Li-ion Battery Data",
Mendeley Data, V1, doi: 10.17632/wykht8y7tg.1 -- celda 18650PF real, 2.9Ah,
tests HPPC (0.5/1/2/4/6C) y descarga C/20 a 25 grados C. Descargado de
https://github.com/fededc88/Panasonic-18650PF-Data (mismo dataset, mirror).

Esto NO es una comparacion circular contra nuestro propio modelo Python de
referencia (esc_battery_model.py) -- es contra voltaje MEDIDO de una celda
real por un tercero (Univ. Wisconsin-Madison / McMaster). Metodologia:

1. OCV(SOC): extraida de la descarga lenta C/20 (suficientemente lenta para
   que la caida IR y la dinamica transitoria sean despreciables, V ~= OCV).
2. R0, R1, C1: ajustados de los pulsos HPPC reales (salto instantaneo de
   voltaje al inicio del pulso -> R0; relajacion exponencial tras el pulso
   -> R1, C1), no supuestos ni copiados de la literatura.
3. Corremos el modelo ESC (la misma clase ESCModel de
   dron_px4_battery_models/esc_battery_model.py, sin reescribirla) con estos
   parametros REALES contra el perfil de corriente REAL del archivo HPPC, y
   comparamos voltaje simulado vs. voltaje medido -- RMSE real.

Requiere: numpy, scipy, matplotlib (pip3 install numpy scipy matplotlib).
"""

import os
import sys

import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Reutilizar la clase ESCModel real (misma fuente de verdad que el plugin
# C++ porta) en vez de reimplementar la matematica una tercera vez.
ESC_MODEL_PATH = os.path.join(
    HERE, "..", "..", "..", "Dron_PX4_ROS2", "ros2_ws", "src",
    "dron_px4_battery_models", "dron_px4_battery_models")
sys.path.insert(0, os.path.abspath(ESC_MODEL_PATH))
from esc_battery_model import ESCModel, ESCParams  # noqa: E402


def load_meas(path):
    d = loadmat(path)
    m = d["meas"][0, 0]
    return {
        "t": m["Time"].flatten().astype(float),
        "v": m["Voltage"].flatten().astype(float),
        "i": m["Current"].flatten().astype(float),  # dataset: negativo = descarga
        "ah": m["Ah"].flatten().astype(float),
    }


def extract_ocv_curve(c20, n_points=41):
    """OCV(SOC) desde la descarga C/20 -- SOC = 1 + Ah/capacidad_medida."""
    capacity_measured = abs(c20["ah"].min())
    soc = 1.0 + c20["ah"] / capacity_measured
    order = np.argsort(soc)
    soc_sorted, v_sorted = soc[order], c20["v"][order]

    soc_grid = np.linspace(0.0, 1.0, n_points)
    v_grid = np.interp(soc_grid, soc_sorted, v_sorted)
    return soc_grid, v_grid, capacity_measured


def find_pulses(hppc, i_threshold=0.5, min_len=5):
    """Detecta segmentos contiguos de corriente no nula (pulsos)."""
    active = np.abs(hppc["i"]) > i_threshold
    edges = np.diff(active.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if active[0]:
        starts = np.insert(starts, 0, 0)
    if active[-1]:
        ends = np.append(ends, len(active))
    pulses = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_len]
    return pulses


def fit_r0_r1_c1_per_pulse(hppc, pulses, capacity_ah, rest_after_s=25.0):
    """Ajusta R0 (salto instantaneo) y R1/C1 (relajacion exponencial tras
    el pulso) para CADA pulso individualmente, junto con el SOC en el
    momento del pulso -- para poder construir R0(SOC)/R1(SOC) despues, en
    vez de una sola mediana global que ignora que la celda es mas resistiva
    en los extremos de SOC."""
    r0_list, r1_list, tau_list, soc_list_r0, soc_list_r1 = [], [], [], [], []
    t, v, i, ah = hppc["t"], hppc["v"], hppc["i"], hppc["ah"]

    for (s, e) in pulses:
        if s < 2 or e + 5 >= len(t):
            continue
        i_pulse = i[s:e].mean()
        if abs(i_pulse) < 1.0:
            continue  # ignorar pulsos muy chicos, ruido domina el ajuste

        # R0: salto de voltaje en la primera muestra del pulso vs. la
        # muestra de reposo inmediatamente anterior.
        v_rest_before = v[s - 1]
        v_pulse_start = v[s]
        r0 = abs((v_rest_before - v_pulse_start) / i_pulse)
        if not (0.005 < r0 < 0.5):
            continue  # descarta ajustes no fisicos (ruido / borde de dataset)
        soc_at_pulse = 1.0 + ah[s] / capacity_ah
        r0_list.append(r0)
        soc_list_r0.append(soc_at_pulse)

        # R1, C1: relajacion exponencial de v hacia v_final tras el pulso.
        dt = np.median(np.diff(t[s:e + 200])) if e + 200 < len(t) else 0.1
        n_rest = int(rest_after_s / max(dt, 1e-3))
        n_rest = min(n_rest, len(t) - e - 1)
        if n_rest < 10:
            continue
        t_rel = t[e:e + n_rest] - t[e]
        v_rel = v[e:e + n_rest]
        v_inf = v_rel[-1]

        def relax(tt, dv0, tau):
            return v_inf + dv0 * np.exp(-tt / tau)

        try:
            dv0_guess = v_rel[0] - v_inf
            popt, _ = curve_fit(
                relax, t_rel, v_rel,
                p0=[dv0_guess, 20.0],
                bounds=([-2.0, 0.5], [2.0, 500.0]),
                maxfev=2000)
            dv0, tau = popt
            r1 = abs(dv0 / i_pulse)
            if 0.001 < r1 < 0.5 and 1.0 < tau < 300.0:
                r1_list.append(r1)
                tau_list.append(tau)
                soc_list_r1.append(soc_at_pulse)
        except RuntimeError:
            continue

    c1 = float(np.median(tau_list)) / float(np.median(r1_list))
    return (np.array(soc_list_r0), np.array(r0_list),
            np.array(soc_list_r1), np.array(r1_list), c1)


def fit_two_rc_branches_per_pulse(hppc, pulses, rest_after_s=90.0,
                                   tau1_bounds=(15.0, 300.0),
                                   tau2_bounds=(0.5, 15.0)):
    """Como fit_r0_r1_c1_per_pulse, pero ajusta la relajacion post-pulso con
    DOS exponenciales (dos ramas RC) en vez de una: v(t) = v_inf + dv1*exp(-t/tau1)
    + dv2*exp(-t/tau2), con tau1 (rama lenta, difusion) >> tau2 (rama rapida,
    transferencia de carga) forzado por los bounds -- evita la ambiguedad de
    "cual exponencial es cual rama" en el ajuste. Motivado por que ambos
    datasets reales (Panasonic 18650PF, Molicel 21700-P42A) muestran el mismo
    patron de error grande solo en SOC bajo/corriente alta con 1 sola rama RC.
    Requiere una ventana de reposo mas larga que el ajuste de 1 rama (para
    poder resolver las DOS constantes de tiempo, no solo una) -- pulsos donde
    no hay suficiente reposo despues simplemente no contribuyen (mismo patron
    defensivo de "skip si no converge" que ya usa el ajuste de 1 rama).

    Devuelve arrays por pulso: r0, r1, tau1, r2, tau2 (sin binning por SOC en
    esta primera pasada -- se usa la mediana global de cada uno)."""
    r0_list, r1_list, tau1_list, r2_list, tau2_list = [], [], [], [], []
    t, v, i = hppc["t"], hppc["v"], hppc["i"]

    for (s, e) in pulses:
        if s < 2 or e + 5 >= len(t):
            continue
        i_pulse = i[s:e].mean()
        if abs(i_pulse) < 1.0:
            continue

        r0 = abs((v[s - 1] - v[s]) / i_pulse)
        if not (0.005 < r0 < 0.5):
            continue

        dt = np.median(np.diff(t[s:e + 200])) if e + 200 < len(t) else 0.1
        n_rest = int(rest_after_s / max(dt, 1e-3))
        n_rest = min(n_rest, len(t) - e - 1)
        if n_rest < 30:
            continue
        t_rel = t[e:e + n_rest] - t[e]
        v_rel = v[e:e + n_rest]
        v_inf = v_rel[-1]
        dv0_guess = v_rel[0] - v_inf

        def relax2(tt, dv1, tau1, dv2, tau2):
            return v_inf + dv1 * np.exp(-tt / tau1) + dv2 * np.exp(-tt / tau2)

        try:
            popt, _ = curve_fit(
                relax2, t_rel, v_rel,
                p0=[dv0_guess * 0.6, 30.0, dv0_guess * 0.4, 3.0],
                bounds=([-2.0, tau1_bounds[0], -2.0, tau2_bounds[0]],
                        [2.0, tau1_bounds[1], 2.0, tau2_bounds[1]]),
                maxfev=4000)
            dv1, tau1, dv2, tau2 = popt
            r1 = abs(dv1 / i_pulse)
            r2 = abs(dv2 / i_pulse)
            if 0.0005 < r1 < 0.5 and 0.0005 < r2 < 0.5:
                r0_list.append(r0)
                r1_list.append(r1)
                tau1_list.append(tau1)
                r2_list.append(r2)
                tau2_list.append(tau2)
        except RuntimeError:
            continue

    return {
        "r0": np.array(r0_list), "r1": np.array(r1_list),
        "tau1": np.array(tau1_list), "r2": np.array(r2_list),
        "tau2": np.array(tau2_list),
    }


def build_soc_curve(soc_vals, y_vals, bin_width=0.05, min_points=2):
    """Agrupa (soc, y) en bins de SOC (redondeo a bin_width), toma la
    mediana por bin -- mas robusto que ajustar una spline directo sobre
    puntos individuales ruidosos. Devuelve (soc_grid, y_grid) ordenados,
    o None si no hay suficiente cobertura de SOC para una curva util."""
    binned = {}
    for soc, y in zip(soc_vals, y_vals):
        key = round(soc / bin_width) * bin_width
        binned.setdefault(key, []).append(y)
    if len(binned) < min_points:
        return None
    socs_sorted = sorted(binned.keys())
    y_sorted = [float(np.median(binned[k])) for k in socs_sorted]
    return np.array(socs_sorted), np.array(y_sorted)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    c20 = load_meas(os.path.join(HERE, "c20_ocv_25degC.mat"))
    hppc = load_meas(os.path.join(HERE, "hppc_25degC.mat"))

    soc_grid, v_grid, capacity_ah_c20 = extract_ocv_curve(c20)
    print(f"Capacidad medida (C/20, archivo del {'-'.join('05-08-17'.split('-'))}): "
          f"{capacity_ah_c20:.3f} Ah")

    # La celda se degrada con el tiempo entre sesiones de test (ver README
    # del dataset, paso 11: 1C cae de 2.8Ah a 2.3Ah tras ~110 ciclos). El
    # archivo HPPC (03-11-17) y el archivo C/20 usado para la curva OCV
    # (05-08-17) son de sesiones ~2 meses separadas -- verificado: la propia
    # descarga total dentro del archivo HPPC es 2.773 Ah, 6.6% menos que los
    # 2.968 Ah del C/20. Usar la capacidad del C/20 para convertir el Ah del
    # archivo HPPC a SOC arrastra ese 6.6% de error, que se amplifica en SOC
    # bajo por lo empinada que es la curva OCV ahi -- exactamente el patron
    # de error creciente visto en las primeras corridas. Se usa la capacidad
    # medida DENTRO del propio archivo HPPC para el SOC de referencia (la
    # forma de la curva OCV del C/20 se sigue reutilizando, normalizada a
    # SOC fraccional 0-1, que es insensible a este desajuste de capacidad).
    capacity_ah_hppc = float(abs(hppc["ah"].min()))
    print(f"Capacidad propia del archivo HPPC: {capacity_ah_hppc:.3f} Ah "
          f"({(capacity_ah_c20 - capacity_ah_hppc) / capacity_ah_c20 * 100:.1f}% "
          f"menos que el C/20 -- degradacion entre sesiones de test)")
    capacity_ah = capacity_ah_hppc

    pulses = find_pulses(hppc)
    print(f"Pulsos HPPC detectados: {len(pulses)}")
    soc_r0, r0_vals, soc_r1, r1_vals, c1 = fit_r0_r1_c1_per_pulse(
        hppc, pulses, capacity_ah)
    print(f"R0: {len(r0_vals)} pulsos ajustados, rango "
          f"[{r0_vals.min():.4f}, {r0_vals.max():.4f}] Ohm")
    print(f"R1: {len(r1_vals)} pulsos ajustados, rango "
          f"[{r1_vals.min():.4f}, {r1_vals.max():.4f}] Ohm")
    print(f"C1 derivado (mediana tau / mediana R1): {c1:.1f} F")

    from scipy.interpolate import interp1d
    ocv_func = interp1d(soc_grid, v_grid, kind="cubic",
                         bounds_error=False,
                         fill_value=(v_grid[0], v_grid[-1]))

    r0_curve = build_soc_curve(soc_r0, r0_vals)
    r1_curve = build_soc_curve(soc_r1, r1_vals)
    r0_func = r1_func = None
    r0_soc_grid = r0_soc_vals = r1_soc_grid = r1_soc_vals = None
    if r0_curve is not None:
        r0_soc_grid, r0_soc_vals = r0_curve
        r0_func = interp1d(r0_soc_grid, r0_soc_vals, kind="linear",
                            bounds_error=False,
                            fill_value=(r0_soc_vals[0], r0_soc_vals[-1]))
        print(f"R0(SOC): curva con {len(r0_soc_grid)} bins -- "
              f"{dict(zip(np.round(r0_soc_grid, 2), np.round(r0_soc_vals, 4)))}")
    if r1_curve is not None:
        r1_soc_grid, r1_soc_vals = r1_curve
        r1_func = interp1d(r1_soc_grid, r1_soc_vals, kind="linear",
                            bounds_error=False,
                            fill_value=(r1_soc_vals[0], r1_soc_vals[-1]))

    r0_median = float(np.median(r0_vals))
    r1_median = float(np.median(r1_vals))
    params = ESCParams(
        capacity_ah=capacity_ah,
        r0_ohm=r0_median,   # fallback si r0_func es None (no deberia pasar aqui)
        r1_ohm=r1_median,
        c1_farad=c1,
        hyst_m=0.0,   # no ajustado en esta primera pasada -- ver limitaciones
        hyst_m0=0.0,
        coulombic_eff=1.0,
        ocv_func=ocv_func,
        r0_func=r0_func,
        r1_func=r1_func,
    )

    # Perfil de corriente REAL del archivo HPPC completo (signo invertido:
    # dataset usa negativo=descarga, nuestro modelo usa positivo=descarga).
    t_real = hppc["t"]
    i_real = -hppc["i"]
    v_real = hppc["v"]
    ah_real = hppc["ah"]

    dt_arr = np.diff(t_real, prepend=t_real[0])
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 0.1

    # SOC de referencia (verdad de terreno) directo del Ah medido -- este
    # archivo HPPC solo loguea los pulsos, no las descargas ENTRE grupos de
    # pulsos (ver README del dataset, archivo "dis5_10p" aparte). El
    # contador Ah SI las contabiliza (verificado: saltos de hasta -0.18 Ah
    # entre grupos), pero integrar solo la corriente presente en ESTE
    # archivo subestima la descarga real y el voltaje simulado deriva por
    # encima del real con el tiempo. Se ancla z al Ah real en cada paso;
    # R1/histéresis se siguen actualizando con la dinámica real del
    # modelo -- esto valida específicamente la respuesta de impedancia
    # (R0/R1/OCV), no la integración de carga (ya resuelta trivialmente si
    # se tiene la corriente completa, que es lo que falta en este archivo).
    soc_reference = 1.0 + ah_real / capacity_ah

    model = ESCModel(params, z0=1.0)
    v_sim = np.zeros_like(v_real)
    for k in range(len(t_real)):
        model.z = float(np.clip(soc_reference[k], 0.0, 1.0))
        v_sim[k] = model.step(float(i_real[k]), float(dt_arr[k]))

    error_mv = (v_sim - v_real) * 1000.0
    rmse_mv = float(np.sqrt(np.mean(error_mv**2)))
    max_err_mv = float(np.max(np.abs(error_mv)))
    print(f"\nRMSE vs. celda real medida (1 rama RC): {rmse_mv:.2f} mV")
    print(f"Error maximo (1 rama RC): {max_err_mv:.2f} mV")

    # --- Experimento: 2 ramas RC (motivado por el mismo patron de error en
    # SOC bajo/corriente alta visto en 2 datasets reales independientes) ---
    fit2 = fit_two_rc_branches_per_pulse(hppc, pulses)
    n2 = len(fit2["r0"])
    print(f"\n--- 2 ramas RC: {n2} pulsos con relajacion bi-exponencial "
          f"resuelta ---")
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
            r2_ohm=r2_2, c2_farad=c2_2, hyst_m=0.0, hyst_m0=0.0,
            coulombic_eff=1.0, ocv_func=ocv_func)
        model_2rc = ESCModel(params_2rc, z0=1.0)
        v_sim_2rc = np.zeros_like(v_real)
        for k in range(len(t_real)):
            model_2rc.z = float(np.clip(soc_reference[k], 0.0, 1.0))
            v_sim_2rc[k] = model_2rc.step(float(i_real[k]), float(dt_arr[k]))
        error_2rc_mv = (v_sim_2rc - v_real) * 1000.0
        rmse_2rc_mv = float(np.sqrt(np.mean(error_2rc_mv**2)))
        max_err_2rc_mv = float(np.max(np.abs(error_2rc_mv)))
        print(f"RMSE vs. celda real medida (2 ramas RC): {rmse_2rc_mv:.2f} mV "
              f"(1 rama: {rmse_mv:.2f} mV)")
        print(f"Error maximo (2 ramas RC): {max_err_2rc_mv:.2f} mV "
              f"(1 rama: {max_err_mv:.2f} mV)")

        fig2, ax2 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax2[0].plot(t_real / 3600, v_real, "k", lw=1.0, label="Medido (real)")
        ax2[0].plot(t_real / 3600, v_sim, "steelblue", lw=0.8, ls="--",
                     label=f"1 rama RC (RMSE={rmse_mv:.1f}mV)")
        ax2[0].plot(t_real / 3600, v_sim_2rc, "darkorange", lw=0.8, ls=":",
                     label=f"2 ramas RC (RMSE={rmse_2rc_mv:.1f}mV)")
        ax2[0].set_ylabel("Voltaje [V]")
        ax2[0].legend(fontsize=9)
        ax2[0].set_title("Panasonic 18650PF: 1 vs. 2 ramas RC contra celda real")
        ax2[0].grid(alpha=0.3)
        ax2[1].plot(t_real / 3600, error_mv, "steelblue", lw=0.5, label="1 rama")
        ax2[1].plot(t_real / 3600, error_2rc_mv, "darkorange", lw=0.5, label="2 ramas")
        ax2[1].axhline(0, color="k", lw=0.5)
        ax2[1].set_ylabel("Error [mV]")
        ax2[1].set_xlabel("Tiempo [h]")
        ax2[1].legend(fontsize=9)
        ax2[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "panasonic_1rc_vs_2rc.png"),
                    dpi=130, bbox_inches="tight")
    else:
        print("Muy pocos pulsos con relajacion bi-exponencial resuelta "
              "(<5) -- no se genera comparacion 2-RC para este dataset.")

    csv_path = os.path.join(RESULTS_DIR, "validation_vs_panasonic18650pf.csv")
    np.savetxt(csv_path,
               np.column_stack([t_real, i_real, v_real, v_sim, error_mv,
                                 soc_reference]),
               header="t_s,current_a,v_measured,v_simulated,error_mv,soc_reference",
               delimiter=",", comments="")

    # Curva OCV y parametros ajustados, para que el harness de validacion en
    # C++ (validate_cpp_plugin.cc) use EXACTAMENTE los mismos valores sin
    # tener que re-implementar la extraccion/ajuste en C++ tambien.
    ocv_csv_path = os.path.join(RESULTS_DIR, "fitted_ocv_curve.csv")
    np.savetxt(ocv_csv_path, np.column_stack([soc_grid, v_grid]),
               header="soc,voltage", delimiter=",", comments="")
    params_path = os.path.join(RESULTS_DIR, "fitted_params.csv")
    with open(params_path, "w") as f:
        f.write("param,value\n")
        f.write(f"capacity_ah,{capacity_ah}\n")
        f.write(f"r0_ohm,{r0_median}\n")  # fallback/referencia -- el C++ usa la curva si existe
        f.write(f"r1_ohm,{r1_median}\n")
        f.write(f"c1_farad,{c1}\n")

    if r0_soc_grid is not None:
        np.savetxt(os.path.join(RESULTS_DIR, "fitted_r0_curve.csv"),
                   np.column_stack([r0_soc_grid, r0_soc_vals]),
                   header="soc,r0_ohm", delimiter=",", comments="")
    if r1_soc_grid is not None:
        np.savetxt(os.path.join(RESULTS_DIR, "fitted_r1_curve.csv"),
                   np.column_stack([r1_soc_grid, r1_soc_vals]),
                   header="soc,r1_ohm", delimiter=",", comments="")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t_real / 3600, i_real, "C1", lw=0.8)
    axes[0].set_ylabel("Corriente [A]")
    axes[0].set_title(
        "Modelo ESC vs. celda Panasonic 18650PF real medida "
        "(Kollmeyer/Univ. Wisconsin-Madison, doi:10.17632/wykht8y7tg.1)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_real / 3600, v_real, "k", lw=1.0, label="Medido (real)")
    axes[1].plot(t_real / 3600, v_sim, "steelblue", lw=0.8, ls="--",
                 label="ESC simulado (R0/R1/C1/OCV ajustados de este mismo dataset)")
    axes[1].set_ylabel("Voltaje [V]")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_real / 3600, error_mv, "purple", lw=0.6)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Error [mV]")
    axes[2].set_xlabel("Tiempo [h]")
    axes[2].set_title(f"RMSE = {rmse_mv:.1f} mV | error máx. = {max_err_mv:.1f} mV")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "validation_vs_panasonic18650pf.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"\nCSV: {csv_path}")
    print(f"Figura: {fig_path}")

    print("\n--- Parametros ajustados (para el plugin / calculations.md) ---")
    print(f"capacity_ah = {capacity_ah:.3f}")
    print(f"r0_ohm (mediana, fallback) = {r0_median:.4f}")
    print(f"r1_ohm (mediana, fallback) = {r1_median:.4f}")
    print(f"c1_farad = {c1:.1f}")


if __name__ == "__main__":
    main()
