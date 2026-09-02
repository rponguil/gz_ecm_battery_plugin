"""Modelo ESC (Enhanced Self-Correcting) de celda Li-ion -- Plett (2015), Cap. 2.

Portado tal cual (misma matematica, sin reinventarla) desde
`OTROS/3. Research/12. Baterias/notebooks/ch02_equivalent_circuit_models.ipynb`
(clase `ESCModel` / `ESCModel_nRC` del notebook), reescrito como modulo reutilizable
tanto por un nodo ROS2 en vivo (`esc_battery_node.py`) como por el script de comparacion
offline (`scripts/compare_and_plot.py`) -- una sola fuente de verdad para el modelo.

Referencia: Gregory L. Plett, *Battery Management Systems, Volume 1: Battery Modeling*,
Artech House, 2015, Cap. 2 (Equivalent-Circuit Models).

Diferencia respecto al demo del notebook: la curva OCV(SOC) de abajo es representativa de
una celda Li-ion NMC/grafito (18650, ~3.6V nominal, 4.2V full), no la LiFePO4 usada como
demo en el notebook -- es la quimica tipica de un pack LiPo/Li-ion de dron (ver BOM.md), la
LiFePO4 del notebook original tiene una curva mucho mas plana y no es representativa aqui.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

# OCV(SOC) para celda Li-ion NMC/grafito, 18650-tipo (curva tipica, ver
# 00_index.ipynb -> "Typical Li-ion Cell Parameters (NMC/Graphite, 18650-type)":
# V_nom=3.6V. Puntos de referencia de forma de curva NMC estandar en la literatura
# (p.ej. parametrizacion tipo Chen 2020, tambien citada en 00_index.ipynb).
_NMC_SOC_POINTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                             0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
_NMC_OCV_POINTS = np.array([3.00, 3.30, 3.45, 3.55, 3.62, 3.68, 3.72,
                             3.78, 3.85, 3.95, 4.08, 4.14, 4.20])

NMC_OCV_FUNC = interp1d(_NMC_SOC_POINTS, _NMC_OCV_POINTS, kind="cubic",
                         bounds_error=False,
                         fill_value=(_NMC_OCV_POINTS[0], _NMC_OCV_POINTS[-1]))


@dataclass
class ESCParams:
    """Parametros de celda -- misma celda LIR18650 que usa el demo oficial de Gazebo
    (`examples/worlds/linear_battery_demo.sdf`, rama gz-sim8: "Li-ion battery spec from
    LIR18650 datasheet") para que la comparacion sea sobre la MISMA celda fisica, no dos
    celdas distintas con parametros inventados por separado.

    R0+R1 = 0.07 Ohm = la resistencia serie total que usa el modelo lineal de Gazebo
    (`resistance` en el SDF) -- el ESC divide esa misma resistencia total en una
    componente instantanea (R0) y una de relajacion/difusion (R1*rama RC), que es
    exactamente la riqueza fisica que al modelo lineal le falta por diseno.
    """

    capacity_ah: float = 2.5       # Q -- igual a <capacity> del demo de Gazebo
    r0_ohm: float = 0.050          # resistencia serie instantanea [Ohm] (usado si r0_func es None)
    r1_ohm: float = 0.020          # resistencia de la rama RC 1 (difusion lenta) [Ohm] (usado si r1_func es None)
    c1_farad: float = 4000.0       # capacitancia de la rama RC 1 [F] (tau ~= 80s, celda pequena)
    # Rama RC 2 (opcional, r2_ohm=0 = desactivada = comportamiento previo sin
    # cambios). Motivada por validacion contra 2 datasets reales
    # independientes (Panasonic 18650PF y Molicel 21700-P42A, ver
    # Dron_Racing_UPM/gz_ecm_battery_plugin/PAPER_DRAFT_OUTLINE.md): ambos
    # muestran el mismo patron de error grande solo en SOC muy bajo/corriente
    # muy alta -- consistente con la literatura de que una sola rama RC no
    # separa bien la dinamica rapida (transferencia de carga) de la lenta
    # (difusion); dos ramas RC reducen el error maximo de forma consistente
    # en varios estudios comparativos publicados.
    r2_ohm: float = 0.0            # resistencia de la rama RC 2 (transferencia de carga, mas rapida) [Ohm]
    c2_farad: float = 100.0        # capacitancia de la rama RC 2 [F]
    hyst_m: float = 0.010          # M, amplitud histeresis dinamica [V]
    hyst_m0: float = 0.003         # M0, amplitud histeresis instantanea [V]
    hyst_gamma: float = 50.0       # tasa de decaimiento de histeresis
    coulombic_eff: float = 0.998   # eta, eficiencia coulombica al cargar
    ocv_func: callable = field(default=NMC_OCV_FUNC)
    # Opcionales: R0(SOC)/R1(SOC) -- si se proveen, reemplazan a r0_ohm/r1_ohm
    # como funcion del SOC en vez de constante. None = comportamiento previo
    # sin cambios (celdas reales tienen impedancia mayor en los extremos de
    # SOC; ver validation_data/validate_against_panasonic18650pf.py en
    # Dron_Racing_UPM para el ajuste real que motiva esto).
    r0_func: callable = None
    r1_func: callable = None


class ESCModel:
    """Modelo ESC de una rama RC + histeresis dinamica + instantanea (Plett Cap. 2).

    Uso en vivo (nodo ROS2): crear una instancia, llamar `step(i, dt)` en cada
    iteracion -- mantiene su propio estado interno (z, iR, h).

    Uso offline (comparacion/notebook): `simulate(current_profile, dt)` corre el
    modelo completo sobre un array de corriente y devuelve las trayectorias.
    """

    def __init__(self, params: ESCParams, z0: float = 1.0):
        self.p = params
        self.q_as = params.capacity_ah * 3600.0  # Ah -> A*s
        self.z = z0
        self.i_r1 = 0.0
        self.i_r2 = 0.0
        self.h = 0.0
        self.s = 0.0

    def reset(self, z0: float = 1.0):
        self.z = z0
        self.i_r1 = 0.0
        self.i_r2 = 0.0
        self.h = 0.0
        self.s = 0.0

    @staticmethod
    def _f_rc(dt: float, r: float, c: float) -> float:
        # r<=0 => rama desactivada (evita division por cero); la corriente de
        # esa rama se mantiene en 0 en step().
        if r <= 0.0:
            return 0.0
        return np.exp(-dt / (r * c))

    def step(self, current_a: float, dt: float) -> float:
        """Avanza un paso de tiempo. `current_a` positivo = descarga. Devuelve v[k]."""
        eta = self.p.coulombic_eff if current_a > 0 else 1.0
        soc_clamped = np.clip(self.z, 0.0, 1.0)
        r0 = self.p.r0_func(soc_clamped) if self.p.r0_func is not None else self.p.r0_ohm
        r1 = self.p.r1_func(soc_clamped) if self.p.r1_func is not None else self.p.r1_ohm
        r2 = self.p.r2_ohm

        if abs(current_a) > 1e-9:
            self.s = np.sign(current_a)

        v = (self.p.ocv_func(soc_clamped)
             + self.p.hyst_m0 * self.s
             + self.p.hyst_m * self.h
             - r1 * self.i_r1
             - r2 * self.i_r2
             - r0 * current_a)

        f1 = self._f_rc(dt, r1, self.p.c1_farad)
        f2 = self._f_rc(dt, r2, self.p.c2_farad)
        a_h = np.exp(-abs(eta * current_a * self.p.hyst_gamma * dt / self.q_as))

        self.z = float(np.clip(self.z - (eta * dt / self.q_as) * current_a, 0.0, 1.0))
        self.i_r1 = f1 * self.i_r1 + (1 - f1) * current_a if r1 > 0.0 else 0.0
        self.i_r2 = f2 * self.i_r2 + (1 - f2) * current_a if r2 > 0.0 else 0.0
        self.h = float(np.clip(a_h * self.h - (1 - a_h) * np.sign(current_a), -1.0, 1.0))

        return float(v)

    def simulate(self, current_profile: np.ndarray, dt: float, z0: float = 1.0):
        """Corre el modelo sobre un perfil de corriente completo (uso offline)."""
        self.reset(z0)
        n = len(current_profile)
        t = np.arange(n) * dt
        v = np.zeros(n)
        z = np.zeros(n)
        for k in range(n):
            v[k] = self.step(float(current_profile[k]), dt)
            z[k] = self.z
        return t, v, z
