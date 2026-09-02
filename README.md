# gz_ecm_battery_plugin

A **chemistry-agnostic** Equivalent-Circuit-Model (ECM) battery plugin for **Gazebo Sim**
(tested on Gazebo Harmonic, `gz-sim8`), implementing Plett's "Enhanced Self-Correcting" (ESC)
model — one RC polarization branch plus dynamic/instantaneous hysteresis — as a **drop-in
alternative to the stock `LinearBatteryPlugin`**.

Verified: **no drone-racing/agile-flight simulation framework we surveyed ships with any
battery model at all** — not Aerostack2 (CVAR-UPM), not Flightmare (UZH-RPG). Even Gazebo's own
default battery plugin is a plain linear open-circuit-voltage model. This plugin exists to fill
that gap without forcing anyone to hand-tune C++ for their specific cell — every physical
parameter (including the whole OCV(SOC) curve) is set from SDF, no recompilation needed to
model a different cell chemistry.

## Why not just use LinearBatteryPlugin?

`V = e0 + e1*(1 - q/c) - r*i` is a single series resistance and a **linear** OCV curve. Real
Li-ion cells have a non-linear OCV(SOC) curve, a slow polarization/diffusion dynamic a single
resistor can't capture, and charge/discharge hysteresis. This plugin adds exactly those three
things while keeping the same consumer/power-load/topic mechanics as `LinearBatteryPlugin`, so
swapping between them only requires SDF changes, not a different simulation setup.

## Build

Dependencies: Gazebo Sim (`gz-sim8`/Harmonic — the four `gz-*` packages below), a C++17
compiler. No ROS 2 dependency in the plugin code itself, but it builds fine inside a ROS 2
workspace via `colcon` too (has a `package.xml`).

```bash
# plain CMake
mkdir build && cd build
cmake .. && make

# or, inside a ROS 2 workspace
colcon build --packages-select gz_ecm_battery_plugin
```

## Test

Unit tests (CubicSpline + EscModel invariants: SOC stays in [0,1], voltage never negative,
RC branches disabled without producing NaN/Inf, SOC-dependent curves override the constant
fallback correctly) run via GTest/CTest, built automatically if `libgtest-dev` (or, inside a
ROS 2 workspace, `ros-jazzy-gtest-vendor`) is found:

```bash
cd build && ctest --output-on-failure
# or directly: ./test_esc_model
```

These are fast, standalone unit tests (no Gazebo needed). The against-real-cell validation
(comparing the plugin's output to measured voltage from public battery datasets) is a
separate, heavier check documented in `validation_data/`.

## Parameterizing it for your own cell — read this first

Where you get the OCV(SOC) curve from matters **more than the model structure**. In a
controlled experiment on the same cell, the same test and the same fitted `R0`/`R1`/`C1`,
changing only the source of the OCV curve moved RMSE by a factor of 2.5:

| OCV curve taken from | RMSE vs. measured cell |
|---|---|
| A separate slow-discharge (C/20) session, recorded ~2 months earlier | 85.7 mV |
| The rest periods **inside the same pulse-test file** | **34.3 mV** |

Reproduce it with `python3 validation_data/validate_ocv_provenance.py`.

**Guidance:** extract your OCV curve from the same test session as the pulse data you use to
fit the resistances, using the relaxation voltage at the end of each rest period — not from a
separately recorded discharge curve, and not from a datasheet plot. Cells age between
sessions (6.6% capacity difference in the case above), and that mismatch propagates into
every voltage the model predicts. A same-session curve gave ~34 mV RMSE on two entirely
different cells (Panasonic 18650PF NCA and Molicel INR-21700-P42A NMC).

This does **not** fix the worst-case error near end of discharge, which has a different
cause and is documented as an open limitation.

## Try it

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/build:$GZ_SIM_SYSTEM_PLUGIN_PATH   # plain cmake build
# or, if built via colcon: source install/setup.bash (plugin path is set automatically)

gz sim worlds/ecm_battery_demo.sdf
```

Watch the battery state live:
```bash
gz topic -e -t /model/battery_test_box/battery/lir18650_esc/state
```

There's also a second, more interesting example: `worlds/quadrotor_cvar_racing_with_battery.sdf`
attaches this plugin to CVAR-UPM's own racing drone model (`quadrotor_cvar_racing`, from
`aerostack2`) via `<include>+<plugin>`, without touching the original model file. Needs
`GZ_SIM_RESOURCE_PATH` to include the `as2_simulation_assets/as2_gazebo_assets/models`
directory so `model://quadrotor_cvar_racing` resolves.

**Troubleshooting: `gz topic -l`/`gz topic -e` show nothing even though the sim is running.**
This is a gz-transport multicast discovery issue, not a plugin bug -- on machines with
multiple network interfaces (Docker bridges are a common culprit), the client tool and the
`gz sim` server can fail to discover each other. Fix: set `GZ_IP=127.0.0.1` and a fixed
`GZ_PARTITION` (any string, just needs to match between processes started in the same shell)
before launching.

## SDF parameters

Same as `LinearBatteryPlugin`: `<battery_name>`, `<voltage>` (optional — defaults to OCV at
SOC=1 if omitted), `<capacity>` (Ah), `<initial_charge>` (Ah), `<power_load>` (W),
`<smooth_current_tau>`, `<start_draining>`.

New (the ECM part):

| Parameter | Meaning | Units |
|---|---|---|
| `<r0>` | Series (instantaneous) resistance | Ohm |
| `<r1>` | Polarization branch resistance (0 = disable the RC branch) | Ohm |
| `<c1>` | Polarization branch capacitance | F |
| `<hysteresis_m>` | Dynamic hysteresis amplitude (0 = disable) | V |
| `<hysteresis_m0>` | Instantaneous hysteresis amplitude (0 = disable) | V |
| `<hysteresis_gamma>` | Hysteresis decay rate | — |
| `<coulombic_efficiency>` | Charging efficiency, (0, 1] | — |
| `<ocv_point soc="..." voltage="...">` | One point of the OCV(SOC) curve. **At least 2 required, ascending SOC order.** This is what makes the plugin work for any cell — just drop in your own datasheet discharge curve. | — |

See `worlds/ecm_battery_demo.sdf` for a complete working example (LIR18650/NMC parameters,
same values validated in the companion Python model, see below).

## Relationship to other work in this line of research

- Math ported 1:1 from the validated Python reference implementation used to produce the
  RMSE numbers reported in the `Dron_PX4_ROS2` project (`esc_battery_model.py`, itself
  following Plett 2015 Chapter 2). This plugin is the **live, in-simulation** version of that
  same model.
- Built to attach to any Gazebo model — including `aerostack2`'s `quadrotor_cvar_racing` and
  `x500`, neither of which ship with any battery plugin (verified by inspecting their SDF —
  see `Dron_Racing_UPM/02_OTROS_REPOS_Y_GAPS.md`).
- Gazebo integration scaffolding (Configure/PreUpdate/Update/PostUpdate, `common::Battery`
  consumer bookkeeping, `BatterySoC` component, `BatteryState` topic) follows the structure of
  Gazebo Sim's own `LinearBatteryPlugin.cc` (Apache-2.0, Open Source Robotics Foundation) — the
  established, tested way to integrate a battery into the ECM (Entity Component Manager, not to
  be confused with the battery model also called ECM — unfortunate acronym collision, both are
  standard terms in their own fields).

## Citation

See `CITATION.cff`. Repository: https://github.com/rponguil/gz_ecm_battery_plugin (currently
private — will be made public before/at paper submission; a Zenodo DOI will be minted from the
first public release).

## License

MIT — see `LICENSE`. Deliberately permissive (unlike some other academic robotics software in
this space, which restricts reuse to non-commercial purposes) so anyone can adopt it, modify
it, and cite it back.
