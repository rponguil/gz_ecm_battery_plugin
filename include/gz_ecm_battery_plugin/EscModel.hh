// Copyright (c) 2026 Ronald Ponguillo-Intriago
// Licensed under the MIT License (see LICENSE at the root of this repo).
//
// Pure C++ port of the ESC (Enhanced Self-Correcting) equivalent-circuit
// model -- Plett (2015) Chapter 2 -- factored out of EscBatteryPlugin so it
// has NO Gazebo dependency and can be unit-tested / validated standalone
// against real measured data, exactly the same class the plugin uses in
// production (not a second, parallel reimplementation that could drift).
//
// 1:1 port of dron_px4_battery_models/esc_battery_model.py (ESCModel.step):
// same state variables (z, iR1, h, s), same update order, same equations.

#ifndef GZ_ECM_BATTERY_PLUGIN_ESCMODEL_HH_
#define GZ_ECM_BATTERY_PLUGIN_ESCMODEL_HH_

#include <algorithm>
#include <cmath>

#include "gz_ecm_battery_plugin/CubicSpline.hh"

namespace gz_ecm_battery_plugin
{

struct EscModelParams
{
  double capacityAh{2.5};
  double r0{0.05};   // used when r0Curve is empty
  double r1{0.02};   // used when r1Curve is empty (RC branch 1, slower)
  double c1{4000.0};
  // RC branch 2 (optional, r2<=0 = disabled = unchanged behaviour). Added
  // after validating against two independent real datasets (Panasonic
  // 18650PF and Molicel 21700-P42A) both showing the same large-error
  // pattern only at very low SOC / very high current -- see
  // validation_data/ in this repo and the literature it cites.
  double r2{0.0};    // RC branch 2 (faster, e.g. charge-transfer)
  double c2{100.0};
  double hystM{0.0};
  double hystM0{0.0};
  double hystGamma{50.0};
  double coulombicEff{1.0};
};

class EscModel
{
  public: EscModel() = default;

  public: void Configure(const EscModelParams &_params, CubicSpline _ocv)
  {
    this->p = _params;
    this->ocv = std::move(_ocv);
  }

  /// \brief Optional: make R0 vary with SOC instead of using the constant
  /// `EscModelParams::r0`. Real cells are more resistive near SOC extremes
  /// -- see validation_data/validate_against_panasonic18650pf.py for the
  /// fit this is meant to carry into the plugin.
  public: void SetR0Curve(CubicSpline _r0OfSoc)
  {
    this->r0Curve = std::move(_r0OfSoc);
    this->hasR0Curve = true;
  }

  public: void SetR1Curve(CubicSpline _r1OfSoc)
  {
    this->r1Curve = std::move(_r1OfSoc);
    this->hasR1Curve = true;
  }

  public: void Reset(double _z0 = 1.0)
  {
    this->z = _z0;
    this->iR1 = 0.0;
    this->iR2 = 0.0;
    this->h = 0.0;
    this->s = 0.0;
  }

  /// \brief Advance one time step. `_currentA` positive = discharge.
  /// Returns terminal voltage. Mirrors ESCModel.step() exactly.
  public: double Step(double _currentA, double _dt)
  {
    const double etaEff = (_currentA > 0.0) ? this->p.coulombicEff : 1.0;
    if (std::abs(_currentA) > 1e-9)
      this->s = (_currentA > 0.0) ? 1.0 : -1.0;

    const double socClamped = std::clamp(this->z, 0.0, 1.0);
    const double r0Eff = this->hasR0Curve
        ? this->r0Curve.Eval(socClamped) : this->p.r0;
    const double r1Eff = this->hasR1Curve
        ? this->r1Curve.Eval(socClamped) : this->p.r1;

    const double voltage = this->ocv.Eval(socClamped) +
        this->p.hystM0 * this->s + this->p.hystM * this->h -
        r1Eff * this->iR1 - this->p.r2 * this->iR2 - r0Eff * _currentA;

    const double qAs = this->p.capacityAh * 3600.0;
    const double aH = std::exp(
        -std::abs(etaEff * _currentA * this->p.hystGamma * _dt / qAs));

    this->z = std::clamp(
        this->z - (etaEff * _dt / qAs) * _currentA, 0.0, 1.0);

    if (r1Eff > 0.0)
    {
      const double f1 = std::exp(-_dt / (r1Eff * this->p.c1));
      this->iR1 = f1 * this->iR1 + (1.0 - f1) * _currentA;
    }
    else
    {
      this->iR1 = 0.0;
    }
    if (this->p.r2 > 0.0)
    {
      const double f2 = std::exp(-_dt / (this->p.r2 * this->p.c2));
      this->iR2 = f2 * this->iR2 + (1.0 - f2) * _currentA;
    }
    else
    {
      this->iR2 = 0.0;
    }
    this->h = std::clamp(
        aH * this->h - (1.0 - aH) * this->s, -1.0, 1.0);

    return std::max(voltage, 0.0);
  }

  /// \brief State of charge, [0, 1]. Public (like the Python dataclass) so
  /// a validation harness can anchor it to external ground truth when a
  /// dataset's own current log is incomplete -- see
  /// validation_data/validate_against_panasonic18650pf.py for why that is
  /// sometimes necessary. Normal (production) use never needs to touch
  /// this directly; EscBatteryPlugin only calls Step().
  public: double z{1.0};

  private: EscModelParams p;
  private: CubicSpline ocv;
  private: CubicSpline r0Curve;
  private: CubicSpline r1Curve;
  private: bool hasR0Curve{false};
  private: bool hasR1Curve{false};
  private: double iR1{0.0};
  private: double iR2{0.0};
  private: double h{0.0};
  private: double s{0.0};
};

}  // namespace gz_ecm_battery_plugin

#endif
