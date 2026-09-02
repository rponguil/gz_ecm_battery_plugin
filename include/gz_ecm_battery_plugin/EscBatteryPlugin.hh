// Copyright (c) 2026 Ronald Ponguillo-Intriago
//
// Licensed under the MIT License (see LICENSE at the root of this repo).
//
// Structural pattern (Configure/PreUpdate/Update/PostUpdate, common::Battery
// consumer/power-load bookkeeping, BatterySoC component, BatteryState topic)
// follows gz-sim's own LinearBatteryPlugin (Apache-2.0, Open Source Robotics
// Foundation) -- see https://github.com/gazebosim/gz-sim -- reused here as
// the established, tested way to integrate a battery into gz-sim's ECM.
// What is NEW here is the voltage model itself: an Enhanced Self-Correcting
// (ESC) equivalent-circuit model (one RC branch + dynamic/instantaneous
// hysteresis), per Gregory L. Plett, "Battery Management Systems, Volume 1:
// Battery Modeling", Artech House, 2015, Chapter 2 -- instead of the linear
// two-parameter open-circuit-voltage model LinearBatteryPlugin uses.

#ifndef GZ_ECM_BATTERY_PLUGIN_ESCBATTERYPLUGIN_HH_
#define GZ_ECM_BATTERY_PLUGIN_ESCBATTERYPLUGIN_HH_

#include <memory>

#include <gz/common/Battery.hh>
#include <gz/sim/System.hh>

namespace gz_ecm_battery_plugin
{
  class EscBatteryPluginPrivate;

  /// \brief Gazebo Sim battery plugin implementing an ECM (Equivalent
  /// Circuit Model) with one RC polarization branch and hysteresis --
  /// Plett's "Enhanced Self-Correcting" (ESC) model -- as a drop-in
  /// alternative to the stock LinearBatteryPlugin.
  ///
  /// ## Why this exists
  /// The stock LinearBatteryPlugin uses `V = e0 + e1*(1 - q/c) - r*i`: a
  /// single series resistance and a LINEAR open-circuit-voltage curve. Real
  /// Li-ion cells have a non-linear OCV(SOC) curve, a slow polarization
  /// (diffusion) dynamic that a single resistor cannot capture, and
  /// charge/discharge hysteresis. This plugin adds exactly those three
  /// things while keeping the same consumer/power-load/topic mechanics as
  /// LinearBatteryPlugin, so it can be swapped in with minimal SDF changes.
  ///
  /// ## System parameters
  /// Same as LinearBatteryPlugin: `<battery_name>`, `<voltage>`,
  /// `<capacity>` (Ah), `<initial_charge>` (Ah), `<power_load>` (W),
  /// `<smooth_current_tau>`, `<start_draining>`, `<power_draining_topic>`,
  /// `<stop_power_draining_topic>`.
  ///
  /// New / different from LinearBatteryPlugin:
  /// - `<r0>` series (instantaneous) resistance [Ohm], replaces `<resistance>`.
  /// - `<r1>` polarization branch resistance [Ohm].
  /// - `<c1>` polarization branch capacitance [F]. Time constant tau_RC = r1*c1.
  /// - `<hysteresis_m>` dynamic hysteresis amplitude [V] (0 to disable).
  /// - `<hysteresis_m0>` instantaneous hysteresis amplitude [V] (0 to disable).
  /// - `<hysteresis_gamma>` hysteresis decay rate (only used if hysteresis
  ///   amplitudes are non-zero).
  /// - `<coulombic_efficiency>` charging efficiency in (0, 1], default 1.0.
  /// - one or more `<ocv_point soc="0.0" voltage="3.0"/>` elements: the
  ///   open-circuit-voltage curve, in ascending SOC order. At least 2
  ///   points required. This is what makes the plugin chemistry-agnostic --
  ///   drop in any cell's datasheet discharge curve, no recompilation.
  class EscBatteryPlugin
      : public gz::sim::System,
        public gz::sim::ISystemConfigure,
        public gz::sim::ISystemPreUpdate,
        public gz::sim::ISystemUpdate,
        public gz::sim::ISystemPostUpdate
  {
    public: EscBatteryPlugin();
    public: ~EscBatteryPlugin() override;

    public: void Configure(const gz::sim::Entity &_entity,
                            const std::shared_ptr<const sdf::Element> &_sdf,
                            gz::sim::EntityComponentManager &_ecm,
                            gz::sim::EventManager &_eventMgr) final;

    public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                            gz::sim::EntityComponentManager &_ecm) override;

    public: void Update(const gz::sim::UpdateInfo &_info,
                         gz::sim::EntityComponentManager &_ecm) final;

    public: void PostUpdate(
                const gz::sim::UpdateInfo &_info,
                const gz::sim::EntityComponentManager &_ecm) override;

    /// \brief Voltage-update callback bound to the common::Battery. Runs the
    /// ESC model one time step and returns the new terminal voltage.
    private: double OnUpdateVoltage(const gz::common::Battery *_battery);

    private: std::unique_ptr<EscBatteryPluginPrivate> dataPtr;
  };

}  // namespace gz_ecm_battery_plugin

#endif
