// Copyright (c) 2026 Ronald Ponguillo-Intriago
// Licensed under the MIT License (see LICENSE at the root of this repo).
//
// ESC (Enhanced Self-Correcting) model math ported from the validated
// Python reference implementation:
//   dron_px4_battery_models/esc_battery_model.py (ESCModel.step), itself a
//   port of Plett (2015) Chapter 2. Same equations, same state update order
//   -- see that file's docstring for the full citation and derivation notes.
// gz-sim integration scaffolding (Configure/PreUpdate/Update/PostUpdate,
// common::Battery consumer bookkeeping, BatterySoC component, BatteryState
// publisher) follows the structure of gz-sim's own LinearBatteryPlugin.cc
// (Apache-2.0, Open Source Robotics Foundation).

#include "gz_ecm_battery_plugin/EscBatteryPlugin.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <gz/msgs/battery_state.pb.h>

#include <gz/common/Battery.hh>
#include <gz/common/Console.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <sdf/Element.hh>

#include "gz/sim/Model.hh"
#include "gz/sim/components/BatteryPowerLoad.hh"
#include "gz/sim/components/BatterySoC.hh"
#include "gz/sim/components/Name.hh"

#include "gz_ecm_battery_plugin/CubicSpline.hh"
#include "gz_ecm_battery_plugin/EscModel.hh"

using namespace gz_ecm_battery_plugin;

namespace gz_ecm_battery_plugin
{

class EscBatteryPluginPrivate
{
  public: double StateOfCharge() const { return this->escModel.z; }

  public: std::string modelName;
  public: std::string batteryName;
  public: gz::common::BatteryPtr battery;
  public: int32_t consumerId{-1};
  public: gz::sim::Entity batteryEntity{gz::sim::kNullEntity};
  public: gz::sim::Model model{gz::sim::kNullEntity};

  // capacityAh kept here too (not just inside EscModelParams) because it is
  // also needed to fill BatteryState.capacity/charge on every PostUpdate.
  public: double capacityAh{0.0};

  // The actual ECM (Plett ESC) physics -- same class validated standalone
  // in validation_data/, not a separate copy of the math.
  public: EscModel escModel;

  // Shared bookkeeping (same role as in LinearBatteryPlugin).
  public: double tau{1.0};
  public: double iraw{0.0};
  public: double ismooth{0.0};
  public: double initialPowerLoad{0.0};
  public: bool startDraining{false};
  public: std::chrono::steady_clock::duration stepSize{0};

  public: gz::transport::Node node;
  public: gz::transport::Node::Publisher statePub;
};

}  // namespace gz_ecm_battery_plugin

/////////////////////////////////////////////////
EscBatteryPlugin::EscBatteryPlugin()
    : dataPtr(std::make_unique<EscBatteryPluginPrivate>())
{
}

/////////////////////////////////////////////////
EscBatteryPlugin::~EscBatteryPlugin()
{
  if (this->dataPtr->battery)
  {
    if (this->dataPtr->consumerId != -1)
      this->dataPtr->battery->RemoveConsumer(this->dataPtr->consumerId);
    this->dataPtr->battery->ResetUpdateFunc();
  }
}

/////////////////////////////////////////////////
void EscBatteryPlugin::Configure(const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager & /*_eventMgr*/)
{
  auto model = gz::sim::Model(_entity);
  if (!model.Valid(_ecm))
  {
    gzerr << "EscBatteryPlugin must be attached to a model entity. "
          << "Failed to initialize." << std::endl;
    return;
  }
  this->dataPtr->model = model;
  this->dataPtr->modelName = model.Name(_ecm);

  if (!_sdf->HasElement("capacity"))
  {
    gzerr << "EscBatteryPlugin: <capacity> (Ah) is required.\n";
    return;
  }
  this->dataPtr->capacityAh = _sdf->Get<double>("capacity");
  if (this->dataPtr->capacityAh <= 0)
  {
    gzerr << "EscBatteryPlugin: <capacity> must be > 0.\n";
    return;
  }

  EscModelParams escParams;
  escParams.capacityAh = this->dataPtr->capacityAh;
  escParams.r0 = _sdf->Get<double>("r0", 0.05).first;
  escParams.r1 = _sdf->Get<double>("r1", 0.0).first;
  escParams.c1 = _sdf->Get<double>("c1", 1.0).first;
  if (escParams.c1 <= 0)
  {
    gzerr << "EscBatteryPlugin: <c1> must be > 0. Using 1.0.\n";
    escParams.c1 = 1.0;
  }
  // Segunda rama RC, opcional (<r2> ausente o <=0 = desactivada, mismo
  // comportamiento que antes). Ver EscModel.hh para la motivación.
  escParams.r2 = _sdf->Get<double>("r2", 0.0).first;
  escParams.c2 = _sdf->Get<double>("c2", 1.0).first;
  if (escParams.c2 <= 0)
  {
    gzerr << "EscBatteryPlugin: <c2> must be > 0. Using 1.0.\n";
    escParams.c2 = 1.0;
  }
  escParams.hystM = _sdf->Get<double>("hysteresis_m", 0.0).first;
  escParams.hystM0 = _sdf->Get<double>("hysteresis_m0", 0.0).first;
  escParams.hystGamma = _sdf->Get<double>("hysteresis_gamma", 50.0).first;
  escParams.coulombicEff = _sdf->Get<double>("coulombic_efficiency", 1.0).first;

  // OCV(SOC) curve: one or more <ocv_point soc="..." voltage="..."/>.
  std::vector<double> socPts;
  std::vector<double> ocvPts;
  sdf::ElementConstPtr ocvElem = _sdf->FindElement("ocv_point");
  while (ocvElem)
  {
    socPts.push_back(ocvElem->Get<double>("soc"));
    ocvPts.push_back(ocvElem->Get<double>("voltage"));
    ocvElem = ocvElem->GetNextElement("ocv_point");
  }
  if (socPts.size() < 2)
  {
    gzerr << "EscBatteryPlugin: need at least 2 <ocv_point> elements "
          << "(got " << socPts.size() << "). Failed to initialize.\n";
    return;
  }
  for (std::size_t i = 1; i < socPts.size(); ++i)
  {
    if (socPts[i] <= socPts[i - 1])
    {
      gzerr << "EscBatteryPlugin: <ocv_point> soc values must be strictly "
            << "increasing.\n";
      return;
    }
  }
  CubicSpline ocvSpline;
  ocvSpline.Build(socPts, ocvPts);

  // --- Parameterization sanity checks -------------------------------------
  //
  // An equivalent-circuit model can only reproduce the end-of-discharge
  // voltage collapse if its OCV curve actually reaches down to a depleted
  // state. If the lowest OCV point sits well above the voltage the cell
  // really reaches, the model is *structurally* unable to predict that
  // collapse, no matter how well R0/R1/C1 are fitted -- and end of discharge
  // is precisely where low-voltage failsafes and return-to-home decisions are
  // made. This was measured: an OCV curve whose floor sat 733 mV above the
  // measured minimum made the modelled failsafe fire hours late, while a
  // curve reaching within 362 mV of it fired within a minute of the real
  // cell. See the paper and validation_data/validate_ocv_provenance.py.
  //
  // These are warnings, not errors: the user may legitimately be simulating
  // only a partial discharge, or a chemistry with a different cutoff.
  {
    const double socFloor = socPts.front();
    const double ocvFloor = ocvPts.front();

    if (socFloor > 0.02)
    {
      gzwarn << "EscBatteryPlugin: the OCV curve starts at SOC=" << socFloor
             << ", so everything below that is flat-extrapolated. Real cells "
             << "drop steeply near 0% SOC; end-of-discharge behaviour will be "
             << "optimistic. Add <ocv_point> entries down to SOC=0.\n";
    }

    // Largest ohmic drop this configuration can produce, i.e. the deepest
    // terminal voltage the model can ever reach.
    const double rTotal = escParams.r0
        + (escParams.r1 > 0.0 ? escParams.r1 : 0.0)
        + (escParams.r2 > 0.0 ? escParams.r2 : 0.0);
    double iMax = 0.0;
    if (_sdf->HasElement("power_load") && ocvFloor > 0.0)
      iMax = _sdf->Get<double>("power_load") / ocvFloor;
    const double vFloorReachable = ocvFloor - iMax * rTotal;

    gzmsg << "EscBatteryPlugin: OCV curve spans SOC ["
          << socFloor << ", " << socPts.back() << "], voltage ["
          << ocvFloor << ", " << ocvPts.back() << "] V. Deepest terminal "
          << "voltage this configuration can reach is about "
          << vFloorReachable << " V (OCV floor minus " << iMax * rTotal
          << " V of ohmic drop at the configured load). Compare that against "
          << "your cell's datasheet cutoff: if the cutoff is lower, the model "
          << "cannot reproduce the final part of the discharge.\n";
  }
  // ------------------------------------------------------------------------

  double initVoltage = ocvSpline.Eval(1.0);
  if (_sdf->HasElement("voltage"))
    initVoltage = _sdf->Get<double>("voltage");

  double z0 = 1.0;
  if (_sdf->HasElement("initial_charge"))
  {
    double q0 = _sdf->Get<double>("initial_charge");
    z0 = std::clamp(q0 / this->dataPtr->capacityAh, 0.0, 1.0);
  }
  this->dataPtr->escModel.Configure(escParams, ocvSpline);

  // Optional: R0(SOC) / R1(SOC) curves, same repeated-element pattern as
  // <ocv_point>. Real cells are more resistive near SOC extremes; if these
  // are absent, the plugin keeps using the constant <r0>/<r1> as before
  // (fully backward compatible).
  auto buildOptionalCurve = [&](const char *_elemName,
      std::function<void(CubicSpline)> _setter)
  {
    std::vector<double> xs, ys;
    sdf::ElementConstPtr elem = _sdf->FindElement(_elemName);
    while (elem)
    {
      xs.push_back(elem->Get<double>("soc"));
      ys.push_back(elem->Get<double>("ohm"));
      elem = elem->GetNextElement(_elemName);
    }
    if (xs.size() >= 2)
    {
      CubicSpline spline;
      spline.Build(xs, ys);
      _setter(std::move(spline));
      gzmsg << "EscBatteryPlugin: using SOC-dependent " << _elemName
            << " (" << xs.size() << " points).\n";
    }
    else if (xs.size() == 1)
    {
      gzwarn << "EscBatteryPlugin: need >= 2 <" << _elemName
             << "> to build a curve, got 1. Ignoring, using constant.\n";
    }
  };
  buildOptionalCurve("r0_point", [this](CubicSpline s)
      { this->dataPtr->escModel.SetR0Curve(std::move(s)); });
  buildOptionalCurve("r1_point", [this](CubicSpline s)
      { this->dataPtr->escModel.SetR1Curve(std::move(s)); });

  this->dataPtr->escModel.Reset(z0);

  if (!_sdf->HasElement("battery_name"))
  {
    gzerr << "EscBatteryPlugin: <battery_name> is required.\n";
    return;
  }
  this->dataPtr->batteryName = _sdf->Get<std::string>("battery_name");

  this->dataPtr->tau = _sdf->Get<double>("smooth_current_tau", 1.0).first;
  if (this->dataPtr->tau <= 0)
    this->dataPtr->tau = 1.0;

  this->dataPtr->batteryEntity = _ecm.CreateEntity();
  _ecm.CreateComponent(this->dataPtr->batteryEntity,
      gz::sim::components::Name(this->dataPtr->batteryName));
  _ecm.SetParentEntity(this->dataPtr->batteryEntity, _entity);

  this->dataPtr->battery = std::make_shared<gz::common::Battery>(
      this->dataPtr->batteryName, initVoltage);
  this->dataPtr->battery->Init();
  this->dataPtr->battery->SetUpdateFunc(
      std::bind(&EscBatteryPlugin::OnUpdateVoltage, this,
          std::placeholders::_1));

  if (_sdf->HasElement("power_load"))
  {
    this->dataPtr->initialPowerLoad = _sdf->Get<double>("power_load");
    this->dataPtr->consumerId = this->dataPtr->battery->AddConsumer();
    if (!this->dataPtr->battery->SetPowerLoad(
            this->dataPtr->consumerId, this->dataPtr->initialPowerLoad))
      gzerr << "EscBatteryPlugin: failed to set initial power load.\n";
  }
  else
  {
    gzwarn << "EscBatteryPlugin: no <power_load> specified.\n";
  }

  this->dataPtr->startDraining =
      _sdf->Get<bool>("start_draining", false).first;

  _ecm.CreateComponent(this->dataPtr->batteryEntity,
      gz::sim::components::BatterySoC(z0));

  std::string stateTopic{"/model/" + this->dataPtr->modelName +
      "/battery/" + this->dataPtr->batteryName + "/state"};
  auto validTopic = gz::transport::TopicUtils::AsValidTopic(stateTopic);
  if (validTopic.empty())
  {
    gzerr << "EscBatteryPlugin: invalid state topic [" << stateTopic
          << "]\n";
    return;
  }
  gz::transport::AdvertiseMessageOptions opts;
  opts.SetMsgsPerSec(50);
  this->dataPtr->statePub =
      this->dataPtr->node.Advertise<gz::msgs::BatteryState>(validTopic, opts);

  gzmsg << "EscBatteryPlugin configured. Battery [" << this->dataPtr->batteryName
        << "] on model [" << this->dataPtr->modelName << "], capacity "
        << this->dataPtr->capacityAh << " Ah, R0=" << escParams.r0
        << " Ohm, R1=" << escParams.r1 << " Ohm, C1="
        << escParams.c1 << " F, R2=" << escParams.r2 << " Ohm, C2="
        << escParams.c2 << " F, publishing on [" << stateTopic << "]"
        << std::endl;
}

/////////////////////////////////////////////////
void EscBatteryPlugin::PreUpdate(const gz::sim::UpdateInfo & /*_info*/,
    gz::sim::EntityComponentManager &_ecm)
{
  if (!this->dataPtr->battery)
    return;

  double totalPowerLoad = this->dataPtr->initialPowerLoad;
  _ecm.Each<gz::sim::components::BatteryPowerLoad>(
      [&](const gz::sim::Entity &,
          const gz::sim::components::BatteryPowerLoad *_load) -> bool
      {
        if (_load->Data().batteryId == this->dataPtr->batteryEntity)
          totalPowerLoad += _load->Data().batteryPowerLoad;
        return true;
      });

  if (!this->dataPtr->battery->SetPowerLoad(
          this->dataPtr->consumerId, totalPowerLoad))
    gzerr << "EscBatteryPlugin: failed to set consumer power load.\n";
}

/////////////////////////////////////////////////
void EscBatteryPlugin::Update(const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
  if (!this->dataPtr->battery || _info.paused)
    return;

  this->dataPtr->stepSize = _info.dt;
  this->dataPtr->battery->Update();

  auto *soc = _ecm.Component<gz::sim::components::BatterySoC>(
      this->dataPtr->batteryEntity);
  if (soc)
    soc->Data() = this->dataPtr->StateOfCharge();
}

/////////////////////////////////////////////////
void EscBatteryPlugin::PostUpdate(const gz::sim::UpdateInfo &_info,
    const gz::sim::EntityComponentManager & /*_ecm*/)
{
  if (_info.paused || !this->dataPtr->statePub || !this->dataPtr->battery)
    return;

  gz::msgs::BatteryState msg;
  msg.set_voltage(this->dataPtr->battery->Voltage());
  msg.set_current(this->dataPtr->ismooth);
  msg.set_charge(this->dataPtr->StateOfCharge() * this->dataPtr->capacityAh);
  msg.set_capacity(this->dataPtr->capacityAh);
  msg.set_percentage(this->dataPtr->StateOfCharge());
  msg.set_power_supply_status(this->dataPtr->startDraining
      ? gz::msgs::BatteryState::DISCHARGING
      : gz::msgs::BatteryState::NOT_CHARGING);
  this->dataPtr->statePub.Publish(msg);
}

/////////////////////////////////////////////////
double EscBatteryPlugin::OnUpdateVoltage(const gz::common::Battery *_battery)
{
  const double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
      this->dataPtr->stepSize).count() * 1e-9;
  if (dt <= 0.0)
    return _battery->Voltage();

  // Convert the summed power load [W] into a discharge current [A] --
  // same convention as LinearBatteryPlugin (positive = discharging).
  double totalPower = 0.0;
  if (this->dataPtr->startDraining)
  {
    for (const auto &load : _battery->PowerLoads())
      totalPower += load.second;
  }
  this->dataPtr->iraw = totalPower / std::max(_battery->Voltage(), 1e-3);

  const double k = std::min(dt / this->dataPtr->tau, 1.0);
  this->dataPtr->ismooth += k * (this->dataPtr->iraw - this->dataPtr->ismooth);
  const double currentA = this->dataPtr->ismooth;

  // Same EscModel class validated standalone against real measured cell
  // data in validation_data/ -- not a parallel reimplementation.
  return this->dataPtr->escModel.Step(currentA, dt);
}

GZ_ADD_PLUGIN(EscBatteryPlugin, gz::sim::System,
    EscBatteryPlugin::ISystemConfigure,
    EscBatteryPlugin::ISystemPreUpdate,
    EscBatteryPlugin::ISystemUpdate,
    EscBatteryPlugin::ISystemPostUpdate)

GZ_ADD_PLUGIN_ALIAS(EscBatteryPlugin,
    "gz_ecm_battery_plugin::EscBatteryPlugin")
