// Copyright (c) 2026 Ronald Ponguillo-Intriago
// Licensed under the MIT License (see LICENSE at the root of this repo).
//
// Unit tests for CubicSpline and EscModel -- pure C++, no Gazebo dependency,
// so these run fast and standalone (no simulator needed). This is the
// invariant-level testing (JOSS requires it, SoftwareX values it); the
// against-real-cell-data validation lives in validation_data/, which is a
// heavier, separate kind of check (external ground truth, not unit tests).

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "gz_ecm_battery_plugin/CubicSpline.hh"
#include "gz_ecm_battery_plugin/EscModel.hh"

using namespace gz_ecm_battery_plugin;

namespace
{
CubicSpline MakeNmcOcv()
{
  CubicSpline ocv;
  ocv.Build({0.0, 0.5, 1.0}, {3.0, 3.72, 4.2});
  return ocv;
}
}  // namespace

// ------------------------------- CubicSpline -------------------------------

TEST(CubicSpline, PassesThroughDataPoints)
{
  CubicSpline s;
  s.Build({0.0, 0.5, 1.0}, {3.0, 3.7, 4.2});
  EXPECT_NEAR(s.Eval(0.0), 3.0, 1e-9);
  EXPECT_NEAR(s.Eval(0.5), 3.7, 1e-9);
  EXPECT_NEAR(s.Eval(1.0), 4.2, 1e-9);
}

TEST(CubicSpline, FlatExtrapolationOutsideRange)
{
  CubicSpline s;
  s.Build({0.0, 1.0}, {3.0, 4.2});
  EXPECT_DOUBLE_EQ(s.Eval(-1.0), 3.0);
  EXPECT_DOUBLE_EQ(s.Eval(2.0), 4.2);
}

TEST(CubicSpline, TwoPointsIsAStraightLine)
{
  CubicSpline s;
  s.Build({0.0, 1.0}, {0.0, 10.0});
  EXPECT_NEAR(s.Eval(0.5), 5.0, 1e-9);
  EXPECT_NEAR(s.Eval(0.25), 2.5, 1e-9);
}

TEST(CubicSpline, ThrowsOnTooFewPoints)
{
  CubicSpline s;
  EXPECT_THROW(s.Build({1.0}, {2.0}), std::invalid_argument);
}

TEST(CubicSpline, ThrowsOnMismatchedSizes)
{
  CubicSpline s;
  EXPECT_THROW(s.Build({0.0, 1.0}, {2.0}), std::invalid_argument);
}

// --------------------------------- EscModel ---------------------------------

TEST(EscModel, ZeroCurrentGivesOcvAtInitialSoc)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(1.0);
  // No current ever applied yet: hysteresis/RC states are still at their
  // initial zero, so voltage should equal the bare OCV at z0.
  EXPECT_NEAR(m.Step(0.0, 1.0), 4.2, 1e-6);
}

TEST(EscModel, SocNeverLeavesZeroOneUnderSustainedDischarge)
{
  EscModelParams p;
  p.capacityAh = 0.001;  // tiny on purpose: depletes in a couple of steps
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(1.0);
  for (int k = 0; k < 1000; ++k)
  {
    m.Step(5.0, 1.0);
    ASSERT_GE(m.z, 0.0);
    ASSERT_LE(m.z, 1.0);
  }
  EXPECT_NEAR(m.z, 0.0, 1e-9);  // should have hit the floor and stayed there
}

TEST(EscModel, SocNeverExceedsOneUnderSustainedCharge)
{
  EscModelParams p;
  p.capacityAh = 0.001;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(0.0);
  for (int k = 0; k < 1000; ++k)
  {
    m.Step(-5.0, 1.0);  // negative current = charging, per this project's convention
    ASSERT_GE(m.z, 0.0);
    ASSERT_LE(m.z, 1.0);
  }
  EXPECT_NEAR(m.z, 1.0, 1e-9);
}

TEST(EscModel, VoltageNeverNegative)
{
  EscModelParams p;
  p.capacityAh = 0.0001;
  p.r0 = 5.0;   // deliberately huge, would drive voltage very negative if unclamped
  p.r1 = 0.0;
  p.c1 = 1.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(0.0);
  EXPECT_GE(m.Step(1000.0, 1.0), 0.0);
}

TEST(EscModel, DischargeDecreasesSoc)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(0.5);
  m.Step(1.0, 60.0);
  EXPECT_LT(m.z, 0.5);
}

TEST(EscModel, ChargeIncreasesSoc)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(0.5);
  m.Step(-1.0, 60.0);
  EXPECT_GT(m.z, 0.5);
}

TEST(EscModel, R1ZeroDisablesFirstRcBranchWithoutNaN)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.0;  // would divide by zero in the RC time-constant if unguarded
  p.c1 = 4000.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(1.0);
  double v = m.Step(2.0, 1.0);
  EXPECT_FALSE(std::isnan(v));
  EXPECT_FALSE(std::isinf(v));
}

TEST(EscModel, R2ZeroDisablesSecondRcBranchWithoutNaN)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;
  p.r2 = 0.0;  // second branch disabled -- default state, must stay stable
  p.c2 = 100.0;
  EscModel m;
  m.Configure(p, MakeNmcOcv());
  m.Reset(1.0);
  double v = m.Step(2.0, 1.0);
  EXPECT_FALSE(std::isnan(v));
  EXPECT_FALSE(std::isinf(v));
}

TEST(EscModel, SocDependentR0CurveOverridesConstantR0)
{
  EscModelParams p;
  p.capacityAh = 2.5;
  p.r0 = 0.05;
  p.r1 = 0.02;
  p.c1 = 4000.0;

  EscModel withCurve;
  withCurve.Configure(p, MakeNmcOcv());
  CubicSpline r0Curve;
  r0Curve.Build({0.0, 1.0}, {0.1, 0.01});  // much smaller than the constant 0.05 at SOC=1
  withCurve.SetR0Curve(r0Curve);
  withCurve.Reset(1.0);

  EscModel constant;
  constant.Configure(p, MakeNmcOcv());  // no curve set -> falls back to p.r0
  constant.Reset(1.0);

  // Same discharge current, same OCV/SOC: only R0 differs, so the curve
  // variant (smaller R0 at SOC=1) must show a smaller ohmic drop, i.e.
  // higher terminal voltage.
  EXPECT_GT(withCurve.Step(1.0, 1.0), constant.Step(1.0, 1.0));
}
