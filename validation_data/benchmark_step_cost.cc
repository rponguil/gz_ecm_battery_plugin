// Cost of one EscModel::Step call, and the model's memory footprint.
//
// A simulator plugin runs inside the physics loop, so its per-step cost is a
// property users need before adopting it. Reviewers of the manuscript asked
// for this figure and the omission was fair: accuracy was reported, cost was
// not.
//
// What is measured is the simulator-independent core -- the same translation
// unit the plugin's shared library builds and the validation harness links.
// The Gazebo integration layer around it (SDF parsing at load time, component
// write, telemetry publication) is not included; parsing happens once, and the
// per-step part of the wrapper is a component write and an optional message
// publish whose cost belongs to Gazebo rather than to this model.
//
// Build:
//   g++ -O2 -std=c++17 -I../include benchmark_step_cost.cc -o benchmark_step_cost
// Run:
//   ./benchmark_step_cost            (prints JSON on the last line)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gz_ecm_battery_plugin/CubicSpline.hh"
#include "gz_ecm_battery_plugin/EscModel.hh"

using gz_ecm_battery_plugin::CubicSpline;
using gz_ecm_battery_plugin::EscModel;
using gz_ecm_battery_plugin::EscModelParams;

namespace {

// The OCV curve used in the reported validation: 19 points over [0, 1].
CubicSpline MakeOcv()
{
  std::vector<double> xs, ys;
  for (int k = 0; k < 19; ++k)
  {
    const double z = static_cast<double>(k) / 18.0;
    xs.push_back(z);
    ys.push_back(3.0 + 1.2 * z - 0.35 * (1.0 - z) * (1.0 - z));
  }
  CubicSpline s;
  s.Build(xs, ys);
  return s;
}

double Bench(bool twoBranches, bool socCurves, std::size_t n)
{
  EscModelParams p;
  p.capacityAh = 2.773;
  p.r0 = 0.0254;
  p.r1 = 0.0065;
  p.c1 = 946.0;
  p.hystM = 0.02;
  p.hystM0 = 0.005;
  if (twoBranches)
  {
    p.r2 = 0.004;
    p.c2 = 100.0;
  }

  EscModel m;
  m.Configure(p, MakeOcv());
  if (socCurves)
  {
    std::vector<double> xs, r0s, r1s;
    for (int k = 0; k < 19; ++k)
    {
      const double z = static_cast<double>(k) / 18.0;
      xs.push_back(z);
      r0s.push_back(0.02 + 0.02 * (1.0 - z));
      r1s.push_back(0.005 + 0.01 * (1.0 - z));
    }
    CubicSpline r0c, r1c;
    r0c.Build(xs, r0s);
    r1c.Build(xs, r1s);
    m.SetR0Curve(r0c);
    m.SetR1Curve(r1c);
  }

  // A varying current, so the branch updates and the spline lookups do not
  // land on the same cached interval every step.
  double sink = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t k = 0; k < n; ++k)
  {
    const double i = 2.0 + 6.0 * std::sin(0.001 * static_cast<double>(k));
    sink += m.Step(i, 0.001);
    if ((k % 100000) == 0)
      m.Reset(1.0);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ns =
      std::chrono::duration<double, std::nano>(t1 - t0).count() /
      static_cast<double>(n);
  if (sink == 12345.6789)  // never true; keeps the loop from being elided
    std::printf(" ");
  return ns;
}

}  // namespace

int main()
{
  const std::size_t n = 20000000;
  const double base = Bench(false, false, n);
  const double two = Bench(true, false, n);
  const double curves = Bench(true, true, n);

  std::printf("EscModel::Step, mean over %zu calls\n", n);
  std::printf("  1 RC branch, scalar R0/R1        %7.2f ns\n", base);
  std::printf("  2 RC branches, scalar R0/R1      %7.2f ns\n", two);
  std::printf("  2 RC branches, R0(z)/R1(z)       %7.2f ns\n", curves);
  std::printf("  sizeof(EscModel)                 %7zu bytes\n",
              sizeof(EscModel));
  std::printf("  sizeof(CubicSpline)              %7zu bytes\n",
              sizeof(CubicSpline));
  // Last line is machine readable for the manifest.
  std::printf("JSON {\"step_ns_1rc\": %.4f, \"step_ns_2rc\": %.4f, "
              "\"step_ns_curves\": %.4f, \"sizeof_model\": %zu}\n",
              base, two, curves, sizeof(EscModel));
  return 0;
}
