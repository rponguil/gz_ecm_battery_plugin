// Validates the SAME EscModel class the compiled Gazebo plugin uses
// (gz_ecm_battery_plugin::EscModel, include/gz_ecm_battery_plugin/EscModel.hh)
// against the real measured Panasonic 18650PF data, using the exact
// parameters and OCV curve already fitted by validate_against_panasonic18650pf.py.
//
// Purpose: validate_against_panasonic18650pf.py proves the MODEL (and its
// Python reference implementation) matches a real cell. This program proves
// the C++ PORT used inside the actual plugin produces the same numbers as
// that Python reference, given identical inputs -- i.e. that translating
// the math to C++ did not introduce a discrepancy. No Gazebo dependency:
// EscModel.hh is plain C++, this only needs a compiler.
//
// Build: g++ -std=c++17 -I../include validate_cpp_plugin.cc -o validate_cpp_plugin
// Run:   ./validate_cpp_plugin

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gz_ecm_battery_plugin/EscModel.hh"

using gz_ecm_battery_plugin::CubicSpline;
using gz_ecm_battery_plugin::EscModel;
using gz_ecm_battery_plugin::EscModelParams;

namespace
{

std::vector<std::vector<double>> ReadCsv(const std::string &_path,
    std::size_t _numCols, bool _skipHeader = true)
{
  std::ifstream file(_path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open " + _path);

  std::vector<std::vector<double>> cols(_numCols);
  std::string line;
  if (_skipHeader)
    std::getline(file, line);

  while (std::getline(file, line))
  {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string cell;
    for (std::size_t c = 0; c < _numCols && std::getline(ss, cell, ','); ++c)
      cols[c].push_back(std::stod(cell));
  }
  return cols;
}

}  // namespace

int main()
{
  const std::string dir = "results/";

  // fitted_params.csv: param,value (2 rows header + 4 data rows)
  std::ifstream pf(dir + "fitted_params.csv");
  if (!pf.is_open())
  {
    std::cerr << "Run validate_against_panasonic18650pf.py first "
              << "(need " << dir << "fitted_params.csv).\n";
    return 1;
  }
  std::string line;
  std::getline(pf, line);  // header
  EscModelParams params;
  while (std::getline(pf, line))
  {
    auto commaPos = line.find(',');
    std::string key = line.substr(0, commaPos);
    double value = std::stod(line.substr(commaPos + 1));
    if (key == "capacity_ah") params.capacityAh = value;
    else if (key == "r0_ohm") params.r0 = value;
    else if (key == "r1_ohm") params.r1 = value;
    else if (key == "c1_farad") params.c1 = value;
  }
  params.hystM = 0.0;
  params.hystM0 = 0.0;
  params.coulombicEff = 1.0;

  auto ocvCols = ReadCsv(dir + "fitted_ocv_curve.csv", 2);
  CubicSpline ocv;
  ocv.Build(ocvCols[0], ocvCols[1]);

  std::cout << "Parametros: capacity=" << params.capacityAh
            << " Ah, R0=" << params.r0 << " Ohm (fallback), R1=" << params.r1
            << " Ohm (fallback), C1=" << params.c1 << " F, "
            << ocvCols[0].size() << " puntos OCV\n";

  EscModel model;
  model.Configure(params, ocv);

  std::ifstream r0f(dir + "fitted_r0_curve.csv");
  if (r0f.good())
  {
    r0f.close();
    auto r0Cols = ReadCsv(dir + "fitted_r0_curve.csv", 2);
    CubicSpline r0Spline;
    r0Spline.Build(r0Cols[0], r0Cols[1]);
    model.SetR0Curve(r0Spline);
    std::cout << "Usando R0(SOC): " << r0Cols[0].size() << " puntos\n";
  }
  std::ifstream r1f(dir + "fitted_r1_curve.csv");
  if (r1f.good())
  {
    r1f.close();
    auto r1Cols = ReadCsv(dir + "fitted_r1_curve.csv", 2);
    CubicSpline r1Spline;
    r1Spline.Build(r1Cols[0], r1Cols[1]);
    model.SetR1Curve(r1Spline);
    std::cout << "Usando R1(SOC): " << r1Cols[0].size() << " puntos\n";
  }

  // validation_vs_panasonic18650pf.csv:
  // t_s,current_a,v_measured,v_simulated(python),error_mv,soc_reference
  auto data = ReadCsv(dir + "validation_vs_panasonic18650pf.csv", 6);
  const auto &t = data[0];
  const auto &current = data[1];
  const auto &vMeasured = data[2];
  const auto &vSimPython = data[3];
  const auto &socRef = data[5];

  model.Reset(1.0);

  std::vector<double> vSimCpp(t.size());
  double sumSqErrVsMeasured = 0.0;
  double sumSqErrVsPython = 0.0;
  double maxErrVsMeasured = 0.0;

  for (std::size_t k = 0; k < t.size(); ++k)
  {
    double dt = (k == 0) ? (t.size() > 1 ? t[1] - t[0] : 0.1)
                         : (t[k] - t[k - 1]);
    if (dt <= 0.0)
      dt = 0.1;

    model.z = std::max(0.0, std::min(1.0, socRef[k]));
    vSimCpp[k] = model.Step(current[k], dt);

    double errMeasured = (vSimCpp[k] - vMeasured[k]) * 1000.0;
    double errPython = (vSimCpp[k] - vSimPython[k]) * 1000.0;
    sumSqErrVsMeasured += errMeasured * errMeasured;
    sumSqErrVsPython += errPython * errPython;
    maxErrVsMeasured = std::max(maxErrVsMeasured, std::abs(errMeasured));
  }

  double n = static_cast<double>(t.size());
  double rmseVsMeasured = std::sqrt(sumSqErrVsMeasured / n);
  double rmseVsPython = std::sqrt(sumSqErrVsPython / n);

  std::cout << "\nPlugin C++ (EscModel) vs. celda real medida:\n";
  std::cout << "  RMSE = " << rmseVsMeasured << " mV\n";
  std::cout << "  error max = " << maxErrVsMeasured << " mV\n";
  std::cout << "\nPlugin C++ (EscModel) vs. referencia Python "
            << "(esc_battery_model.py) -- fidelidad del port:\n";
  std::cout << "  RMSE = " << rmseVsPython << " mV "
            << "(deberia ser ~0 si el port es numericamente fiel)\n";

  std::ofstream out(dir + "validation_cpp_plugin.csv");
  out << "t_s,current_a,v_measured,v_sim_python,v_sim_cpp\n";
  for (std::size_t k = 0; k < t.size(); ++k)
  {
    out << t[k] << "," << current[k] << "," << vMeasured[k] << ","
        << vSimPython[k] << "," << vSimCpp[k] << "\n";
  }
  std::cout << "\nCSV: " << dir << "validation_cpp_plugin.csv\n";

  return 0;
}
