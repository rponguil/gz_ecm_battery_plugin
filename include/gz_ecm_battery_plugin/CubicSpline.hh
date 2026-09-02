// Natural cubic spline interpolation over user-supplied (x, y) points.
// Self-contained (no external numerical dependency) so this plugin only
// depends on Gazebo Sim itself -- keeps it easy to drop into any project.
//
// Standard method: solve the tridiagonal system for the second derivatives
// at each knot (natural boundary: second derivative = 0 at both ends), then
// evaluate per-segment cubic polynomials. Same family of interpolation as
// scipy.interpolate.interp1d(kind="cubic") used in the original Python ESC
// model this plugin ports (dron_px4_battery_models/esc_battery_model.py) --
// not bit-for-bit identical (different boundary condition convention) but
// equivalent smoothness/fidelity for an OCV(SOC) curve.

#ifndef GZ_ECM_BATTERY_PLUGIN_CUBICSPLINE_HH_
#define GZ_ECM_BATTERY_PLUGIN_CUBICSPLINE_HH_

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gz_ecm_battery_plugin
{

class CubicSpline
{
  public: CubicSpline() = default;

  /// \brief Build the spline from points. `xs` must be strictly increasing
  /// and have the same size as `ys`, with at least 2 points.
  public: void Build(const std::vector<double> &_xs,
                      const std::vector<double> &_ys)
  {
    if (_xs.size() != _ys.size() || _xs.size() < 2)
    {
      throw std::invalid_argument(
          "CubicSpline: xs/ys must have equal size >= 2");
    }
    this->xs = _xs;
    this->ys = _ys;
    const std::size_t n = this->xs.size();
    this->m.assign(n, 0.0);

    if (n == 2)
    {
      // Straight line, no curvature to solve for.
      return;
    }

    // Tridiagonal system for second derivatives (natural spline: m[0] = m[n-1] = 0).
    std::vector<double> h(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i)
      h[i] = this->xs[i + 1] - this->xs[i];

    std::vector<double> alpha(n, 0.0);
    for (std::size_t i = 1; i < n - 1; ++i)
    {
      alpha[i] = 3.0 * ((this->ys[i + 1] - this->ys[i]) / h[i] -
                         (this->ys[i] - this->ys[i - 1]) / h[i - 1]);
    }

    std::vector<double> l(n, 1.0), mu(n, 0.0), z(n, 0.0);
    for (std::size_t i = 1; i < n - 1; ++i)
    {
      l[i] = 2.0 * (this->xs[i + 1] - this->xs[i - 1]) - h[i - 1] * mu[i - 1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    for (std::size_t j = n - 1; j-- > 0;)
    {
      if (j == 0)
        break;
      this->m[j] = z[j] - mu[j] * this->m[j + 1];
    }
  }

  /// \brief Evaluate the spline at `_x`, clamped to the input range (flat
  /// extrapolation, matching the Python model's `bounds_error=False,
  /// fill_value=(first, last)` behaviour).
  public: double Eval(double _x) const
  {
    const std::size_t n = this->xs.size();
    if (_x <= this->xs.front())
      return this->ys.front();
    if (_x >= this->xs.back())
      return this->ys.back();

    // Find segment via binary search (xs is sorted).
    std::size_t i = std::upper_bound(this->xs.begin(), this->xs.end(), _x) -
                     this->xs.begin() - 1;
    i = std::min(i, n - 2);

    const double h = this->xs[i + 1] - this->xs[i];
    const double t = _x - this->xs[i];
    const double a = this->ys[i];
    const double b = (this->ys[i + 1] - this->ys[i]) / h -
                      h * (2.0 * this->m[i] + this->m[i + 1]) / 3.0;
    const double c = this->m[i];
    const double d = (this->m[i + 1] - this->m[i]) / (3.0 * h);

    return a + b * t + c * t * t + d * t * t * t;
  }

  private: std::vector<double> xs;
  private: std::vector<double> ys;
  private: std::vector<double> m;  // second derivatives at knots
};

}  // namespace gz_ecm_battery_plugin

#endif
