#include "terrain_costmap/plane_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace terrain_costmap
{
namespace
{
// Moments accumulated per cell: n, x, y, h, xx, yy, xy, xh, yh, hh.
constexpr int kMoments = 10;
}  // namespace

void analyzeTerrain(
  const std::vector<float> & heights, int width, int height, double resolution,
  int window_radius, double min_valid_fraction,
  std::vector<float> & slope_deg, std::vector<float> & roughness_m)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const size_t cells = static_cast<size_t>(width) * height;
  slope_deg.assign(cells, nan);
  roughness_m.assign(cells, nan);
  if (width <= 0 || height <= 0 || heights.size() != cells) {
    return;
  }

  const int W1 = width + 1;
  std::vector<std::array<double, kMoments>> integral(
    static_cast<size_t>(W1) * (height + 1), std::array<double, kMoments>{});

  for (int j = 0; j < height; ++j) {
    std::array<double, kMoments> row{};
    for (int i = 0; i < width; ++i) {
      const float z = heights[static_cast<size_t>(j) * width + i];
      if (std::isfinite(z)) {
        const double x = i, y = j, h = z;
        row[0] += 1.0;
        row[1] += x;
        row[2] += y;
        row[3] += h;
        row[4] += x * x;
        row[5] += y * y;
        row[6] += x * y;
        row[7] += x * h;
        row[8] += y * h;
        row[9] += h * h;
      }
      auto & out = integral[static_cast<size_t>(j + 1) * W1 + (i + 1)];
      const auto & up = integral[static_cast<size_t>(j) * W1 + (i + 1)];
      for (int m = 0; m < kMoments; ++m) {
        out[m] = up[m] + row[m];
      }
    }
  }

  const int k = 2 * window_radius + 1;
  const double min_count = std::max(3.0, min_valid_fraction * k * k);

  for (int j = 0; j < height; ++j) {
    const int j0 = std::max(0, j - window_radius);
    const int j1 = std::min(height, j + window_radius + 1);
    for (int i = 0; i < width; ++i) {
      const size_t idx = static_cast<size_t>(j) * width + i;
      if (!std::isfinite(heights[idx])) {
        continue;
      }
      const int i0 = std::max(0, i - window_radius);
      const int i1 = std::min(width, i + window_radius + 1);
      const auto & a = integral[static_cast<size_t>(j1) * W1 + i1];
      const auto & b = integral[static_cast<size_t>(j0) * W1 + i1];
      const auto & c = integral[static_cast<size_t>(j1) * W1 + i0];
      const auto & d = integral[static_cast<size_t>(j0) * W1 + i0];
      std::array<double, kMoments> s;
      for (int m = 0; m < kMoments; ++m) {
        s[m] = a[m] - b[m] - c[m] + d[m];
      }
      const double n = s[0];
      if (n < min_count) {
        continue;
      }
      // Central moments keep the 2x2 solve well conditioned.
      const double mx = s[1] / n, my = s[2] / n, mh = s[3] / n;
      const double cxx = s[4] / n - mx * mx;
      const double cyy = s[5] / n - my * my;
      const double cxy = s[6] / n - mx * my;
      const double cxh = s[7] / n - mx * mh;
      const double cyh = s[8] / n - my * mh;
      const double chh = s[9] / n - mh * mh;
      const double det = cxx * cyy - cxy * cxy;
      if (det <= 1e-9) {
        continue;
      }
      const double gx = (cyy * cxh - cxy * cyh) / det;  // m per cell
      const double gy = (cxx * cyh - cxy * cxh) / det;
      const double resid = std::max(0.0, chh - gx * cxh - gy * cyh);
      slope_deg[idx] = static_cast<float>(
        std::atan(std::hypot(gx, gy) / resolution) * 180.0 / M_PI);
      roughness_m[idx] = static_cast<float>(std::sqrt(resid));
    }
  }
}

void markSteps(
  const std::vector<float> & heights, int width, int height,
  int radius_cells, double max_step, std::vector<unsigned char> & steps)
{
  const size_t cells = static_cast<size_t>(width) * height;
  steps.assign(cells, 0);
  if (heights.size() != cells || radius_cells < 1) {
    return;
  }
  const int r2 = radius_cells * radius_cells;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const float hc = heights[static_cast<size_t>(j) * width + i];
      if (!std::isfinite(hc)) {
        continue;
      }
      bool step = false;
      for (int dj = -radius_cells; dj <= radius_cells && !step; ++dj) {
        const int nj = j + dj;
        if (nj < 0 || nj >= height) {
          continue;
        }
        for (int di = -radius_cells; di <= radius_cells; ++di) {
          const int ni = i + di;
          if ((di == 0 && dj == 0) || ni < 0 || ni >= width || di * di + dj * dj > r2) {
            continue;
          }
          const float hn = heights[static_cast<size_t>(nj) * width + ni];
          if (std::isfinite(hn) && std::fabs(hn - hc) > max_step) {
            step = true;
            break;
          }
        }
      }
      steps[static_cast<size_t>(j) * width + i] = step ? 1 : 0;
    }
  }
}

}  // namespace terrain_costmap
