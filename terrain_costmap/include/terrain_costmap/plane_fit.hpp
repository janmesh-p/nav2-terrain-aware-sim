#pragma once

#include <vector>

namespace terrain_costmap
{

/// Windowed least-squares plane fit over a height grid.
///
/// For every cell with a valid height, fits z = a + b*x + c*y to all valid
/// cells in a (2r+1) x (2r+1) window. Slope comes from the plane gradient,
/// roughness is the RMS residual about the plane, so a steady incline does
/// not register as rough. Sums come from integral images, so cost is
/// O(width * height) regardless of window size.
///
/// heights: row-major, width * height, NaN marks unknown cells.
/// Outputs are NaN where the cell is unknown or the window has too few
/// valid cells for a stable fit.
void analyzeTerrain(
  const std::vector<float> & heights, int width, int height, double resolution,
  int window_radius, double min_valid_fraction,
  std::vector<float> & slope_deg, std::vector<float> & roughness_m);

}  // namespace terrain_costmap
