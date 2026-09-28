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

/// Flag cells that sit on a height step a wheel cannot climb.
///
/// A known cell is flagged when any known cell within radius_cells differs
/// in height by more than max_step metres. This catches a ramp's side from
/// both the floor and the top, even when the vertical face itself has no
/// returns, and it catches drop-offs that obstacle layers never see.
///
/// heights: row-major, NaN unknown. steps: output, same size, 1 = step.
void markSteps(
  const std::vector<float> & heights, int width, int height,
  int radius_cells, double max_step, std::vector<unsigned char> & steps);

}  // namespace terrain_costmap
