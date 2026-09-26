#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "terrain_costmap/plane_fit.hpp"

using terrain_costmap::analyzeTerrain;

namespace
{
constexpr double kRes = 0.05;

std::vector<float> plane(int w, int h, double grade_deg, double yaw_deg)
{
  const double g = std::tan(grade_deg * M_PI / 180.0);
  const double c = std::cos(yaw_deg * M_PI / 180.0), s = std::sin(yaw_deg * M_PI / 180.0);
  std::vector<float> z(static_cast<size_t>(w) * h);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      z[j * w + i] = static_cast<float>(g * (c * i + s * j) * kRes + 0.3);
    }
  }
  return z;
}
}  // namespace

TEST(PlaneFit, RecoversGradeInAnyDirection)
{
  for (double yaw : {0.0, 37.0, 90.0, 215.0}) {
    auto z = plane(60, 50, 12.0, yaw);
    std::vector<float> slope, rough;
    analyzeTerrain(z, 60, 50, kRes, 3, 0.5, slope, rough);
    for (int j = 3; j < 47; ++j) {
      for (int i = 3; i < 57; ++i) {
        ASSERT_NEAR(slope[j * 60 + i], 12.0, 0.01) << "yaw " << yaw;
        ASSERT_NEAR(rough[j * 60 + i], 0.0, 1e-4);
      }
    }
  }
}

TEST(PlaneFit, UnknownCellsStayUnknownAndDoNotBias)
{
  auto z = plane(40, 40, 8.0, 20.0);
  std::mt19937 rng(3);
  std::bernoulli_distribution drop(0.3);
  for (auto & v : z) {
    if (drop(rng)) {
      v = std::numeric_limits<float>::quiet_NaN();
    }
  }
  std::vector<float> slope, rough;
  analyzeTerrain(z, 40, 40, kRes, 3, 0.3, slope, rough);
  int checked = 0;
  for (size_t k = 0; k < z.size(); ++k) {
    if (std::isnan(z[k])) {
      EXPECT_TRUE(std::isnan(slope[k]));
    } else if (std::isfinite(slope[k])) {
      EXPECT_NEAR(slope[k], 8.0, 0.01);
      ++checked;
    }
  }
  EXPECT_GT(checked, 800);
}

TEST(PlaneFit, RoughnessTracksNoiseNotSlope)
{
  auto z = plane(80, 80, 10.0, 0.0);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.01f);
  for (auto & v : z) {
    v += noise(rng);
  }
  std::vector<float> slope, rough;
  analyzeTerrain(z, 80, 80, kRes, 4, 0.5, slope, rough);
  double sum = 0;
  int n = 0;
  for (int j = 10; j < 70; ++j) {
    for (int i = 10; i < 70; ++i) {
      sum += rough[j * 80 + i];
      ++n;
    }
  }
  // Residual RMS of 81 samples about a 3-parameter fit is slightly below sigma.
  EXPECT_NEAR(sum / n, 0.01, 0.0015);
}

TEST(PlaneFit, SparseWindowIsRejected)
{
  std::vector<float> z(25, std::numeric_limits<float>::quiet_NaN());
  z[12] = 0.1f;
  std::vector<float> slope, rough;
  analyzeTerrain(z, 5, 5, kRes, 2, 0.5, slope, rough);
  EXPECT_TRUE(std::isnan(slope[12]));
}
