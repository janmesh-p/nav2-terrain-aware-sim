#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nav2_costmap_2d/layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace terrain_costmap
{

/// Costmap layer that grades terrain from 3D lidar ground returns.
///
/// Stock obstacle layers discard points below a height threshold so the
/// floor is not marked as an obstacle. That same filter hides ramps, bumps
/// and rough ground. This layer keeps those returns, fuses them into a
/// robot-centred elevation grid in the costmap frame, fits a local plane
/// per cell and converts slope and residual roughness into a graded cost.
class TerrainLayer : public nav2_costmap_2d::Layer
{
public:
  TerrainLayer() = default;
  ~TerrainLayer() override = default;

  void onInitialize() override;
  void activate() override;
  void deactivate() override;
  void reset() override;
  bool isClearable() override {return false;}

  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void recenter(double robot_x, double robot_y);
  void recomputeCosts();
  void publishDebug(const rclcpp::Time & stamp);
  bool worldToCell(double wx, double wy, int & i, int & j) const;

  template<typename T>
  T param(const std::string & name, const T & default_value);

  // Parameters
  std::string cloud_topic_;
  std::string robot_base_frame_;
  double resolution_{0.05};
  double size_m_{12.0};
  double recenter_distance_{1.0};
  double min_range_{0.7};
  double max_range_{6.0};
  double band_below_{0.5};
  double band_above_{0.4};
  double fusion_alpha_{0.3};
  double max_cell_spread_{0.12};
  int point_stride_{1};
  double window_m_{0.35};
  double min_valid_fraction_{0.5};
  double slope_free_deg_{5.0};
  double slope_max_cost_deg_{20.0};
  double slope_lethal_deg_{25.0};
  double rough_free_m_{0.005};
  double rough_max_cost_m_{0.025};
  unsigned char max_cost_{252};
  double transform_timeout_{0.1};
  int debug_every_n_{5};

  // Robot-centred grid in the costmap global frame. Cell (0, 0) is the
  // minimum corner, rows run along +y.
  int cells_{0};
  double origin_x_{0.0};
  double origin_y_{0.0};
  bool have_origin_{false};
  std::vector<float> height_;
  std::vector<float> slope_deg_;
  std::vector<float> roughness_m_;
  std::vector<unsigned char> cost_;  // 255 = no estimate
  std::vector<float> scratch_;      // per-scan max z
  std::vector<float> scratch_min_;  // per-scan min z

  std::mutex mutex_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr slope_pub_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr rough_pub_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_pub_;
  rclcpp::Logger logger_{rclcpp::get_logger("terrain_layer")};
  rclcpp::Clock::SharedPtr clock_;
  size_t cloud_count_{0};
};

}  // namespace terrain_costmap
