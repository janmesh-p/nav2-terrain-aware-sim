#include "terrain_costmap/terrain_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2/exceptions.h"
#include "tf2/time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/buffer_interface.h"
#include "terrain_costmap/plane_fit.hpp"

namespace terrain_costmap
{
namespace
{
constexpr unsigned char kNoEstimate = 255;
const float kNaN = std::numeric_limits<float>::quiet_NaN();

tf2::Transform toTf(const geometry_msgs::msg::Transform & t)
{
  return tf2::Transform(
    tf2::Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w),
    tf2::Vector3(t.translation.x, t.translation.y, t.translation.z));
}

// Angle between a frame's z axis and world vertical, degrees.
double tiltDeg(double qx, double qy, double qz, double qw)
{
  const double n = qx * qx + qy * qy + qz * qz + qw * qw;
  if (n < 1e-9) {
    return 0.0;
  }
  const double c = 1.0 - 2.0 * (qx * qx + qy * qy) / n;
  return std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / M_PI;
}

double ramp01(double v, double lo, double hi)
{
  if (hi <= lo) {
    return v >= hi ? 1.0 : 0.0;
  }
  return std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
}
}  // namespace

template<typename T>
T TerrainLayer::param(const std::string & name, const T & default_value)
{
  declareParameter(name, rclcpp::ParameterValue(default_value));
  auto node = node_.lock();
  T value = default_value;
  node->get_parameter(name_ + "." + name, value);
  return value;
}

void TerrainLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("terrain_layer: failed to lock node");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  enabled_ = param<bool>("enabled", true);
  cloud_topic_ = param<std::string>("cloud_topic", "/front_3d_lidar/lidar_points");
  robot_base_frame_ = param<std::string>("robot_base_frame", "base_link");
  resolution_ = param<double>("resolution", 0.10);
  size_m_ = param<double>("size", 12.0);
  recenter_distance_ = param<double>("recenter_distance", 1.0);
  min_range_ = param<double>("min_range", 0.7);
  max_range_ = param<double>("max_range", 6.0);
  band_below_ = param<double>("band_below", 0.5);
  band_above_ = param<double>("band_above", 0.35);
  fusion_alpha_ = param<double>("fusion_alpha", 0.3);
  max_cell_spread_ = param<double>("max_cell_spread", 0.12);
  max_step_ = param<double>("max_step", 0.10);
  step_radius_ = param<double>("step_radius", 0.2);
  vertical_confirm_ = std::clamp(param<int>("vertical_confirm", 2), 1, 3);
  imu_topic_ = param<std::string>("imu_topic", "/chassis/imu");
  tilt_gate_deg_ = param<double>("tilt_gate_deg", 3.0);
  tilt_trust_tol_deg_ = param<double>("tilt_trust_tol_deg", 1.0);
  tilt_hold_s_ = param<double>("tilt_hold_s", 0.5);
  point_stride_ = std::max(1, param<int>("point_stride", 1));
  window_m_ = param<double>("window", 0.5);
  min_valid_fraction_ = param<double>("min_valid_fraction", 0.25);
  slope_free_deg_ = param<double>("slope_free_deg", 5.0);
  slope_max_cost_deg_ = param<double>("slope_max_cost_deg", 20.0);
  slope_lethal_deg_ = param<double>("slope_lethal_deg", 25.0);
  rough_free_m_ = param<double>("roughness_free", 0.005);
  rough_max_cost_m_ = param<double>("roughness_max_cost", 0.025);
  max_cost_ = static_cast<unsigned char>(
    std::clamp(param<int>("max_cost", 252), 1, 252));
  transform_timeout_ = param<double>("transform_timeout", 0.1);
  debug_every_n_ = std::max(1, param<int>("debug_every_n", 5));

  cells_ = std::max(8, static_cast<int>(std::round(size_m_ / resolution_)));
  const size_t n = static_cast<size_t>(cells_) * cells_;
  height_.assign(n, kNaN);
  slope_deg_.assign(n, kNaN);
  roughness_m_.assign(n, kNaN);
  cost_.assign(n, kNoEstimate);
  scratch_.assign(n, kNaN);
  scratch_min_.assign(n, kNaN);
  vertical_hits_.assign(n, 0);
  steps_.assign(n, 0);

  cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&TerrainLayer::cloudCallback, this, std::placeholders::_1));

  if (!imu_topic_.empty()) {
    imu_sub_ = node->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&TerrainLayer::imuCallback, this, std::placeholders::_1));
  }

  const std::string prefix = "~/" + name_ + "/";
  auto latched = rclcpp::QoS(1).transient_local();
  slope_pub_ = node->create_publisher<nav_msgs::msg::OccupancyGrid>(prefix + "slope", latched);
  rough_pub_ = node->create_publisher<nav_msgs::msg::OccupancyGrid>(prefix + "roughness", latched);
  cost_pub_ = node->create_publisher<nav_msgs::msg::OccupancyGrid>(prefix + "cost", latched);

  current_ = true;
  RCLCPP_INFO(
    logger_, "%s: %dx%d cells at %.3f m, cloud '%s', frame '%s'",
    name_.c_str(), cells_, cells_, resolution_, cloud_topic_.c_str(),
    layered_costmap_->getGlobalFrameID().c_str());
}

void TerrainLayer::activate()
{
  slope_pub_->on_activate();
  rough_pub_->on_activate();
  cost_pub_->on_activate();
}

void TerrainLayer::deactivate()
{
  slope_pub_->on_deactivate();
  rough_pub_->on_deactivate();
  cost_pub_->on_deactivate();
}

void TerrainLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::fill(height_.begin(), height_.end(), kNaN);
  std::fill(slope_deg_.begin(), slope_deg_.end(), kNaN);
  std::fill(roughness_m_.begin(), roughness_m_.end(), kNaN);
  std::fill(cost_.begin(), cost_.end(), kNoEstimate);
  std::fill(vertical_hits_.begin(), vertical_hits_.end(), 0);
  std::fill(steps_.begin(), steps_.end(), 0);
  have_origin_ = false;
  current_ = true;
}

bool TerrainLayer::worldToCell(double wx, double wy, int & i, int & j) const
{
  i = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
  j = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
  return i >= 0 && j >= 0 && i < cells_ && j < cells_;
}

template<typename T>
void TerrainLayer::shiftGrid(std::vector<T> & grid, int di, int dj, T fill)
{
  std::vector<T> shifted(grid.size(), fill);
  for (int j = 0; j < cells_; ++j) {
    const int oj = j + dj;
    if (oj < 0 || oj >= cells_) {
      continue;
    }
    for (int i = 0; i < cells_; ++i) {
      const int oi = i + di;
      if (oi >= 0 && oi < cells_) {
        shifted[static_cast<size_t>(j) * cells_ + i] = grid[static_cast<size_t>(oj) * cells_ + oi];
      }
    }
  }
  grid.swap(shifted);
}

void TerrainLayer::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  double tilt;
  const auto & q = msg->orientation;
  const bool has_orientation = msg->orientation_covariance[0] >= 0.0 &&
    (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) > 0.5;
  if (has_orientation) {
    tilt = tiltDeg(q.x, q.y, q.z, q.w);
  } else {
    // Fall back to the gravity direction in the accelerometer.
    const auto & a = msg->linear_acceleration;
    const double n = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (n < 1e-3) {
      return;
    }
    tilt = std::acos(std::clamp(a.z / n, -1.0, 1.0)) * 180.0 / M_PI;
  }
  std::lock_guard<std::mutex> lock(imu_mutex_);
  imu_tilt_deg_ = tilt;
  imu_stamp_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);
  have_imu_ = true;
}

void TerrainLayer::recenter(double robot_x, double robot_y)
{
  const double half = 0.5 * cells_ * resolution_;
  const double new_x = std::floor((robot_x - half) / resolution_) * resolution_;
  const double new_y = std::floor((robot_y - half) / resolution_) * resolution_;
  if (!have_origin_) {
    origin_x_ = new_x;
    origin_y_ = new_y;
    have_origin_ = true;
    return;
  }
  const double cx = origin_x_ + half, cy = origin_y_ + half;
  if (std::hypot(robot_x - cx, robot_y - cy) < recenter_distance_) {
    return;
  }
  const int di = static_cast<int>(std::lround((new_x - origin_x_) / resolution_));
  const int dj = static_cast<int>(std::lround((new_y - origin_y_) / resolution_));
  // Shift stored state so world-fixed cells keep their values.
  shiftGrid(height_, di, dj, kNaN);
  shiftGrid(vertical_hits_, di, dj, static_cast<unsigned char>(0));
  origin_x_ += di * resolution_;
  origin_y_ += dj * resolution_;
}

void TerrainLayer::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!enabled_) {
    return;
  }
  const std::string global_frame = layered_costmap_->getGlobalFrameID();
  tf2::Transform cloud_to_global, base_to_global;
  try {
    const auto stamp = tf2_ros::fromMsg(msg->header.stamp);
    const auto timeout = tf2::durationFromSec(transform_timeout_);
    cloud_to_global = toTf(
      tf_->lookupTransform(global_frame, msg->header.frame_id, stamp, timeout).transform);
    base_to_global = toTf(
      tf_->lookupTransform(global_frame, robot_base_frame_, stamp, timeout).transform);
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "%s: tf failed: %s", name_.c_str(), e.what());
    return;
  }

  // Tilt gating. If the robot is tilted and TF does not carry that tilt,
  // projecting this scan would put ground points at wrong heights and paint
  // phantom slopes and steps. Skip it, and a short while after.
  {
    const rclcpp::Time stamp(msg->header.stamp, RCL_ROS_TIME);
    const tf2::Quaternion qb = base_to_global.getRotation();
    const double tf_tilt = tiltDeg(qb.x(), qb.y(), qb.z(), qb.w());
    double imu_tilt = 0.0;
    bool imu_fresh = false;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_fresh = have_imu_ && std::fabs((stamp - imu_stamp_).seconds()) < 0.5;
      imu_tilt = imu_tilt_deg_;
    }
    if (imu_fresh && imu_tilt > tilt_gate_deg_) {
      const bool tf_untrusted = std::fabs(imu_tilt - tf_tilt) > tilt_trust_tol_deg_;
      if (tf_untrusted && !tf_planar_reported_) {
        RCLCPP_WARN(logger_, "%s: IMU tilt %.1f deg but TF tilt %.1f deg, TF is planar; "
          "gating tilted scans", name_.c_str(), imu_tilt, tf_tilt);
        tf_planar_reported_ = true;
      } else if (!tf_untrusted && !tf_tilted_reported_) {
        RCLCPP_INFO(logger_, "%s: TF carries body tilt (%.1f deg); no gating needed",
          name_.c_str(), tf_tilt);
        tf_tilted_reported_ = true;
      }
      if (tf_untrusted) {
        last_untrusted_ = stamp;
      }
    }
    if (last_untrusted_.nanoseconds() > 0 &&
      (stamp - last_untrusted_).seconds() >= 0.0 &&
      (stamp - last_untrusted_).seconds() < tilt_hold_s_)
    {
      if (++gated_clouds_ % 50 == 1) {
        RCLCPP_INFO(logger_, "%s: %zu tilted scans skipped so far", name_.c_str(), gated_clouds_);
      }
      return;
    }
  }

  const tf2::Vector3 robot = base_to_global.getOrigin();
  const double z_lo = robot.z() - band_below_;
  const double z_hi = robot.z() + band_above_;
  const double min_r2 = min_range_ * min_range_;
  const double max_r2 = max_range_ * max_range_;

  std::lock_guard<std::mutex> lock(mutex_);
  recenter(robot.x(), robot.y());
  std::fill(scratch_.begin(), scratch_.end(), kNaN);
  std::fill(scratch_min_.begin(), scratch_min_.end(), kNaN);

  sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
  const size_t total = static_cast<size_t>(msg->width) * msg->height;
  for (size_t k = 0; k < total; k += point_stride_) {
    const float x = *it_x, y = *it_y, z = *it_z;
    it_x += point_stride_;
    it_y += point_stride_;
    it_z += point_stride_;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    const tf2::Vector3 p = cloud_to_global * tf2::Vector3(x, y, z);
    if (p.z() < z_lo || p.z() > z_hi) {
      continue;  // below the floor band or tall enough to be a real obstacle
    }
    const double dx = p.x() - robot.x(), dy = p.y() - robot.y();
    const double r2 = dx * dx + dy * dy;
    if (r2 < min_r2 || r2 > max_r2) {
      continue;  // own chassis, or too sparse to trust
    }
    int i, j;
    if (!worldToCell(p.x(), p.y(), i, j)) {
      continue;
    }
    const size_t c = static_cast<size_t>(j) * cells_ + i;
    const float pz = static_cast<float>(p.z());
    // Highest return in a cell approximates the surface a wheel meets.
    if (std::isnan(scratch_[c]) || pz > scratch_[c]) {
      scratch_[c] = pz;
    }
    if (std::isnan(scratch_min_[c]) || pz < scratch_min_[c]) {
      scratch_min_[c] = pz;
    }
  }

  size_t rejected = 0;
  for (size_t c = 0; c < scratch_.size(); ++c) {
    if (std::isnan(scratch_[c])) {
      continue;
    }
    // A tall spread of returns inside one cell is a vertical face (wall,
    // pallet side, ramp edge), not a surface to fit a plane through.
    if (scratch_[c] - scratch_min_[c] > max_cell_spread_) {
      // Keep it out of the plane fit, but remember it: repeated evidence
      // of a vertical face marks the cell lethal in recomputeCosts().
      height_[c] = kNaN;
      if (vertical_hits_[c] < 3) {
        ++vertical_hits_[c];
      }
      ++rejected;
      continue;
    }
    if (vertical_hits_[c] > 0) {
      --vertical_hits_[c];  // flat observation weakens earlier evidence
    }
    float & h = height_[c];
    h = std::isnan(h) ? scratch_[c] :
      static_cast<float>((1.0 - fusion_alpha_) * h + fusion_alpha_ * scratch_[c]);
  }

  recomputeCosts();
  current_ = true;
  RCLCPP_DEBUG(logger_, "%s: %zu vertical cells rejected", name_.c_str(), rejected);
  if (++cloud_count_ % debug_every_n_ == 0) {
    publishDebug(msg->header.stamp);
  }
}

void TerrainLayer::recomputeCosts()
{
  const int radius = std::max(1, static_cast<int>(std::round(0.5 * window_m_ / resolution_)));
  analyzeTerrain(
    height_, cells_, cells_, resolution_, radius, min_valid_fraction_,
    slope_deg_, roughness_m_);

  const int step_cells = std::max(1, static_cast<int>(std::round(step_radius_ / resolution_)));
  markSteps(height_, cells_, cells_, step_cells, max_step_, steps_);

  for (size_t c = 0; c < cost_.size(); ++c) {
    if (steps_[c] || vertical_hits_[c] >= vertical_confirm_) {
      cost_[c] = nav2_costmap_2d::LETHAL_OBSTACLE;
      continue;
    }
    const float s = slope_deg_[c], r = roughness_m_[c];
    if (!std::isfinite(s) || !std::isfinite(r)) {
      cost_[c] = kNoEstimate;
      continue;
    }
    if (s >= slope_lethal_deg_) {
      cost_[c] = nav2_costmap_2d::LETHAL_OBSTACLE;
      continue;
    }
    const double severity = std::max(
      ramp01(s, slope_free_deg_, slope_max_cost_deg_),
      ramp01(r, rough_free_m_, rough_max_cost_m_));
    cost_[c] = static_cast<unsigned char>(std::lround(severity * max_cost_));
  }
}

void TerrainLayer::updateBounds(
  double robot_x, double robot_y, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!have_origin_) {
    recenter(robot_x, robot_y);
  }
  const double extent = cells_ * resolution_;
  *min_x = std::min(*min_x, origin_x_);
  *min_y = std::min(*min_y, origin_y_);
  *max_x = std::max(*max_x, origin_x_ + extent);
  *max_y = std::max(*max_y, origin_y_ + extent);
}

void TerrainLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!have_origin_) {
    return;
  }
  for (int mj = min_j; mj < max_j; ++mj) {
    for (int mi = min_i; mi < max_i; ++mi) {
      double wx, wy;
      master_grid.mapToWorld(mi, mj, wx, wy);
      int i, j;
      if (!worldToCell(wx, wy, i, j)) {
        continue;
      }
      const unsigned char c = cost_[static_cast<size_t>(j) * cells_ + i];
      if (c == kNoEstimate || c == 0) {
        continue;
      }
      const unsigned char old = master_grid.getCost(mi, mj);
      if (old == nav2_costmap_2d::NO_INFORMATION || c > old) {
        master_grid.setCost(mi, mj, c);
      }
    }
  }
}

void TerrainLayer::publishDebug(const rclcpp::Time & stamp)
{
  if (!slope_pub_->is_activated()) {
    return;
  }
  nav_msgs::msg::OccupancyGrid grid;
  grid.header.frame_id = layered_costmap_->getGlobalFrameID();
  grid.header.stamp = stamp;
  grid.info.resolution = static_cast<float>(resolution_);
  grid.info.width = cells_;
  grid.info.height = cells_;
  grid.info.origin.position.x = origin_x_;
  grid.info.origin.position.y = origin_y_;
  grid.info.origin.orientation.w = 1.0;

  const size_t n = cost_.size();
  auto fill = [&](auto value_fn) {
      grid.data.resize(n);
      for (size_t c = 0; c < n; ++c) {
        grid.data[c] = value_fn(c);
      }
    };

  // Occupancy values 0..100, -1 unknown. Scaled to each metric's full-cost limit.
  fill([&](size_t c) -> int8_t {
      const float s = slope_deg_[c];
      return std::isfinite(s) ?
      static_cast<int8_t>(std::lround(100.0 * std::min(1.0, s / slope_lethal_deg_))) : -1;
    });
  slope_pub_->publish(grid);

  fill([&](size_t c) -> int8_t {
      const float r = roughness_m_[c];
      return std::isfinite(r) ?
      static_cast<int8_t>(std::lround(100.0 * std::min(1.0, r / rough_max_cost_m_))) : -1;
    });
  rough_pub_->publish(grid);

  fill([&](size_t c) -> int8_t {
      return cost_[c] == kNoEstimate ? -1 :
      static_cast<int8_t>(std::lround(100.0 * cost_[c] / 254.0));
    });
  cost_pub_->publish(grid);
}

}  // namespace terrain_costmap

PLUGINLIB_EXPORT_CLASS(terrain_costmap::TerrainLayer, nav2_costmap_2d::Layer)
