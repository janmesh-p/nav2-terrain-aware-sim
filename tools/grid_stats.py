#!/usr/bin/env python3
"""Print coverage and value stats for the terrain layer debug grids."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2

BASE = "/local_costmap/local_costmap/terrain_layer/"


class Stats(Node):
    def __init__(self):
        super().__init__("terrain_grid_stats")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.seen = {}
        for name in ("cost", "slope", "roughness"):
            self.create_subscription(OccupancyGrid, BASE + name,
                                     lambda m, n=name: self.grid(n, m), qos)
        self.create_subscription(PointCloud2, "/front_3d_lidar/lidar_points",
                                 self.cloud, 10)

    def grid(self, name, msg):
        d = np.array(msg.data, dtype=np.int16)
        known = d[d >= 0]
        self.get_logger().info(
            f"{name:9s} known {known.size}/{d.size} ({100 * known.size / d.size:.1f}%)"
            + (f"  p50 {np.percentile(known, 50):.0f}  p95 {np.percentile(known, 95):.0f}"
               f"  max {known.max()}" if known.size else ""))
        self.seen[name] = True

    def cloud(self, msg):
        if "cloud" in self.seen:
            return
        self.seen["cloud"] = True
        self.get_logger().info(
            f"cloud frame '{msg.header.frame_id}' points {msg.width * msg.height} "
            f"fields {[f.name for f in msg.fields]}")


def main():
    rclpy.init()
    node = Stats()
    end = node.get_clock().now().nanoseconds + 5e9
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()