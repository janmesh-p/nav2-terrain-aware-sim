cmake_minimum_required(VERSION 3.8)
project(terrain_costmap)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

set(deps
  rclcpp rclcpp_lifecycle nav2_costmap_2d pluginlib
  sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros)

add_library(${PROJECT_NAME} SHARED
  src/terrain_layer.cpp
  src/plane_fit.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME} ${deps})

pluginlib_export_plugin_description_file(nav2_costmap_2d terrain_layer.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/)
install(FILES terrain_layer.xml DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_plane_fit test/test_plane_fit.cpp src/plane_fit.cpp)
  target_include_directories(test_plane_fit PUBLIC include)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${deps})
ament_package()
