cmake_minimum_required(VERSION 3.0.2)
project(simple_nav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  move_base
  map_server
  amcl
  pointcloud_to_laserscan
  dwa_local_planner
  navfn
  costmap_2d
  move_base_msgs
  actionlib
  std_srvs
  tf
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/auto_patrol.py
  scripts/scene2_table_nav.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
