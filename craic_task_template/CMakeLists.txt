cmake_minimum_required(VERSION 3.0.2)
project(craic_task_template)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/apriltag_reader.py
  scripts/grab_controller.py
  scripts/grab_parts.py
  scripts/scene1_patrol.py
  scripts/scene2_sorting.py
  scripts/stairClimbPlanner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
