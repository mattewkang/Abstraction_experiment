cmake_minimum_required(VERSION 3.0.2)
project(g1_camera)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_service_files(
  FILES
  GetObjectPosition.srv
  GetBerries.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
)

catkin_install_python(PROGRAMS
  scripts/object_detector_node.py
  scripts/berry_detector_node.py
  scripts/fetch_weights.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Remote-bringup shell helpers (PC1 -> G1 PC2 over SSH). install(PROGRAMS ...)
# rather than catkin_install_python: these are bash, and catkin_install_python
# rewrites the shebang to the Python interpreter, which would break them.
install(PROGRAMS
  scripts/start_remote_camera.sh
  scripts/setup_remote_timesync.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
