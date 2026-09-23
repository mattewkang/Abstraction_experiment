cmake_minimum_required(VERSION 3.0.2)
project(g1_hands)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  TactileStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    message_runtime
)

catkin_install_python(PROGRAMS
  scripts/revo2_tactile_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Remote-bringup shell helper (PC1 -> G1 PC2 over SSH). See the note in
# g1_camera/CMakeLists.txt on why this is install(PROGRAMS ...), not
# catkin_install_python.
install(PROGRAMS
  scripts/start_remote_hands.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
