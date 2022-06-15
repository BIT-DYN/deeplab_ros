cmake_minimum_required(VERSION 2.8.3)
project(img_seg)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  # message_generation
  # message_runtime
  rospy
  # rostest
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
  # CATKIN_DEPENDS cv_bridge message_generation message_runtime rospy rostest std_msgs
  CATKIN_DEPENDS cv_bridge rospy  sensor_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  # predict
  predict.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)
install(FILES
  predict
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)
