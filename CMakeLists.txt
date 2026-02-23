cmake_minimum_required(VERSION 3.0.2)
project(astar)

# 1. 设置 C++ 标准为 17 (工程规范)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启编译警告，有助于发现潜在 bug
add_definitions(-Wall) 

# 2. 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  mavros_msgs
  nav_msgs
  tf
  livox_ros_driver
  pcl_conversions
  pcl_ros
  eigen_conversions
)

# 查找 PCL (必须)
find_package(PCL REQUIRED)

# 3. 声明 Catkin 包信息
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    roscpp 
    std_msgs 
    geometry_msgs 
    mavros_msgs 
    nav_msgs 
    tf 
    livox_ros_driver 
    pcl_conversions 
    pcl_ros
)

# 4. 包含头文件路径
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# PCL 需要的一些定义
add_definitions(${PCL_DEFINITIONS})

# 5. 生成可执行文件
# 注意：这里源文件只包含 src/planner_core.cpp
# 因为 detect_obs 的逻辑是包含在头文件里的 (new_detect_obs.h)
add_executable(astar_node src/planner_core.cpp)

# 6. 链接库文件
target_link_libraries(astar_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 7. 安装规则 (可选，方便 install)
install(TARGETS astar_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)