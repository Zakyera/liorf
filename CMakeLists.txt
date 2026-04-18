cmake_minimum_required(VERSION 2.8.3)
project(liorf)

set(CMAKE_BUILD_TYPE "Release")
# zy Step 2_fix_a
# CBS and aria_common use C++17 APIs, so LIORF must compile in C++17 mode.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
# zy Step 1_a
# Intuition: add an optional CBS build switch so LIORF can keep legacy behavior while we integrate incrementally.
option(LIORF_USE_CBS "Build LIORF with CBS/BPSAM integration hooks" OFF)
if(LIORF_USE_CBS)
  set(_CBS_WS_HINT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  set(_CBS_SRC_HINT "${CMAKE_CURRENT_SOURCE_DIR}/../cbs")
  if(NOT cbs_DIR)
    if(EXISTS "${_CBS_WS_HINT}/devel/lib/cmake/cbs/cbsConfig.cmake")
      set(cbs_DIR "${_CBS_WS_HINT}/devel/lib/cmake/cbs" CACHE PATH
          "Directory containing cbsConfig.cmake")
    elseif(EXISTS "${_CBS_WS_HINT}/build/cbs/cbsConfig.cmake")
      set(cbs_DIR "${_CBS_WS_HINT}/build/cbs" CACHE PATH
          "Directory containing cbsConfig.cmake")
    elseif(EXISTS "${_CBS_SRC_HINT}/build/cbsConfig.cmake")
      set(cbs_DIR "${_CBS_SRC_HINT}/build" CACHE PATH
          "Directory containing cbsConfig.cmake")
    endif()
  endif()
  find_package(cbs REQUIRED CONFIG)
endif()
find_package(Boost REQUIRED COMPONENTS timer)

# --- Make catkin_package(DEPENDS GTSAM) happy ---
set(GTSAM_INCLUDE_DIRS ${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")
find_package(GeographicLib REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

# include directories
include_directories(
  lib
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIRS}
)

# link directories
link_directories(
  lib
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

file(GLOB common_lib "lib/*.cpp")

###########
## Build ##
###########
# <!-- liorf_yjz_lucky_boy -->
# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp ${common_lib})
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp ${common_lib} include/Scancontext.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization Boost::timer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} ${GeographicLib_LIBRARIES} gtsam)
# zy Step 1_b
# Intuition: only mapOptimization needs CBS first, because it is LIORF's global optimization heart.
if(LIORF_USE_CBS)
  target_link_libraries(${PROJECT_NAME}_mapOptmization cbs)
  target_compile_definitions(${PROJECT_NAME}_mapOptmization PRIVATE LIORF_USE_CBS=1)
endif()



# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp ${common_lib})
target_link_libraries(${PROJECT_NAME}_imuPreintegration Boost::timer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)
