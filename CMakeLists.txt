cmake_minimum_required(VERSION 3.19)
project(slam)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -pthread -march=native")

set(CMAKE_CXX_STANDARD 14)

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

set(CMAKE_PREFIX_PATH "/opt/ros/melodic")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

Find_Package(Ceres REQUIRED)

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})


# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_transport
    geometry_msgs
    sensor_msgs
    nav_msgs
    std_msgs
    message_filters
    roscpp
    rospy
    tf
    tf2
    message_generation
)

#add_service_files(
#        FILES
#)

#generate_messages(
#    DEPENDENCIES
#    std_msgs
#)

catkin_package (
    CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs dynamic_reconfigure message_runtime
    LIBRARIES {PROJECT_NAME} libDBoW2 libg2o
)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty
)

#include(${PROJECT_SOURCE_DIR}/3rdparty/g2o/CMakeLists.txt)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${G2O_LIBRARIES}
        ${CERES_LIBRARIES}
        #${PROJECT_SOURCE_DIR}/3rdparty/g2o/lib/libg2o.so
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        ${catkin_LIBRARIES}
        )

enable_testing()

############### source and test ######################
MESSAGE("PROJECT_SOURCE_DIR:")
message(${PROJECT_SOURCE_DIR})

add_subdirectory(src)
#add_subdirectory(test)
add_subdirectory(app)

#add_executable(slam main.cpp)