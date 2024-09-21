cmake_minimum_required(VERSION 3.16.3)
# set(CMAKE_CXX_STANDARD 17)
project(Hik_camera_test)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

# link_directories($ENV{MVCAM_COMMON_RUNENV}/64)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# add_executable(test test.cpp)

# target_link_libraries(test ${OpenCV_LIBS} MvCameraControl fmt::fmt MVSDK)

add_executable(mvtest mv_test.cpp io/camera.cpp)
target_link_libraries(mvtest ${OpenCV_LIBS} MVSDK )