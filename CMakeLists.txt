cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD_REQUIRED 17)

project(ov_test)

set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp yolov8.hpp yolov8.cpp)
target_link_libraries(main openvino::runtime ${OpenCV_LIBS})