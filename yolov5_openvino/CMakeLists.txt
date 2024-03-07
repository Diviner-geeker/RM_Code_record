cmake_minimum_required(VERSION 3.16)
project(openvino_detect)

set(CMAKE_CXX_STANDARD 11)
set(ngraph_DIR "/opt/intel/openvino_2022.3.1/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2022.3.1/runtime/cmake")

find_package(OpenCV 4 REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(
        #OpenVINO推理引擎的头文件
        /opt/intel/openvino_2022.3.1/runtime/include/ie/
        /opt/intel/openvino_2022.3.1/runtime/include/ngraph/
        /opt/intel/openvino_2022.3.1/runtime/include/openvino/

)
include_directories(
        ${OpenCV_LIBRARY_DIRS}
)
# add_subdirectory(yolov5_openvino)#add
add_executable(yolo_test ${sourcefile} openvino_detect.cpp openvino_detect.h demo.cpp)

# add_executable(yolo_test demo.cpp)
target_link_libraries(
        yolo_test
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        fmt::fmt
        # yolov5_openvino
)
