cmake_minimum_required(VERSION 3.10)
project(ONNXResNet)

add_definitions(-std=c++17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/sfy/SFY/camera/TensorRT-8.6.1.6/include)
link_directories(/home/sfy/SFY/camera/TensorRT-8.6.1.6/lib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")

add_executable(demo main.cpp ${SOURCES})
target_link_libraries(demo nvinfer)
target_link_libraries(demo cudart)
target_link_libraries(demo nvonnxparser)
target_link_libraries(demo ${OpenCV_LIBS})
