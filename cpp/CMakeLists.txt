cmake_minimum_required(VERSION 2.6)

project(dis)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.4.4/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-11.4.4/lib64)
# tensorrt
include_directories(path/TensorRT-8.2.1.8/include)
link_directories(path/TensorRT-8.2.1.8/targets/x86_64-linux-gnu/lib/)

# opencv
include_directories(/usr/local/include/opencv4/)
link_directories(/usr/local/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#target_link_libraries(nvinfer cudart)

set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4/)
set(OpenCV_LIBRARIES opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_dnn)
set(OpenCV_LIB_DIR /usr/local/lib64/)

message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "Find OpenCV lib dir: ${OpenCV_LIB_DIR}")

add_executable(dis ${PROJECT_SOURCE_DIR}/dis.cpp)

target_link_libraries(dis nvinfer)
target_link_libraries(dis cudart)
target_link_libraries(dis ${OpenCV_LIBRARIES})

add_definitions(-O2 -pthread)

