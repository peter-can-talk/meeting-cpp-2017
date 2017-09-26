###########################################################
## CMAKE SETUP
###########################################################

cmake_minimum_required(VERSION 3.2)
project(canvas)

###########################################################
## INCLUDES
###########################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

###########################################################
## SOURCES
###########################################################

set(CANVAS_SOURCES main.cpp)

###########################################################
## TARGETS
###########################################################

# The following variables are for Qt support
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find the actual Qt5 packages
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Qml REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Svg REQUIRED)

add_executable(canvas ${CANVAS_SOURCES})

###########################################################
## COMPILER FLAGS
###########################################################

target_compile_options(canvas PUBLIC -std=c++14)
