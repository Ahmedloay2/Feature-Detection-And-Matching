cmake_minimum_required(VERSION 3.14)
project(FeatureDetectionAndMatching LANGUAGES CXX)

set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(PROJECT_UI_DIR      "${CMAKE_SOURCE_DIR}/ui")
set(PROJECT_SRC_DIR     "${CMAKE_SOURCE_DIR}/src")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${PROJECT_UI_DIR}")
set(CMAKE_AUTOMOC_COMPILER_PREDEFINES ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)
find_package(OpenMP REQUIRED)

set(UI_FILES "${PROJECT_UI_DIR}/mainwindow.ui")

set(CORE_FILES
    "${PROJECT_SRC_DIR}/main.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow_setup.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow_shared.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow_tab1.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow_tab2.cpp"
    "${PROJECT_SRC_DIR}/ui/mainwindow_tab3.cpp"
    "${PROJECT_INCLUDE_DIR}/ui/mainwindow.h"
)

set(WIDGET_FILES
    "${PROJECT_SRC_DIR}/ui/interactive_label.cpp"
    "${PROJECT_INCLUDE_DIR}/ui/interactive_label.h"
    "${PROJECT_SRC_DIR}/ui/zoomable_image.cpp"
    "${PROJECT_INCLUDE_DIR}/ui/zoomable_image.hpp"
)

set(SIFT_FILES
    "${PROJECT_SRC_DIR}/core/SiftCore.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_extract.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_pyramid.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_extrema_orientation.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_descriptor.cpp"
    "${PROJECT_SRC_DIR}/utils/Timer.cpp"
)

set(HARRIS_FILES
    "${PROJECT_SRC_DIR}/processors/harris_main.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/gradient.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/grayscale.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/harris_response.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/shi_tomasi.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/strcutre_tensor.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/threshold.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/nms.cpp"
    "${PROJECT_SRC_DIR}/processors/harris/gaussian.cpp"
)

set(IO_UTIL_FILES
    "${PROJECT_SRC_DIR}/utils/utils.cpp"
    "${PROJECT_SRC_DIR}/io/image_handler.cpp"
)

add_executable(${PROJECT_NAME} 
    ${UI_FILES} ${CORE_FILES} ${WIDGET_FILES} ${SIFT_FILES} ${HARRIS_FILES} ${IO_UTIL_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/FeatureDetectionAndMatching_autogen/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Qt6::Widgets
    Qt6::Concurrent
    OpenMP::OpenMP_CXX
)

find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${CMAKE_PREFIX_PATH}/bin")
if(WINDEPLOYQT_EXECUTABLE)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${WINDEPLOYQT_EXECUTABLE}
            $<IF:$<CONFIG:Debug>,--debug,--release>
            --no-translations
            --no-system-d3d-compiler
            --no-opengl-sw
            "$<TARGET_FILE:${PROJECT_NAME}>"
        COMMENT "Deploying Qt DLLs..."
    )
endif()
