cmake_minimum_required(VERSION 3.14)
project(FeatureDetectionAndMatching LANGUAGES CXX)

# ── Layout roots (project location organization) ─────────────────────────────
set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(PROJECT_UI_DIR      "${CMAKE_SOURCE_DIR}/ui")
set(PROJECT_SRC_DIR     "${CMAKE_SOURCE_DIR}/src")
set(PROJECT_LEGACY_DIR  "${CMAKE_SOURCE_DIR}/src2")

option(SURFACE_LEGACY_LAYOUT "Show legacy src2 files in IDE filters (not built)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${PROJECT_UI_DIR}")
set(CMAKE_AUTOMOC_COMPILER_PREDEFINES ON)

# ── OpenCV ────────────────────────────────────────────────────────────────────
set(OpenCV_INCLUDE_DIRS "D:/opencv/build/include")
set(OpenCV_LIB
    "$<IF:$<CONFIG:Debug>,D:/opencv/build/x64/vc16/lib/opencv_world4120d.lib,D:/opencv/build/x64/vc16/lib/opencv_world4120.lib>"
)
set(OpenCV_DLL
    "$<IF:$<CONFIG:Debug>,D:/opencv/build/x64/vc16/bin/opencv_world4120d.dll,D:/opencv/build/x64/vc16/bin/opencv_world4120.dll>"
)

# ── Qt6 ───────────────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

# ── Sources ───────────────────────────────────────────────────────────────────
set(UI_FILES
    "${PROJECT_UI_DIR}/mainwindow.ui"
)

set(CORE_FILES
    "${PROJECT_SRC_DIR}/main.cpp"
    "${PROJECT_SRC_DIR}/mainwindow.cpp"
    "${PROJECT_SRC_DIR}/mainwindow_setup.cpp"
    "${PROJECT_SRC_DIR}/mainwindow_shared.cpp"
    "${PROJECT_SRC_DIR}/mainwindow_tab1.cpp"
    "${PROJECT_SRC_DIR}/mainwindow_tab2.cpp"
    "${PROJECT_SRC_DIR}/mainwindow_tab3.cpp"
    "${PROJECT_INCLUDE_DIR}/mainwindow.h"
)

set(WIDGET_FILES
    "${PROJECT_SRC_DIR}/interactive_label.cpp"
    "${PROJECT_INCLUDE_DIR}/widgets/interactive_label.h"
    "${PROJECT_SRC_DIR}/zoomable_image.cpp"
    "${PROJECT_INCLUDE_DIR}/widgets/zoomable_image.hpp"
)

set(SIFT_FILES
    "${PROJECT_SRC_DIR}/SiftCore.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_extract.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_pyramid.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_extrema_orientation.cpp"
    "${PROJECT_SRC_DIR}/sift/sift_descriptor.cpp"
    "${PROJECT_SRC_DIR}/Timer.cpp"
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
    ${UI_FILES}
    ${CORE_FILES}
    ${WIDGET_FILES}
    ${SIFT_FILES}
    ${HARRIS_FILES}
    ${IO_UTIL_FILES}
)

source_group(TREE ${CMAKE_SOURCE_DIR} FILES
    ${UI_FILES}
    ${CORE_FILES}
    ${WIDGET_FILES}
    ${SIFT_FILES}
    ${HARRIS_FILES}
    ${IO_UTIL_FILES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/FeatureDetectionAndMatching_autogen/include
)

if(SURFACE_LEGACY_LAYOUT)
    file(GLOB_RECURSE LEGACY_LAYOUT_FILES CONFIGURE_DEPENDS
        "${PROJECT_LEGACY_DIR}/*.cpp"
        "${PROJECT_LEGACY_DIR}/*.h"
        "${PROJECT_LEGACY_DIR}/*.hpp"
        "${CMAKE_SOURCE_DIR}/feature_detection.ui"
    )
    source_group(TREE ${CMAKE_SOURCE_DIR} PREFIX "legacy" FILES ${LEGACY_LAYOUT_FILES})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIB}
    Qt6::Widgets
    Qt6::Concurrent
)

# ── Copy OpenCV DLL ───────────────────────────────────────────────────────────
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DLL}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying OpenCV DLL..."
)

# ── Deploy Qt DLLs ────────────────────────────────────────────────────────────
find_program(WINDEPLOYQT_EXECUTABLE windeployqt
    HINTS "${CMAKE_PREFIX_PATH}/bin"
)
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

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
