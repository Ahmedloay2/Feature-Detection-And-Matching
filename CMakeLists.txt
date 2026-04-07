cmake_minimum_required(VERSION 3.14)
project(FeatureDetectionAndMatching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_SOURCE_DIR}/ui")
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
add_executable(${PROJECT_NAME}

    # Qt UI (AUTOUIC → ui_mainwindow.h)
    "ui/mainwindow.ui"

    # Entry
    "src/main.cpp"

    # Main window
    "src/mainwindow.cpp"
    "include/mainwindow.h"

    # Widgets (moved to widgets/ sub-path)
    "src/interactive_label.cpp"
    "include/widgets/interactive_label.h"
    "src/zoomable_image.cpp"
    "include/widgets/zoomable_image.hpp"

    # SIFT
    "src/SiftCore.cpp"
    "src/Timer.cpp"

    # Harris pipeline
    "src/processors/harris_main.cpp"
    "src/processors/harris/gradient.cpp"
    "src/processors/harris/grayscale.cpp"
    "src/processors/harris/harris_response.cpp"
    "src/processors/harris/shi_tomasi.cpp"
    "src/processors/harris/strcutre_tensor.cpp"
    "src/processors/harris/threshold.cpp"
    "src/processors/harris/nms.cpp"
    "src/processors/harris/gaussian.cpp"

    # Utilities / IO
    "src/utils/utils.cpp"
    "src/io/image_handler.cpp"

)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/FeatureDetectionAndMatching_autogen/include
)

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
