cmake_minimum_required(VERSION 3.14)
project(FeatureDetectionAndMatching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC_COMPILER_PREDEFINES ON)

# ── OpenCV (hardcoded paths) ─────────────────────────────────────
set(OpenCV_INCLUDE_DIRS "D:/opencv/build/include")
set(OpenCV_LIB
    "$<IF:$<CONFIG:Debug>,D:/opencv/build/x64/vc16/lib/opencv_world4120d.lib,D:/opencv/build/x64/vc16/lib/opencv_world4120.lib>"
)
set(OpenCV_DLL
    "$<IF:$<CONFIG:Debug>,D:/opencv/build/x64/vc16/bin/opencv_world4120d.dll,D:/opencv/build/x64/vc16/bin/opencv_world4120.dll>"
)

# ── Qt6 ──────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# ── Collect all source files from src/ ───────────────────────────
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HEADER_FILES "include/*.h" "include/*.hpp")

if(NOT SRC_FILES)
    message(FATAL_ERROR "No source files found in src/ directory! Create at least src/main.cpp")
endif()

# ── Build executable ──────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES} 
"src/main.cpp" 
"src/processors/harris/gradient.cpp"
"src/processors/harris/grayscale.cpp" 
"src/processors/harris/harris_response.cpp"
"src/utils/utils.cpp"
"src/io/image_handler.cpp"
"src/processors/harris/strcutre_tensor.cpp"
"src/processors/harris/threshold.cpp"
"src/processors/harris/nms.cpp" 
"src/processors/harris/shi_tomasi.cpp"
"src/processors/harris_main.cpp" "src/processors/harris/gaussian.cpp")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/Object_Shapes_Detector_autogen/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIB}
    Qt6::Widgets
)

# ── Copy OpenCV DLL next to executable ───────────────────────────
# Note: generator expressions work correctly in add_custom_command with COMMAND
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DLL}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying OpenCV DLL..."
)

# ── Deploy Qt DLLs using windeployqt ─────────────────────────────
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

# ── Startup project for Visual Studio ────────────────────────────
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# ── Helpful message to fix the cache error ───────────────────────
if(EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
    file(STRINGS "${CMAKE_BINARY_DIR}/CMakeCache.txt" CACHE_SOURCE_DIR REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=.*")
    if(CACHE_SOURCE_DIR)
        string(REGEX REPLACE "^CMAKE_HOME_DIRECTORY:INTERNAL=" "" CACHE_SOURCE_DIR "${CACHE_SOURCE_DIR}")
        if(NOT "${CACHE_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
            message(WARNING "\n⚠️  CMake cache was created from a different source directory:\n"
                           "   Old: ${CACHE_SOURCE_DIR}\n"
                           "   New: ${CMAKE_SOURCE_DIR}\n"
                           "   Please delete the 'out' folder and reconfigure.\n")
        endif()
    endif()
endif()