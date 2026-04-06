cmake_minimum_required(VERSION 3.14)
project(FeatureDetectionAndMatching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC_COMPILER_PREDEFINES ON)

find_package(OpenCV REQUIRED)

# ── Qt6 ──────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)
find_package(OpenMP REQUIRED)

# ── Collect all source files from src/ ───────────────────────────
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HEADER_FILES "include/*.h" "include/*.hpp")

if(NOT SRC_FILES)
    message(FATAL_ERROR "No source files found in src/ directory! Create at least src/main.cpp")
endif()

# ── Build executable ──────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_autogen/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Qt6::Widgets
    Qt6::Concurrent
    OpenMP::OpenMP_CXX
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