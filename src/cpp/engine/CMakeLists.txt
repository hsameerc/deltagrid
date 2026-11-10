cmake_minimum_required(VERSION 3.14)
project(deltagrid_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(-DDELTAGRID_EXPORTS)

if(NOT MSVC)
    add_compile_options(-O3 -Wall -Wextra -Wno-unknown-pragmas)
endif()

if(APPLE)
    option(DELTAGRID_USE_OPENMP "Enable OpenMP multithreading" OFF)
else()
    option(DELTAGRID_USE_OPENMP "Enable OpenMP multithreading" ON)
endif()

if(DELTAGRID_USE_OPENMP)
    find_package(OpenMP QUIET)
endif()

# 1. Shared Libraries
add_library(deltagrid_continuous SHARED
    src/deltagrid_continuous.cpp
    src/deltagrid_math.cpp
    src/deltagrid_auth.cpp
)

add_library(deltagrid_fxp SHARED
    src/deltagrid_fxp.cpp
    src/deltagrid_math_fxp.cpp
    src/deltagrid_auth.cpp
)

add_library(deltagrid_llm SHARED
    src/deltagrid_llm.cpp
    src/deltagrid_math.cpp
    src/deltagrid_auth.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(deltagrid_continuous PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(deltagrid_fxp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(deltagrid_llm PRIVATE OpenMP::OpenMP_CXX)
endif()

# 2. Standalone CLI Executables
add_executable(deltagrid_continuous_cli src/main_continuous.cpp)
target_link_libraries(deltagrid_continuous_cli PRIVATE deltagrid_continuous)

add_executable(deltagrid_fxp_cli src/main_fxp.cpp)
target_link_libraries(deltagrid_fxp_cli PRIVATE deltagrid_fxp)

add_executable(deltagrid_llm_cli src/main_llm.cpp)
target_link_libraries(deltagrid_llm_cli PRIVATE deltagrid_llm)

# 3. Unit Tests
enable_testing()
add_executable(test_math tests/test_math.cpp src/deltagrid_math.cpp src/deltagrid_auth.cpp)
add_test(NAME test_math COMMAND test_math)

add_executable(test_fxp tests/test_fxp.cpp src/deltagrid_math_fxp.cpp src/deltagrid_auth.cpp)
add_test(NAME test_fxp COMMAND test_fxp)

add_executable(verify_divergence
    tests/verify_divergence.cpp
    src/deltagrid_continuous.cpp
    src/deltagrid_fxp.cpp
    src/deltagrid_math.cpp
    src/deltagrid_math_fxp.cpp
    src/deltagrid_auth.cpp
)
add_test(NAME verify_divergence COMMAND verify_divergence)
