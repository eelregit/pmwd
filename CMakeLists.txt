cmake_minimum_required(VERSION 3.4...3.18)

project(jaxpmwd LANGUAGES CXX CUDA)

# Handle Python settings passed from scikit-build
if(SKBUILD)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    set(Python_LIBRARY "${PYTHON_LIBRARY}")
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
      list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
else()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
      list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Find and link pybind11
find_package(pybind11 CONFIG REQUIRED)

# Work out compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wno-unknown-pragmas -O3 -funroll-loops)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES "80")
    add_compile_options(-Wall -Wno-unknown-pragmas -O3 -funroll-loops)

    # Find cufft
    find_package(CUDAToolkit)


    pybind11_add_module(_jaxpmwd ${CMAKE_CURRENT_LIST_DIR}/pmwdlib/jax_pmwd_gpu.cc ${CMAKE_CURRENT_LIST_DIR}/pmwdlib/kernels.cc.cu)
    target_link_libraries(_jaxpmwd PRIVATE ${CUDA_nvToolsExt_LIBRARY})
    install(TARGETS _jaxpmwd DESTINATION .)
else()
    message(STATUS "No CUDA compiler found; GPU support will be disabled")
endif()
