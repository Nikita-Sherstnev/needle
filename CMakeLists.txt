cmake_minimum_required(VERSION 3.15)
project(needle LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_CPP_STANDARD -std=c++14)


find_package(Python 3.13 REQUIRED COMPONENTS Development Interpreter)
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Compiler flags
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++14 -O2 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

###################
### CPU BACKEND ###
###################
pybind11_add_module(ndarray_backend_cpu src/ndarray_backend_cpu.cc)

# Output settings
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(APPLE)
  set_target_properties(ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

####################
### CUDA BACKEND ###
####################
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  message(STATUS "Found CUDA ${CUDA_VERSION}, building CUDA backend")

  # GPU architecture detection
  execute_process(COMMAND nvidia-smi RESULT_VARIABLE NV_RET ERROR_QUIET)
  if(NV_RET EQUAL 0)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    set(CUDA_ARCH_FLAGS "-arch=sm_37")  # K80 compatibility
  endif()

  # CUDA module
  pybind11_add_module(ndarray_backend_cuda src/ndarray_backend_cuda.cu)
  target_compile_options(ndarray_backend_cuda PRIVATE ${CUDA_ARCH_FLAGS})

  # Output settings
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
  )
endif()
