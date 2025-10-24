cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# Get ZENO path from environment variable
if(DEFINED ENV{ZENOPATH})
  set(ZENO_ROOT $ENV{ZENOPATH})
  message(STATUS "ZENO path: ${ZENO_ROOT}")
else()
  message(FATAL_ERROR "ZENOPATH environment variable is not set. Please set it to your ZENO installation directory.")
endif()

# Set ZENO include and library paths
set(ZENO_INCLUDE_DIR "${ZENO_ROOT}/cpp/libzeno/include")
set(ZENO_LIBRARY_DIR "${ZENO_ROOT}/zeno-build")

# Find ZENO library
find_library(ZENO_LIBRARY
  NAMES zeno libzeno
  PATHS ${ZENO_LIBRARY_DIR}
  NO_DEFAULT_PATH
  REQUIRED
)

if(ZENO_LIBRARY)
  message(STATUS "Found ZENO library: ${ZENO_LIBRARY}")
else()
  message(FATAL_ERROR "ZENO library not found in ${ZENO_LIBRARY_DIR}")
endif()

# Check if ZENO include directory exists
if(NOT EXISTS ${ZENO_INCLUDE_DIR})
  message(FATAL_ERROR "ZENO include directory not found: ${ZENO_INCLUDE_DIR}")
endif()

message(STATUS "ZENO include directory: ${ZENO_INCLUDE_DIR}")

nanobind_add_module(
    zenowrapper_ext
    STABLE_ABI
    NB_STATIC
    src/zenowrapper/zenolib.cpp
)

# Add ZENO include directory
target_include_directories(zenowrapper_ext PRIVATE ${ZENO_INCLUDE_DIR})

# Link ZENO library
target_link_libraries(zenowrapper_ext PRIVATE ${ZENO_LIBRARY})

# Add ZENO compile definitions (needed for random number generator selection)
# ZENO requires one of these to be defined: USE_SPRNG_RNG, USE_LEAP_FROG_RNG, or USE_RAND_RNG
# Match ZENO library compilation
target_compile_definitions(zenowrapper_ext PRIVATE USE_SPRNG_RNG)

# Install directive for scikit-build-core
set_target_properties(zenowrapper_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/zenowrapper"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/zenowrapper"   # for Windows DLLs
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/zenowrapper"   # for static/archives
)
install(TARGETS zenowrapper_ext LIBRARY DESTINATION src/zenowrapper)
