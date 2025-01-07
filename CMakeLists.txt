cmake_minimum_required(VERSION 3.14.5)

# set the project name
project(im2col
        VERSION 1.0.0 
        DESCRIPTION "im2col 2D SIMD")

include(ExternalProject)

find_package(Git REQUIRED)
find_program(MAKE_EXE NAMES nmake make)
set(MKDIR_COMMAND ${CMAKE_COMMAND} -E make_directory)
find_program(COPY_COMMAND cp)
find_program(BASH bash)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
  set(MKDIR_EXE ${MKDIR_COMMAND})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie -rdynamic")
  add_compile_options(-rdynamic -pg)
  add_link_options(-rdynamic -pg)
elseif(MSVC)
  set(MKDIR_EXE ${MKDIR_COMMAND})
endif()

# Common compile options
if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
  add_compile_options(-no-pie -Wall)
else()
  if(MSVC)
    add_compile_options(/Zc:preprocessor -Wall /MT)
  else()
    add_compile_options(-no-pie -msse4.1 -Wall)
  endif()
endif()

# MSVC-specific flags
if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  set(CMAKE_CXX_FLAGS_RELEASE "/MT -O3 -fassociative-math -ffast-math -ftree-vectorize -mavx2 -mfma -fstrict-aliasing")
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd -mavx2")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8192000")
endif()

if(WIN32)
  set(DEBSTR "--config Debug")
  set(RELSTR "--config Release")
  set(TYPE_BUILD_COMMAND $<IF:$<CONFIG:Debug>,${DEBSTR},${RELSTR}>)
else()
  set(TYPE_BUILD_COMMAND)
endif()

# --------------------------------------------------------
# Detect AVX2 (if on x86_64, non-MSVC) or NEON (if on arm/aarch64)
# Else fallback to reference implementation
# --------------------------------------------------------

# We'll store the chosen source file in IM2COL_SOURCE.
# By default: use reference
set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_ref.cpp")

if(NOT MSVC)
  include(CheckCXXCompilerFlag)

  # Check if the compiler supports -mavx2
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

  if (COMPILER_SUPPORTS_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64"))
    message(STATUS "AVX2 detected; building with im2col_AVX2.cpp")
    set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_AVX2.cpp")

    # Add additional AVX2 compiler options for GCC/Clang
    add_compile_options(
      -mavx2 
      -mfma 
      -fstrict-aliasing 
      -O3 
      -fassociative-math 
      -ffast-math 
      -ftree-vectorize
    )

  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)$")
    message(STATUS "NEON (Arm) detected; building with im2col_NEON.cpp")
    set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_NEON.cpp")

    # For aarch64, NEON is typically enabled by default, but you can add:
    add_compile_options(
      -march=armv8-a
      -O3
      -fassociative-math
      -ffast-math
      -ftree-vectorize
      -fstrict-aliasing
  )
  else()
    message(STATUS "Neither AVX2 nor NEON detected; building reference (im2col_ref.cpp)")
  endif()

else()
  # MSVC path: we rely on /arch:AVX2 for AVX2, or fallback to reference
  # (MSVC doesn't do -march=armv8-a or -mavx2 in the same way)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(STATUS "MSVC x86_64 build; forcing AVX2")
    set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_AVX2.cpp")
    # We'll let the existing /arch:AVX2 handle it
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)$")
    message(STATUS "MSVC ARM build; building NEON code (im2col_NEON.cpp)")
    set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_NEON.cpp")
  else()
    message(STATUS "MSVC unknown CPU; fallback to reference")
    set(IM2COL_SOURCE "${PROJECT_SOURCE_DIR}/src/im2col_ref.cpp")
  endif()
endif()

# --------------------------------------------------------
# Finally, build the im2col library with the chosen source
# --------------------------------------------------------
add_library(im2col SHARED 
    ${IM2COL_SOURCE}
)

# If you need any special definitions for neon2sse or others:
target_compile_definitions(im2col PUBLIC NEON2SSE_DISABLE_PERFORMANCE_WARNING)

# If you want to apply some arch-specific flags to this target (e.g. MSVC /arch:AVX2)
target_compile_options(im2col PUBLIC $<$<BOOL:${MSVC}>:/arch:AVX2>)

