# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Hexagon backend: a DSP op library ported from MNN, driven by a host runtime
# that speaks the same FastRPC interface MNN does.
#
# Two halves with different toolchains:
#   host  - built by the normal ExecuTorch build, links libcdsprpc
#   skel  - cross-compiled per Hexagon arch, one .so per arch, deployed to the
#           device separately
#
# Set HEXAGON_SDK_ROOT and HEXAGON_TOOLS_ROOT, and list the arches to build in
# HEXAGON_ARCHS (e.g. "v79;v81").

if(NOT EXECUTORCH_BUILD_HEXAGON)
  return()
endif()

if(NOT HEXAGON_SDK_ROOT)
  message(FATAL_ERROR "EXECUTORCH_BUILD_HEXAGON requires HEXAGON_SDK_ROOT")
endif()

set(HEXAGON_SDK_ROOT "${HEXAGON_SDK_ROOT}" CACHE PATH "Hexagon SDK root")

if(NOT HEXAGON_ARCHS)
  set(HEXAGON_ARCHS "v79")
endif()

set(HEXAGON_MNN_OPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/mnn-htp-ops")

# The skel is only reachable through the IDL, so the host stub and the skel are
# generated from the same file and can never disagree about handle numbers.
set(HEXAGON_IDL_DIR "${CMAKE_BINARY_DIR}/backends/hexagon/idl")
find_program(HEXAGON_QAIC
    NAMES qaic
    HINTS "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu")

if(NOT HEXAGON_QAIC)
  message(FATAL_ERROR "qaic not found under ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic")
endif()

add_custom_command(
    OUTPUT
        ${HEXAGON_IDL_DIR}/htp_ops.h
        ${HEXAGON_IDL_DIR}/htp_ops_stub.c
        ${HEXAGON_IDL_DIR}/htp_ops_skel.c
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HEXAGON_IDL_DIR}
    COMMAND ${HEXAGON_QAIC} -mdll -o ${HEXAGON_IDL_DIR}
        -I${HEXAGON_SDK_ROOT}/incs
        -I${HEXAGON_SDK_ROOT}/incs/stddef
        -I${HEXAGON_MNN_OPS_ROOT}/include
        ${HEXAGON_MNN_OPS_ROOT}/include/htp_ops.idl
    DEPENDS ${HEXAGON_MNN_OPS_ROOT}/include/htp_ops.idl
    COMMENT "Generating Hexagon FastRPC stubs from htp_ops.idl"
    VERBATIM)

add_custom_target(hexagon_idl DEPENDS
    ${HEXAGON_IDL_DIR}/htp_ops.h
    ${HEXAGON_IDL_DIR}/htp_ops_stub.c
    ${HEXAGON_IDL_DIR}/htp_ops_skel.c)

# The schema header checked in with the vendored library was produced by an
# older flatbuffers codegen and does not compile against the version this tree
# vendors, so it is regenerated here. The schema itself is untouched, and the
# wire format is the same one the DSP was built against.
set(HEXAGON_SCHEMA_DIR "${CMAKE_BINARY_DIR}/backends/hexagon/schema")

if(TARGET flatc)
  set(HEXAGON_FLATC $<TARGET_FILE:flatc>)
else()
  find_program(HEXAGON_FLATC flatc)
endif()

if(NOT HEXAGON_FLATC)
  message(FATAL_ERROR "flatc not found; it is built alongside ExecuTorch's own schema")
endif()

add_custom_command(
    OUTPUT ${HEXAGON_SCHEMA_DIR}/Command_generated.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HEXAGON_SCHEMA_DIR}
    COMMAND ${HEXAGON_FLATC} --cpp --cpp-std c++11 --gen-mutable --scoped-enums
        -o ${HEXAGON_SCHEMA_DIR}
        ${HEXAGON_MNN_OPS_ROOT}/schema/current/Command.fbs
    DEPENDS ${HEXAGON_MNN_OPS_ROOT}/schema/current/Command.fbs
    COMMENT "Generating the Hexagon DSP command schema"
    VERBATIM)

add_custom_target(hexagon_schema DEPENDS
    ${HEXAGON_SCHEMA_DIR}/Command_generated.h)

# libcdsprpc is an on-device vendor library; the host build only needs it at
# link time for an Android target.
set(HEXAGON_DSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
    CACHE FILEPATH "FastRPC client library")

add_subdirectory(runtime)

# ---------------------------------------------------------------------------
# DSP skel, one shared object per Hexagon arch
# ---------------------------------------------------------------------------
#
# Cross-compiling has to use the Hexagon toolchain, which cannot coexist with
# the host toolchain in one CMake configure, so each arch is driven as an
# ExternalProject with its own configure+build.
#
#   hexagon-clang++ -mv79 -mhvx -mhvx-length=128b -mhmx -O2 -fPIC -std=c++17
#   link -L<tools>/target/hexagon/lib/v79/G0/pic -lc++ -lc++abi
#

set(HEXAGON_SKEL_DIR "${CMAKE_BINARY_DIR}/backends/hexagon/skel")

foreach(HEXAGON_ARCH ${HEXAGON_ARCHS})
  string(REGEX REPLACE "^v" "" HEXAGON_ARCH_NUM "${HEXAGON_ARCH}")

  # HVX IEEE fp is required below v79; HMX is the whole point of v79+.
  set(HEXAGON_ARCH_FLAGS "-m${HEXAGON_ARCH} -mhvx -mhvx-length=128b -mhmx")
  if(HEXAGON_ARCH_NUM LESS 79)
    set(HEXAGON_ARCH_FLAGS "${HEXAGON_ARCH_FLAGS} -mhvx-ieee-fp")
  endif()

  ExternalProject_Add(hexagon_skel_${HEXAGON_ARCH}
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/skel
      BINARY_DIR ${HEXAGON_SKEL_DIR}/${HEXAGON_ARCH}
      CMAKE_ARGS
          -DCMAKE_TOOLCHAIN_FILE=${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake
          -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
          -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
          -DMNN_OPS_ROOT=${HEXAGON_MNN_OPS_ROOT}
          -DHEXAGON_FLATBUFFERS_ROOT=${EXECUTORCH_ROOT}/third-party/flatbuffers
          -DSKEL_ARCH=${HEXAGON_ARCH}
          -DSKEL_ARCH_FLAGS=${HEXAGON_ARCH_FLAGS}
          -DIDL_DIR=${HEXAGON_IDL_DIR}
          -DCMAKE_C_FLAGS=${HEXAGON_ARCH_FLAGS}
          -DCMAKE_CXX_FLAGS=${HEXAGON_ARCH_FLAGS}
      BUILD_ALWAYS OFF
      INSTALL_COMMAND ""
      COMMENT "Building Hexagon skel for ${HEXAGON_ARCH}")
  add_dependencies(hexagon_skel_${HEXAGON_ARCH} hexagon_idl)
endforeach()
