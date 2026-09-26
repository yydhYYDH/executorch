# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Cross-compiled DSP skel for one Hexagon arch. Configured by the parent
# ExternalProject with -DSKEL_ARCH=v79 and -DMNN_OPS_ROOT=... .
#
# The sources are the vendored MNN op library. Nothing here is
# ExecuTorch-specific: the library is consumed through the FastRPC interface in
# htp_ops.idl, exactly as upstream consumes it.

cmake_minimum_required(VERSION 3.22)
project(hex-htp-skel LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# hexagon_fun.cmake reads these before it is usable: PREBUILT_LIB_DIR names the
# toolchain as "hexagon_toolv<major>_v<arch>", and HEXAGON_ARCH/DSP_VERSION tell
# the SDK which arch's headers and libs to select.
get_filename_component(_hexagon_tools_version ${HEXAGON_TOOLS_ROOT} NAME)
string(REGEX MATCH "^([0-9]+)" _hexagon_tools_major "${_hexagon_tools_version}")
set(PREBUILT_LIB_DIR "hexagon_toolv${_hexagon_tools_major}_${SKEL_ARCH}")
set(DSP_VERSION ${SKEL_ARCH})
set(HEXAGON_ARCH ${SKEL_ARCH})

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

string(REGEX REPLACE "^v" "" SKEL_ARCH_NUM "${SKEL_ARCH}")

# Derived here rather than injected by the parent so a standalone configure of
# this directory produces the same binary. HVX IEEE fp is required below v79;
# HMX is what the v79+ kernels use for matmul and attention.
set(SKEL_ARCH_FLAGS -m${SKEL_ARCH} -mhvx -mhvx-length=128b -mhmx)
if(SKEL_ARCH_NUM LESS 79)
  list(APPEND SKEL_ARCH_FLAGS -mhvx-ieee-fp)
endif()
add_compile_options(${SKEL_ARCH_FLAGS})

# A standalone configure gets its own header. The parent passes the host
# header instead so the expected value and the value in the skel are literally
# generated from the same source closure.
set(HEXAGON_SKEL_SOURCE_ID_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/generate_skel_source_id.cmake")
set(HEXAGON_SKEL_SOURCE_ID_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/source_id.cmake")
set(HEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
# hexagon_skel_source_manifest reads this in the outer scope, and the full android
# build never sets it there: its parent passes HEXAGON_SKEL_SOURCE_ID_HEADER, so that
# build takes the else() branch below and never reaches the call. A standalone
# `build.sh skel` has no such header and does reach it, so without this default the
# configure stops with "Hexagon skel source manifest variables are incomplete".
if(NOT SKEL_CMAKE)
  set(SKEL_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
endif()
if(NOT HEXAGON_SKEL_SOURCE_ID_CONFIG)
  include("${HEXAGON_SKEL_SOURCE_ID_HELPER}")
  get_filename_component(_hexagon_source_tools_version "${HEXAGON_TOOLS_ROOT}" NAME)
  hexagon_skel_source_id_config(
    HEXAGON_SKEL_SOURCE_ID_CONFIG ${_hexagon_source_tools_version})
endif()
if(NOT HEXAGON_SKEL_SOURCE_ID_HEADER)
  set(HEXAGON_SKEL_SOURCE_ID_HEADER "${CMAKE_CURRENT_BINARY_DIR}/hexagon_skel_source_id.h")
  hexagon_skel_source_manifest(HEXAGON_SKEL_SOURCE_INPUTS)
  add_custom_command(
    OUTPUT ${HEXAGON_SKEL_SOURCE_ID_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND}
            -DMNN_OPS_ROOT=${MNN_OPS_ROOT}
            -DSKEL_CMAKE=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
            -DHEXAGON_SKEL_SOURCE_ID_GENERATOR=${HEXAGON_SKEL_SOURCE_ID_GENERATOR}
            -DHEXAGON_SKEL_SOURCE_ID_HELPER=${HEXAGON_SKEL_SOURCE_ID_HELPER}
            -DHEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE=${HEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE}
            -DHEXAGON_SKEL_SOURCE_ID_CONFIG=${HEXAGON_SKEL_SOURCE_ID_CONFIG}
            -DOUTPUT=${HEXAGON_SKEL_SOURCE_ID_HEADER}
            -P ${HEXAGON_SKEL_SOURCE_ID_GENERATOR}
    DEPENDS ${HEXAGON_SKEL_SOURCE_INPUTS}
    COMMENT "Generating the standalone Hexagon skel source provenance ID"
    VERBATIM
  )
  add_custom_target(hexagon_skel_source_id_local DEPENDS ${HEXAGON_SKEL_SOURCE_ID_HEADER})
  set(HEXAGON_SKEL_SOURCE_ID_TARGET hexagon_skel_source_id_local)
else()
  set(HEXAGON_SKEL_SOURCE_ID_TARGET "")
endif()

# The kernels are unusably stack hungry unoptimized: at -O0 a single fp16 branch
# chain allocates a 14 KB frame, which overflows the DSP RPC thread's stack and
# surfaces as a 0x8000040d from execute_command_group.
add_compile_options(-O2)

if(NOT DEFINED HEXAGON_FLATBUFFERS_ROOT)
  set(HEXAGON_FLATBUFFERS_ROOT ${MNN_OPS_ROOT}/flatbuffers)
endif()

include_directories(
  ${MNN_OPS_ROOT}
  ${MNN_OPS_ROOT}/include
  ${MNN_OPS_ROOT}/include/dsp
  ${HEXAGON_FLATBUFFERS_ROOT}/include
  ${HEXAGON_SDK_ROOT}/incs
  ${HEXAGON_SDK_ROOT}/incs/stddef
  ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${SKEL_ARCH}/include
  ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${SKEL_ARCH}/include/qurt
  ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${SKEL_ARCH}/include/posix
  ${IDL_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

set(_dsp ${MNN_OPS_ROOT}/src/dsp)

# C++17: op entry points and the dispatch layer.
set(_skel_cxx
    ${_dsp}/commu.cc
    ${_dsp}/execute_command.cc
    ${_dsp}/blit_ops.cc
    ${_dsp}/tensor_convert.cc
    ${_dsp}/layer_norm_ops.cc
    ${_dsp}/matmul_ops.cc
    ${_dsp}/matmul_q4block_ops.cc
    ${_dsp}/tmac_ops.cc
    ${_dsp}/conv1x1_w8a16_sym_per_channel.cc
    ${_dsp}/im2col_convolution_fp16.cc
    ${_dsp}/pool_ops.cc
    ${_dsp}/conv_depthwise_ops.cc
    ${_dsp}/attention_ops.cc
    ${_dsp}/attention_hmx.cc
    ${_dsp}/attention_hmx_queue.cc
    ${_dsp}/attention_sync_setup.cc
    ${_dsp}/attention_push_kv.cc
    ${_dsp}/attention_sync_process.cc
    ${_dsp}/attention_entry.cc
    ${_dsp}/flash_attention_block_ops.cc
    ${_dsp}/pwl.cc
    ${_dsp}/unary_ops.cc
    ${_dsp}/cast_ops.cc
    ${_dsp}/eltwise_ops.cc
    ${_dsp}/lstm_ops.cc
    ${_dsp}/topk_ops.cc
    ${_dsp}/softmax_ops.cc
    ${_dsp}/loop_ops.cc
    ${_dsp}/rope_ops.cc
    ${_dsp}/scale_ops.cc
    ${_dsp}/shared_gather_ops.cc
    ${_dsp}/power.cc
    ${_dsp}/hmx_mgr.cc
    ${_dsp}/hmx_queue.cc
    ${_dsp}/worker_pool.cc
    ${_dsp}/mmap_mgr.cc
    ${_dsp}/vtcm_mgr.cc
)

# C: the HVX/HMX kernels themselves.
set(_skel_c
    ${_dsp}/ops/matmul_q4fp16.c
    ${_dsp}/ops/matmul_q4fp16_mle32.c
    ${_dsp}/ops/matmul_q4block_fp16_mle32.c
    ${_dsp}/ops/matmul_q4block_gemv_i8.c
    ${_dsp}/ops/matmul_w8a16_gemv_i8.c
    ${_dsp}/ops/tmac_a16w1_fp16.c
    ${_dsp}/ops/pool_fp16.c
    ${_dsp}/ops/depthwise_conv_fp16.c
)

set(HEX_SKEL_LIB hex-htp-${SKEL_ARCH})

set_source_files_properties(${HEXAGON_SKEL_SOURCE_ID_HEADER} PROPERTIES GENERATED TRUE)
add_library(
  ${HEX_SKEL_LIB} SHARED ${_skel_cxx} ${_skel_c} ${IDL_DIR}/htp_ops_skel.c
                           ${HEXAGON_SKEL_SOURCE_ID_HEADER}
)
if(HEXAGON_SKEL_SOURCE_ID_TARGET)
  add_dependencies(${HEX_SKEL_LIB} ${HEXAGON_SKEL_SOURCE_ID_TARGET})
endif()

# Upstream's kernels gate on the arch and on the PWL variant. learned8 is the
# upstream default: it specialises SiLU/MulSiLU, the others keep the companded
# implementation.
target_compile_definitions(
  ${HEX_SKEL_LIB} PRIVATE HTP_OPS_SKEL_ARCH=0x${SKEL_ARCH_NUM}
                          HTP_OPS_PWL_COMPANDED16=1 HTP_OPS_PWL_LEARNED8=1
)

# Upstream silences the vendored sources; keep their warnings out of our build
# output without hiding ours.
set_source_files_properties(
  ${_skel_cxx} ${_skel_c} PROPERTIES COMPILE_OPTIONS "-w"
)

# The dispatch layer is C++, so the skel needs the DSP C++ runtime. The SDK's
# toolchain file defines HEXAGON_LIB_DIR as <tools>/Tools/target/hexagon/lib.
# Link the static archives: the DSP only ships libc.so and libgcc.so, so a
# NEEDED entry for libc++.so.1 makes FastRPC fail the load with AEE_EFAILED.
target_link_libraries(
  ${HEX_SKEL_LIB} ${HEXAGON_LIB_DIR}/${SKEL_ARCH}/G0/pic/libc++.a
  ${HEXAGON_LIB_DIR}/${SKEL_ARCH}/G0/pic/libc++abi.a
)

# The host loads this by name; keep it in step with hexagon_driver.cpp's
# SelectSkel().
set_target_properties(
  ${HEX_SKEL_LIB}
  PROPERTIES PREFIX "lib"
             OUTPUT_NAME "hex-htp-skel-${SKEL_ARCH}"
             SUFFIX ".so"
             LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ship
)

install(TARGETS ${HEX_SKEL_LIB} LIBRARY DESTINATION lib)
