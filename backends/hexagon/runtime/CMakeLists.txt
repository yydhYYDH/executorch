# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Host side of the Hexagon backend: the FastRPC session and the BackendInterface
# implementation that ExecuTorch registers.

add_library(
  hexagon_backend STATIC hexagon_driver.cpp hexagon_backend.cpp
                        HexagonBackendOptions.cpp
)

# BUILD_INTERFACE keeps source- and build-tree paths out of the exported
# interface; the headers here are build inputs, not part of an installed SDK.
target_include_directories(
  hexagon_backend
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${HEXAGON_IDL_DIR}>
         $<BUILD_INTERFACE:${HEXAGON_SCHEMA_DIR}>
         $<BUILD_INTERFACE:${HEXAGON_MNN_OPS_ROOT}>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
         ${HEXAGON_SDK_ROOT}/incs
         ${HEXAGON_SDK_ROOT}/incs/stddef
         ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
         ${HEXAGON_SDK_ROOT}/utils/examples
)

# The stub turns htp_ops_* into FastRPC invokes. Building it from the same IDL
# the skel is built from is what keeps the handle numbers in sync.
target_sources(
  hexagon_backend
  PRIVATE ${HEXAGON_IDL_DIR}/htp_ops_stub.c
          ${HEXAGON_IDL_DIR}/hexagon_skel_source_id.h
          ${CMAKE_CURRENT_SOURCE_DIR}/hexagon_skel_provenance.h
          ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
)

add_dependencies(hexagon_backend hexagon_idl hexagon_schema hexagon_skel_source_id)

# libcdsprpc provides the remote_handle64_* entry points the qaic stub calls,
# rpcmem_alloc and fastrpc_mmap. Its cache maintenance entry points are not in
# the link-time stub, so hexagon_driver resolves those through dlopen.
target_link_libraries(
  hexagon_backend PUBLIC executorch ${HEXAGON_DSPRPC_LIB} ${CMAKE_DL_LIBS}
)

# The backend registers itself from a static initializer and nothing references
# it, so a consumer that links this archive the ordinary way drops the object and
# gets an unregistered backend at run time rather than a link error. For an
# archive this emits whole-archive retention into every consumer, the same call
# backends/xnnpack and backends/vulkan make.
executorch_target_link_options_shared_lib(hexagon_backend)

# executorch_backends is exported and links this target, so leaving it out of
# the export set fails the generate step rather than the build.
install(
  TARGETS hexagon_backend
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
