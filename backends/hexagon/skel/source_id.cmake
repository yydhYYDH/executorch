# The source identity is a source-closure fingerprint, not a human-maintained
# version or a full toolchain attestation. The manifest is deliberately
# conservative: it includes every file in the vendored DSP tree and the recipe
# files, so a source-only reintroduction of the VLA changes the ID even when
# the revision string does not.

function(hexagon_skel_source_id_config out_config toolchain)
  if(NOT toolchain)
    set(toolchain unspecified)
  endif()
  set(${out_config}
      "arch-independent|c-standard=11|cxx-standard=17|pic=ON|opt=-O2|pwl=companded16,learned8|toolchain=${toolchain}"
      PARENT_SCOPE)
endfunction()

function(hexagon_skel_source_manifest out_files)
  if(NOT MNN_OPS_ROOT OR NOT SKEL_CMAKE OR
     NOT HEXAGON_SKEL_SOURCE_ID_GENERATOR OR
     NOT HEXAGON_SKEL_SOURCE_ID_HELPER OR
     NOT HEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE)
    message(FATAL_ERROR "Hexagon skel source manifest variables are incomplete")
  endif()

  if(CMAKE_SCRIPT_MODE_FILE)
    file(GLOB_RECURSE _mnn_files LIST_DIRECTORIES false "${MNN_OPS_ROOT}/*")
  else()
    file(GLOB_RECURSE _mnn_files CONFIGURE_DEPENDS LIST_DIRECTORIES false "${MNN_OPS_ROOT}/*")
  endif()
  list(SORT _mnn_files)
  set(_files
      ${_mnn_files}
      ${SKEL_CMAKE}
      ${HEXAGON_SKEL_SOURCE_ID_GENERATOR}
      ${HEXAGON_SKEL_SOURCE_ID_HELPER}
      ${HEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE})
  list(REMOVE_DUPLICATES _files)
  list(SORT _files)
  set(${out_files} "${_files}" PARENT_SCOPE)
endfunction()
