# Writes a fixed-width provenance header from the checked-in skel inputs.
# The manifest deliberately includes source bytes, the IDL/schema through the
# vendored tree, the recipe, and the canonical build configuration. It does not
# contain timestamps or absolute source paths.

if(NOT HEXAGON_SKEL_SOURCE_ID_HELPER)
  message(FATAL_ERROR "HEXAGON_SKEL_SOURCE_ID_HELPER is required")
endif()
if(NOT HEXAGON_SKEL_SOURCE_ID_CONFIG)
  message(FATAL_ERROR "HEXAGON_SKEL_SOURCE_ID_CONFIG is required")
endif()
include("${HEXAGON_SKEL_SOURCE_ID_HELPER}")
hexagon_skel_source_manifest(_source_files)

set(_manifest
    "executorch-hexagon-skel-source-id-v3\nconfig=${HEXAGON_SKEL_SOURCE_ID_CONFIG}\n")
foreach(_source_file IN LISTS _source_files)
  file(RELATIVE_PATH _relative "${MNN_OPS_ROOT}" "${_source_file}")
  if(
    NOT _relative STREQUAL ".." AND
    NOT _relative MATCHES "^\\.\\./" AND
    NOT IS_ABSOLUTE "${_relative}")
    set(_label "backends/hexagon/third-party/mnn-htp-ops/${_relative}")
  elseif(_source_file STREQUAL "${SKEL_CMAKE}")
    set(_label "backends/hexagon/skel/CMakeLists.txt")
  elseif(_source_file STREQUAL "${HEXAGON_SKEL_SOURCE_ID_GENERATOR}")
    set(_label "backends/hexagon/skel/generate_skel_source_id.cmake")
  elseif(_source_file STREQUAL "${HEXAGON_SKEL_SOURCE_ID_HELPER}")
    set(_label "backends/hexagon/skel/source_id.cmake")
  elseif(_source_file STREQUAL "${HEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE}")
    set(_label "backends/hexagon/CMakeLists.txt")
  else()
    message(FATAL_ERROR "Unclassified Hexagon skel provenance input: ${_source_file}")
  endif()
  file(SHA256 "${_source_file}" _source_hash)
  list(APPEND _entries "${_label} ${_source_hash}")
endforeach()

# Sort by the label rather than by the path the file was found at: the identity
# is a property of the inputs, not of the directory they were checked out into.
list(SORT _entries)
foreach(_entry IN LISTS _entries)
  string(APPEND _manifest "${_entry}\n")
endforeach()

string(SHA256 _source_id "${_manifest}")

# The RPC carries four uint32s, so the identity the host compares and the one
# it prints are the same 128 bits: four words keeps an accidental collision far
# below the odds of the same skel source being built twice.
string(SUBSTRING "${_source_id}" 0 8 _source_id_0)
string(SUBSTRING "${_source_id}" 8 8 _source_id_1)
string(SUBSTRING "${_source_id}" 16 8 _source_id_2)
string(SUBSTRING "${_source_id}" 24 8 _source_id_3)
string(SUBSTRING "${_source_id}" 0 32 _source_id_text)
file(WRITE "${OUTPUT}"
"#ifndef EXECUTORCH_HEXAGON_SKEL_SOURCE_ID_H\n#define EXECUTORCH_HEXAGON_SKEL_SOURCE_ID_H\n\n#define HEXAGON_SKEL_SOURCE_ID_TEXT \"${_source_id_text}\"\n#define HEXAGON_SKEL_SOURCE_ID_0 0x${_source_id_0}u\n#define HEXAGON_SKEL_SOURCE_ID_1 0x${_source_id_1}u\n#define HEXAGON_SKEL_SOURCE_ID_2 0x${_source_id_2}u\n#define HEXAGON_SKEL_SOURCE_ID_3 0x${_source_id_3}u\n\n#endif\n")
message(STATUS "Hexagon skel source ID: ${_source_id}")
