#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vtcm_manager_setup();
void vtcm_manager_reset();
int vtcm_manager_acquire();
void vtcm_manager_release();
int vtcm_manager_is_acquired();
int vtcm_manager_needs_release();

void *vtcm_manager_get_vtcm_base();
unsigned int vtcm_manager_get_vtcm_size();
// One past the last byte the sequential allocator may hand out: the top of
// VTCM minus whatever the reserve path has already carved off it.
void *vtcm_manager_get_vtcm_alloc_end();
int vtcm_manager_get_ctx_id();

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment);
void *vtcm_manager_query_area(const char *name);

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  // align up to 128 bytes for DMA and HVX requirements
  size_t aligned_size = (size + 127) & ~127;
  uint8_t *p = *vtcm_ptr;
  uint8_t *end = (uint8_t *)vtcm_manager_get_vtcm_alloc_end();
  if (end != NULL && p >= (uint8_t *)vtcm_manager_get_vtcm_base() && (size_t)(end - p) < aligned_size) {
    // Writing past the VTCM this manager owns corrupts the following area
    // instead of failing, so callers have to be able to see the failure.
    return NULL;
  }
  *vtcm_ptr += aligned_size;
  return p;
}

#ifdef __cplusplus
}
#endif
