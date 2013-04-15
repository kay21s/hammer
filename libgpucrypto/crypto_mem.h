#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#include <stdint.h>

void *cuda_device_mem_alloc(unsigned long size);
void cuda_device_mem_free(uint8_t *mem);
void *cuda_pinned_mem_alloc(unsigned long size);
void cuda_pinned_mem_free(uint8_t *mem);

#endif
