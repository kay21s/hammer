#ifndef CUDA_MEM_H
#define CUDA_MEM_H

void *cuda_mem_alloc(unsigned long size);
void cuda_mem_free(uint8_t *mem);
void *cuda_pinned_mem_alloc(unsigned long size);
void cuda_pinned_mem_free(uint8_t *mem);

#endif
