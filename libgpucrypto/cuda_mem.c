#include <cuda_runtime.h>
#include <cutil_inline.h>

void *cuda_device_mem_alloc(unsigned long size)
{
	void *mem;
        cutilSafeCall(cudaMalloc(&mem, size));
        return mem;
}


void cuda_device_mem_free(uint8_t *mem)
{
	if (mem) {
		cutilSafeCall(cudaFree(mem));
		mem = NULL;
	}
}

void *cuda_pinned_mem_alloc(unsigned long size)
{
	void *mem;
	cutilSafeCall(cudaHostAlloc(&mem, size, cudaHostAllocPortable));
	return mem;
}

void cuda_pinned_mem_free(uint8_t *mem)
{
	if (mem) {
		cutilSafeCall(cudaFreeHost(mem));
		mem = NULL;
	}
}
