#include <stdint.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>

void *cuda_device_mem_alloc(unsigned long size)
{
	void *mem;
	cudaMalloc(&mem, size);
	return mem;
}


void cuda_device_mem_free(uint8_t *mem)
{
	if (mem) {
		cudaFree(mem);
		mem = NULL;
	}
}

void *cuda_pinned_mem_alloc(unsigned long size)
{
	void *mem;
	cudaHostAlloc(&mem, size, cudaHostAllocPortable);
	return mem;
}

void cuda_pinned_mem_free(uint8_t *mem)
{
	if (mem) {
		cudaFreeHost(mem);
		mem = NULL;
	}
}
