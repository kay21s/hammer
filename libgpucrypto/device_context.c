#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>

#include "device_context.h"

static uint64_t get_now() {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
};

uint64_t device_context_get_elapsed_time(device_context_t *dc, const unsigned stream_id)
{
	assert(0 <= stream_id && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	return dc->stream_ctx[stream_id].end_usec - dc->stream_ctx[stream_id].begin_usec;
}

void device_context_free(device_context_t *dc)
{
	uint8_t i;
	for (i = 1; i <= dc->nstream; i++) {
		cudaStreamDestroy(dc->stream_ctx[i].stream);
		cudaFreeHost(dc->stream_ctx[i].checkbits);
	}

	if (dc->nstream == 0) {
		cudaFreeHost((void*)dc->stream_ctx[0].checkbits);
	}
}

uint8_t device_context_init(device_context_t *dc, const unsigned nstream)
{
	void *ret = NULL;
	uint8_t i;

	assert(nstream >= 0 && nstream <= MAX_STREAM);

	dc->nstream = nstream;

	if (nstream > 0) {
		for (i = 1; i <= nstream; i++) {
			cudaStreamCreate(&(dc->stream_ctx[i].stream));
			dc->stream_ctx[i].state = READY;

			cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped);
			dc->stream_ctx[i].checkbits = (uint8_t*)ret;
			cudaHostGetDevicePointer((void **)&(dc->stream_ctx[i].checkbits_d), ret, 0);
		}
	} else {
		dc->stream_ctx[0].stream = 0;
		dc->stream_ctx[0].state = READY;

		cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped);
		dc->stream_ctx[0].checkbits = (uint8_t*)ret;
		cudaHostGetDevicePointer((void **)&(dc->stream_ctx[0].checkbits_d), ret, 0);
	}
	return true;
}

uint8_t device_context_sync(device_context_t *dc, const unsigned stream_id, const uint8_t block)
{
	uint8_t i;
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	if (!block) {
		if (dc->stream_ctx[stream_id].finished)
			return true;

		if (dc->stream_ctx[stream_id].state == WAIT_KERNEL && dc->stream_ctx[stream_id].num_blks > 0) {
			volatile uint8_t *checkbits = dc->stream_ctx[stream_id].checkbits;
			for (i = 0; i < dc->stream_ctx[stream_id].num_blks; i++) {
				if (checkbits[i] == 0)
					return false;
			}
		} else if (dc->stream_ctx[stream_id].state != READY) {
			cudaError_t ret = cudaStreamQuery(dc->stream_ctx[stream_id].stream);
			if (ret == cudaErrorNotReady)
				return false;
			assert(ret == cudaSuccess);
		}
		dc->stream_ctx[stream_id].finished = true;
	} else {
		cudaStreamSynchronize(dc->stream_ctx[stream_id].stream);
	}

	return true;
}

void device_context_set_state(device_context_t *dc, const unsigned stream_id, const enum state state)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	if (state == READY) {
		dc->stream_ctx[stream_id].end_usec = get_now();
		dc->stream_ctx[stream_id].num_blks = 0;
	} else if (state == WAIT_KERNEL) {
		dc->stream_ctx[stream_id].begin_usec = get_now();
		dc->stream_ctx[stream_id].finished = false;
	} else if (state == WAIT_COPY) {
		dc->stream_ctx[stream_id].finished = false;
	}
	dc->stream_ctx[stream_id].state = state;
}

enum state device_context_get_state(device_context_t *dc, const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	return dc->stream_ctx[stream_id].state;
}

uint8_t device_context_use_stream(device_context_t *dc)
{
	return (dc->nstream != 0);
}

cudaStream_t device_context_get_stream(device_context_t *dc, const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	return dc->stream_ctx[stream_id].stream;
}

uint8_t *device_context_get_dev_checkbits(device_context_t *dc, const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	return dc->stream_ctx[stream_id].checkbits_d;
}
void device_context_clear_checkbits(device_context_t *dc, const unsigned stream_id, const unsigned num_blks)
{
	uint8_t i;
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	assert(num_blks >= 0 && num_blks <= MAX_BLOCKS);

	dc->stream_ctx[stream_id].num_blks = num_blks;
	volatile uint8_t *checkbits = dc->stream_ctx[stream_id].checkbits;
	for (i = 0; i < num_blks; i++)
		checkbits[i] = 0;
	dc->stream_ctx[stream_id].finished = false;
}
