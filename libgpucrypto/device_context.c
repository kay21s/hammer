#include <sys/time.h>
#include <cutil_inline.h>
#include <assert.h>

#include "device_context.h"

static uint64_t get_now() {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
};

uint64_t device_context_get_elapsed_time(device_context_t dc, const unsigned stream_id)
{
	assert(0 <= stream_id && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	return dc->stream_ctx[stream_id].end_usec - dc->stream_ctx[stream_id].begin_usec;
}

void device_context_free(device_context_t *dc)
{
	for (unsigned i = 1; i <= dc->nstream; i++) {
		cutilSafeCall(cudaStreamDestroy(dc->stream_ctx[i].stream));
		cutilSafeCall(cudaFreeHost(dc->stream_ctx[i].checkbits));
		dc->stream_ctx[i].pool.destroy();
	}

	if (dc->nstream == 0) {
		cutilSafeCall(cudaFreeHost((void*)dc->stream_ctx[0].checkbits));
		dc->stream_ctx[0].pool.destroy();
	}
}

bool device_context_init(device_context_t *dc, const unsigned nstream)
{
	void *ret = NULL;
	assert(nstream >= 0 && nstream <= MAX_STREAM);

	dc->nstream = nstream;

	if (nstream > 0) {
		for (unsigned i = 1; i <= nstream; i++) {
			cutilSafeCall(cudaStreamCreate(&(dc->stream_ctx[i].stream)));
			dc->stream_ctx[i].state = READY;

			cutilSafeCall(cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped));
			dc->stream_ctx[i].checkbits = (uint8_t*)ret;
			cutilSafeCall(cudaHostGetDevicePointer((void **)&(dc->stream_ctx[i].checkbits_d), ret, 0));
		}
	} else {
		dc->stream_ctx[0].stream = 0;
		dc->stream_ctx[0].state = READY;

		cutilSafeCall(cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped));
		dc->stream_ctx[0].checkbits = (uint8_t*)ret;
		cutilSafeCall(cudaHostGetDevicePointer((void **)&(dc->stream_ctx[0].checkbits_d), ret, 0));
	}
	return true;
}

bool device_context_sync(device_context_t *dc, const unsigned stream_id, const bool block)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	if (!block) {
		if (dc->stream_ctx[stream_id].finished)
			return true;

		if (dc->stream_ctx[stream_id].state == WAIT_KERNEL && dc->stream_ctx[stream_id].num_blks > 0) {
			volatile uint8_t *checkbits = dc->stream_ctx[stream_id].checkbits;
			for (unsigned i = 0; i < dc->stream_ctx[stream_id].num_blks; i++) {
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

void device_context_set_state(device_context_t *dc, const unsigned stream_id, const STATE state)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	if (state == READY) {
		dc->stream_ctx[stream_id].end_usec = get_now();
		dc->stream_ctx[stream_id].num_blks = 0;
		dc->stream_ctx[stream_id].pool.reset();
	} else if (state == WAIT_KERNEL) {
		dc->stream_ctx[stream_id].begin_usec = get_now();
		dc->stream_ctx[stream_id].finished = false;
	} else if (state == WAIT_COPY) {
		dc->stream_ctx[stream_id].finished = false;
	}
	dc->stream_ctx[stream_id].state = state;
}

enum STATE device_context_get_state(device_context_t *dc, const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));

	return dc->stream_ctx[stream_id].state;
}

bool device_context_use_stream(device_context_t *dc)
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
	assert(stream_id >= 0 && stream_id <= dc->nstream);
	assert((stream_id == 0) ^ (dc->nstream > 0));
	assert(num_blks >= 0 && num_blks <= MAX_BLOCKS);

	dc->stream_ctx[stream_id].num_blks = num_blks;
	volatile uint8_t *checkbits = dc->stream_ctx[stream_id].checkbits;
	for (unsigned i = 0; i < num_blks; i++)
		checkbits[i] = 0;
	dc->stream_ctx[stream_id].finished = false;
}
