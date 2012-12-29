#include <assert.h>
#include <cutil_inline.h>

#include "crypto_context.h"
#include "aes_kernel.h"
#include "sha1_kernel.h"

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLK 256 // in order to load t box into shared memory on parallel

void crypto_context_init(cyrpto_context_t *crypto_ctx)
{
	for (unsigned i = 0; i <= MAX_STREAM; i++) {
		crypto_ctx->streams[i].out = 0;
		crypto_ctx->streams[i].out_d = 0;
		crypto_ctx->streams[i].out_len = 0;
	}
}

void crypto_context_aes_cbc_encrypt(crypto_context_t *cry_ctx,
			const void	     *memory_start,
			const void	     *memory_d,
			const unsigned long  in_pos,
			const unsigned long  keys_pos,
			const unsigned long  ivs_pos,
			const unsigned long  pkt_offset_pos,
			const unsigned long  tot_in_len,
			unsigned char        *out,
			const unsigned long  num_flows,
			const unsigned long  tot_out_len,
			const unsigned int   stream_id,
			const unsigned int   bits)
{
	assert(bits == 128);
	assert(cry_ctx->dev_ctx->get_state(stream_id) == READY);
	cry_ctx->dev_ctx->set_state(stream_id, WAIT_KERNEL);

	/* Allocate memory on device */
	uint8_t *in_d;
	uint8_t *keys_d;
	uint8_t *ivs_d;
	uint32_t *pkt_offset_d;
	void *memory_d;

	//Calculate # of cuda blks required
	unsigned int  threads_per_blk = THREADS_PER_BLK;
	int           num_blks        = (num_flows + threads_per_blk - 1) / threads_per_blk;

	cudaMemcpyAsync(memory_d, memory_start, tot_in_len,
			cudaMemcpyHostToDevice, cry_ctx->dev_ctx->get_stream(stream_id));

	cry_ctx->dev_ctx->clear_checkbits(stream_id, num_blks);

	in_d         = (uint8_t *) memory_d + in_pos;
	keys_d       = (uint8_t *) memory_d + keys_pos;
	ivs_d        = (uint8_t *) memory_d + ivs_pos;
	pkt_offset_d = (uint32_t *) ((uint8_t *)memory_d + pkt_offset_pos);

	/* Call cbc kernel function to do encryption */
	if (cry_ctx->dev_ctx->use_stream()) {
		AES_cbc_128_encrypt_gpu(in_d,
					cry_ctx->streams[stream_id].out_d,
					pkt_offset_d,
					keys_d,
					ivs_d,
					num_flows,
					cry_ctx->dev_ctx->get_dev_checkbits(stream_id),
					threads_per_blk,
					cry_ctx->dev_ctx->get_stream(stream_id));
	} else {
		AES_cbc_128_encrypt_gpu(in_d,
					cry_ctx->streams[stream_id].out_d,
					pkt_offset_d,
					keys_d,
					ivs_d,
					num_flows,
					cry_ctx->dev_ctx->get_dev_checkbits(stream_id),
					threads_per_blk,
					0);

	}

	assert(cudaGetLastError() == cudaSuccess);

	cry_ctx->streams[stream_id].out     = out;
	cry_ctx->streams[stream_id].out_len = tot_out_len;

	/* Copy data back from device to host */
	if (!cry_ctx->dev_ctx->use_stream()) {
		sync(stream_id);
	}
}

void crypto_context_hmac_sha1(crypto_context_t *cry_ctx,
			    const void           *memory_start,
			    const unsigned long  in_pos,
			    const unsigned long  keys_pos,
			    const unsigned long  offsets_pos,
			    const unsigned long  lengths_pos,
			    const unsigned long  data_size,
			    unsigned char        *out,
			    const unsigned long  num_flows,
			    const unsigned int   stream_id)
{
	assert(dev_ctx_->get_state(stream_id) == READY);
	dev_ctx_->set_state(stream_id, WAIT_KERNEL);

	//copy input data
	cudaMemcpyAsync(memory_d,
			memory_start,
			data_size,
			cudaMemcpyHostToDevice,
			dev_ctx_->get_stream(stream_id));

	//variables need for kernel launch
	int threads_per_blk = SHA1_THREADS_PER_BLK;
	int num_blks = (num_flows+threads_per_blk-1)/threads_per_blk;

	//allocate buffer for output
	uint32_t *out_d = (uint32_t *)pool->alloc(20 * num_flows);

	//initialize input memory offset in device memory
	char     *in_d         = (char *)memory_d + in_pos;
	char     *keys_d       = (char *)memory_d + keys_pos;
	uint32_t *pkt_offset_d = (uint32_t *)((uint8_t *)memory_d + offsets_pos);
	uint16_t *lengths_d    = (uint16_t *)((uint8_t *)memory_d + lengths_pos);

	//clear checkbits before kernel execution
	dev_ctx_->clear_checkbits(stream_id, num_blks);

	if (dev_ctx_->use_stream() && stream_id > 0) {	//with stream
		hmac_sha1_gpu(in_d,
			      keys_d,
			      pkt_offset_d,
			      lengths_d,
			      out_d,
			      num_flows,
			      dev_ctx_->get_dev_checkbits(stream_id),
			      threads_per_blk,
			      dev_ctx_->get_stream(stream_id));
	} else  if (!dev_ctx_->use_stream() && stream_id == 0) {//w/o stream
		hmac_sha1_gpu(in_d,
			      keys_d,
			      pkt_offset_d,
			      lengths_d,
			      out_d,
			      num_flows,
			      dev_ctx_->get_dev_checkbits(stream_id),
			      SHA1_THREADS_PER_BLK,
			      0);
	} else {
		assert(0);
	}

	assert(cudaGetLastError() == cudaSuccess);

	streams[stream_id].out_d   = (uint8_t*)out_d;
	streams[stream_id].out     = out;
	streams[stream_id].out_len = 20 * num_flows;

	//if stream is not used then sync (assuming blocking mode)
	if (dev_ctx_->use_stream() && stream_id == 0) {
		sync(stream_id);
	}
}

bool crypto_context_sync(cyrpto_context_t   *cry_ctx,
			const unsigned int  stream_id,
			const bool          block,
			const bool          copy_result)
{
        if (block) {
		cry_ctx->dev_ctx->sync(stream_id, true);
		if (copy_result && cry_ctx->dev_ctx->get_state(stream_id) == WAIT_KERNEL) {
			cutilSafeCall(cudaMemcpyAsync(cry_ctx->streams[stream_id].out,
						      cry_ctx->streams[stream_id].out_d,
						      cry_ctx->streams[stream_id].out_len,
						      cudaMemcpyDeviceToHost,
						      cry_ctx->dev_ctx->get_stream(stream_id)));
			cry_ctx->dev_ctx->set_state(stream_id, WAIT_COPY);
			cry_ctx->dev_ctx->sync(stream_id, true);
			cry_ctx->dev_ctx->set_state(stream_id, READY);
		} else if (cry_ctx->dev_ctx->get_state(stream_id) == WAIT_COPY) {
			cry_ctx->dev_ctx->sync(stream_id, true);
			cry_ctx->dev_ctx->set_state(stream_id, READY);
		}
		return true;
	} else {
		if (!cry_ctx->dev_ctx->sync(stream_id, false))
			return false;

		if (cry_ctx->dev_ctx->get_state(stream_id) == WAIT_KERNEL) {
			//if no need for data copy
			if (!copy_result) {
				cry_ctx->dev_ctx->set_state(stream_id, READY);
				return true;
			}

			cutilSafeCall(cudaMemcpyAsync(cry_ctx->streams[stream_id].out,
						      cry_ctx->streams[stream_id].out_d,
						      cry_ctx->streams[stream_id].out_len,
						      cudaMemcpyDeviceToHost,
						      cry_ctx->dev_ctx->get_stream(stream_id)));
			cry_ctx->dev_ctx->set_state(stream_id, WAIT_COPY);

		} else if (cry_ctx->dev_ctx->get_state(stream_id) == WAIT_COPY) {
			cry_ctx->dev_ctx->set_state(stream_id, READY);
			return true;
		} else if (cry_ctx->dev_ctx->get_state(stream_id) == READY) {
			return true;
		} else {
			assert(0);
		}
	}
        return false;
}

