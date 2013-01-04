#include <assert.h>
#include <cutil_inline.h>

#include "crypto_context.h"
#include "crypto_kernel.h"

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLK 256 // in order to load t box into shared memory on parallel

void crypto_context_init(cyrpto_context_t *cry_ctx, uint32_t input_size, 
			uint32_t output_size, uint32_t stream_num)
{
	int i;

	cry_ctx->dev_ctx = (device_context_t *)malloc(sizeof(device_context_t));
	device_context_init(cry_ctx->dev_ctx, stream_num);

	cry_ctx->stream_set = (gpu_stream_t *)malloc(stream_num * sizeof(gpu_stream_t));

	for (i = 0; i < config->cpu_worker_num; i ++) {
		/* input buffer need to store key,iv, and offsets info, 
		 * while output buffer doesn't need these */
		cry_ctx->streams[i].input_d = cuda_device_mem_alloc(input_size);
		cry_ctx->streams[i].output_d = cuda_device_mem_alloc(output_size);
		crypto_ctx->streams[i].out_len = 0;
	}

	return;
	/*
	for (unsigned i = 0; i <= MAX_STREAM; i++) {
		crypto_ctx->streams[i].intput_d = 0;
		crypto_ctx->streams[i].output_d = 0;
		crypto_ctx->streams[i].out_len = 0;
	}*/
}

void crypto_context_sha1_aes_encrypt(crypto_context_t *cry_ctx,
			const void	     *input_start,
			void		     *output_start,
			const unsigned long  in_pos,
			const unsigned long  aes_keys_pos,
			const unsigned long  ivs_pos,
			const unsigned long  hmac_keys_pos,
			const unsigned long  pkt_offset_pos,
			const unsigned long  actual_length_pos,
			const unsigned long  tot_in_len,
			const unsigned long  tot_out_len,
			const unsigned long  num_flows,
			const unsigned int   stream_id,
			const unsigned int   bits)
{
	assert(bits == 128);
	device_context_t *dev_ctx = cry_ctx->dev_ctx;
	assert(device_context_get_state(dev_ctx, stream_id) == READY);
	device_context_set_state(dev_ctx, stream_id, WAIT_KERNEL);
	gpu_stream_t stream = cry_ctx->streams[stream_id];

	/* Allocate memory on device */
	uint8_t *in_d;
	uint8_t *aes_keys_d;
	uint8_t *ivs_d;
	uint8_t *hmac_keys_d;
	uint32_t *pkt_offset_d;
	uint16_t *actual_length_d;
	void *input_d = stream.input_d;

	/* Calculate # of cuda blks required */
	unsigned int  threads_per_blk = THREADS_PER_BLK;
	int           num_blks        = (num_flows + threads_per_blk - 1) / threads_per_blk;

	cudaMemcpyAsync(input_d, input_start, tot_in_len,
			cudaMemcpyHostToDevice, device_context_get_stream(dev_ctx, stream_id));

	device_context_clear_checkbits(dev_ctx, stream_id, num_blks);

	in_d         = (uint8_t *) input_d + in_pos;
	aes_keys_d       = (uint8_t *) input_d + aes_keys_pos;
	ivs_d        = (uint8_t *) input_d + ivs_pos;
	hmac_keys_d = (uint8_t *)input_d + hmac_keys_pos;
	pkt_offset_d = (uint32_t *) ((uint8_t *) input_d + pkt_offset_pos);
	actual_length_d = (uint16_t *) ((uint8_t *) input_d + actual_length_pos);

	/* Call cbc kernel function to do encryption */
	if (device_context_use_stream(dev_ctx)) {
		HMAC_AES_together_gpu(  in_d,
					cry_ctx->streams[stream_id].output_d,
					keys_d,
					ivs_d,
					hmac_keys_d,
					pkt_offset_d,
					actual_length_d,
					num_flows,
					device_context_get_dev_checkbits(dev_ctx, stream_id),
					threads_per_blk,
					device_context_get_stream(dev_ctx, stream_id));
	} else {
		HMAC_AES_together_gpu(  in_d,
					cry_ctx->streams[stream_id].output_d,
					keys_d,
					ivs_d,
					hmac_keys_d,
					pkt_offset_d,
					actual_length_d,
					num_flows,
					device_context_get_dev_checkbits(dev_ctx, stream_id),
					threads_per_blk,
					0);
	}

	assert(cudaGetLastError() == cudaSuccess);

	cry_ctx->streams[stream_id].out_len = tot_out_len;

	/* Copy data back from device to host */
	if (!device_context_use_stream(dev_ctx)) {
		crypto_context_sync(cry_ctx, stream_id, output_start, 1, 1);
	}
}

bool crypto_context_sync(cyrpto_context_t   *cry_ctx,
			const unsigned int  stream_id,
			void 		    *output_start
			const bool          block,
			const bool          copy_result)
{
        if (block) {
		device_context_sync(stream_id, true);
		if (copy_result && device_contex_get_state(dev_ctx, stream_id) == WAIT_KERNEL) {
			cutilSafeCall(cudaMemcpyAsync(output_start,
						      cry_ctx->streams[stream_id].output_d,
						      cry_ctx->streams[stream_id].out_len,
						      cudaMemcpyDeviceToHost,
						      device_context_get_stream(dev_ctx, stream_id)));
			device_context_set_state(dev_ctx, stream_id, WAIT_COPY);
			device_context_sync(dev_ctx, stream_id, true);
			device_context_set_state(dev_ctx, stream_id, READY);
		} else if (device_context_get_state(dev_ctx, stream_id) == WAIT_COPY) {
			device_context_sync(dev_ctx, stream_id, true);
			device_context_set_state(dev_ctx, stream_id, READY);
		}
		return true;
	} else {
		if (!device_context_sync(dev_ctx, stream_id, false))
			return false;

		if (device_context_get_state(dev_ctx, stream_id) == WAIT_KERNEL) {
			//if no need for data copy
			if (!copy_result) {
				device_context_set_state(dev_ctx, stream_id, READY);
				return true;
			}

			cutilSafeCall(cudaMemcpyAsync(output_start,
						      cry_ctx->streams[stream_id].output_d,
						      cry_ctx->streams[stream_id].out_len,
						      cudaMemcpyDeviceToHost,
						      device_context_get_stream(stream_id)));
			device_context_set_state(dev_ctx, stream_id, WAIT_COPY);

		} else if (device_context_get_state(dev_ctx, stream_id) == WAIT_COPY) {
			device_context_set_stat(dev_ctx, stream_id, READY);
			return true;
		} else if (device_context_get_state(dev_ctx, stream_id) == READY) {
			return true;
		} else {
			assert(0);
		}
	}
        return false;
}

void crypto_context_aes_cbc_encrypt(crypto_context_t *cry_ctx,
			const void	     *input_start,
			void		     *output_start,
			const unsigned long  in_pos,
			const unsigned long  keys_pos,
			const unsigned long  ivs_pos,
			const unsigned long  pkt_offset_pos,
			const unsigned long  tot_in_len,
			const unsigned long  num_flows,
			const unsigned long  tot_out_len,
			const unsigned int   stream_id,
			const unsigned int   bits)
{
	assert(bits == 128);
	device_context_t *dev_ctx = cry_ctx->dev_ctx;
	assert(device_context_get_state(dev_ctx, stream_id) == READY);
	device_context_set_state(dev_ctx, stream_id, WAIT_KERNEL);
	gpu_stream_t stream = cry_ctx->streams[stream_id];

	/* Allocate memory on device */
	uint8_t *in_d;
	uint8_t *keys_d;
	uint8_t *ivs_d;
	uint32_t *pkt_offset_d;
	void *input_d = stream.input_d;

	//Calculate # of cuda blks required
	unsigned int  threads_per_blk = THREADS_PER_BLK;
	int           num_blks        = (num_flows + threads_per_blk - 1) / threads_per_blk;

	cudaMemcpyAsync(input_d, input_start, tot_in_len,
			cudaMemcpyHostToDevice, device_context_get_stream(dev_ctx, stream_id));

	device_context_clear_checkbits(dev_ctx, stream_id, num_blks);

	in_d         = (uint8_t *) input_d + in_pos;
	keys_d       = (uint8_t *) input_d + keys_pos;
	ivs_d        = (uint8_t *) input_d + ivs_pos;
	pkt_offset_d = (uint32_t *) ((uint8_t *)input_d + pkt_offset_pos);

	/* Call cbc kernel function to do encryption */
	if (device_context_use_stream(dev_ctx)) {
		AES_cbc_128_encrypt_gpu(in_d,
					cry_ctx->streams[stream_id].output_d,
					pkt_offset_d,
					keys_d,
					ivs_d,
					num_flows,
					device_context_get_dev_checkbits(dev_ctx, stream_id),
					threads_per_blk,
					device_context_get_stream(dev_ctx, stream_id));
	} else {
		AES_cbc_128_encrypt_gpu(in_d,
					cry_ctx->streams[stream_id].output_d,
					pkt_offset_d,
					keys_d,
					ivs_d,
					num_flows,
					device_context_get_dev_checkbits(dev_ctx, stream_id),
					threads_per_blk,
					0);

	}

	assert(cudaGetLastError() == cudaSuccess);

	cry_ctx->streams[stream_id].out_len = tot_out_len;

	/* Copy data back from device to host */
	if (!device_context_use_stream(dev_ctx)) {
		crypto_context_sync(cry_ctx, stream_id, output_start, 1, 1);
	}
}

void crypto_context_hmac_sha1(crypto_context_t *cry_ctx,
				const void           *input_start,
				void		     *output_start,
				const unsigned long  in_pos,
				const unsigned long  keys_pos,
				const unsigned long  offsets_pos,
				const unsigned long  lengths_pos,
				const unsigned long  data_size,
				unsigned char        *out,
				const unsigned long  num_flows,
				const unsigned int   stream_id)
{
	device_context_t *dev_ctx = cry_ctx->dev_ctx;
	assert(device_context_get_state(dev_ctx, stream_id) == READY);
	device_context_set_state(dev_ctx, stream_id, WAIT_KERNEL);
	gpu_stream_t stream = cry_ctx->streams[stream_id];
	void *input_d = stream.input_d;

	//copy input data
	cudaMemcpyAsync(input_d, input_start, data_size,
			cudaMemcpyHostToDevice, device_context_get_stream(dev_ctx, stream_id));

	//variables need for kernel launch
	int threads_per_blk = SHA1_THREADS_PER_BLK;
	int num_blks = (num_flows+threads_per_blk-1)/threads_per_blk;

	//initialize input memory offset in device memory
	char     *in_d         = (char *)input_d + in_pos;
	char     *keys_d       = (char *)input_d + keys_pos;
	uint32_t *pkt_offset_d = (uint32_t *)((uint8_t *)input_d + offsets_pos);
	uint16_t *lengths_d    = (uint16_t *)((uint8_t *)input_d + lengths_pos);

	//clear checkbits before kernel execution
	device_context_clear_checkbits(dev_ctx, stream_id, num_blks);

	if (device_context_use_stream(dev_ctx) && stream_id > 0) {	//with stream
		hmac_sha1_gpu(in_d,
			      keys_d,
			      pkt_offset_d,
			      lengths_d,
			      cry_ctx->streams[stream_id].output_d,
			      num_flows,
			      device_context_get_dev_checkbits(dev_ctx, stream_id),
			      SHA1_THREADS_PER_BLK,
			      device_context_get_stream(dev_ctx, stream_id));
	} else  if (!device_context_use_stream(dev_ctx) && stream_id == 0) {//w/o stream
		hmac_sha1_gpu(in_d,
			      keys_d,
			      pkt_offset_d,
			      lengths_d,
			      cry_ctx->streams[stream_id].output_d,
			      num_flows,
			      device_context_get_dev_checkbits(dev_ctx, stream_id),
			      SHA1_THREADS_PER_BLK,
			      0);
	} else {
		assert(0);
	}

	assert(cudaGetLastError() == cudaSuccess);

	streams[stream_id].out_len = 20 * num_flows;

	//if stream is not used then sync (assuming blocking mode)
	if (device_context_use_stream(dev_ctx) && stream_id == 0) {
		crypto_context_sync(cry_ctx, stream_id, output_start, 1, 1);
	}
}

