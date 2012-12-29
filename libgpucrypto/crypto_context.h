#ifndef CRYPTO_CONTEXT_H
#define CRYPTO_CONTEXT_H

#include "device_context.h"

typedef struct cuda_stream_s {
	uint8_t *out;
	uint8_t *out_d;
	unsigned long out_len;
} cuda_stream_t;

typedef struct crypto_context_s {
	device_context_t *dev_ctx;
	cuda_stream_t streams[MAX_STREAM + 1];
} crypto_context_t;

/**
 * Constructior.
 *
 * @param dev_ctx Device context pointer.
 * Device context must be initialized before calling this function.
 */
void crypto_context_init_streams();
/**
 * It executes AES-CBC encryption in GPU.
 *
 * It only supports 128-bit key length at the moment.
 *
 * If stream is enabled it will run in non-blocking mode,
 * if not, it will run in blocking mode and the result
 * will be written back to out at the end of function call.
 * This function takes one or more plaintext and
 * returns ciphertext for all of them.
 *
 * @param memory_start Starting point of input data.
 * @param in_pos Offset of plain texts. All plain texts are
 * gathered in to one big buffer with 16-byte align which is
 * AES block size.
 * @param keys_pos Offset of region that stores keys.
 * @param ivs_pos Offset of region that store IVs.
 * @param pkt_offset_pos Offset of region that stores
 * position of the beginning of each plain text.
 * @param tot_in_len Total amount of input data plus meta data.
 * @param out Buffer to store output.
 * @param block_count Total number of blocks in all plain texts.
 * @param num_flows Total number of plain text.
 * @param tot_out_len Total amount of output length.
 * @param stream_id Stream index.
 * @param bits key length for AES cipher
 */
void crypto_context_aes_cbc_encrypt(device_context  *dev_ctx,
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
			const unsigned int   bits);

/**
 * It executes hmac_sha1 in GPU.
 * If stream is enabled it will run in non-blocking mode,
 * if not, it will run in blocking mode and the result
 * will be written back to out at the end of function call.
 * This function takes one or more data  and
 * returns HMAC-SHA1 value for all of them.
 *
 * @param memory_start Starting point of input data.
 * All input data should be be packed in to single continous region
 * before making call to this function.
 * @param in_pos Offset of plain texts.
 * @param keys_pos Offset of region that stores HHAC keys.
 * @param pkt_offset_pos Offset of region that stores
 * position of each plain text.
	 * @param lengths_pos Offset of region that stores length of
 * each plain text.
 * @param data_size Total amount of input data.
 * @param out Buffer to store output.
 * @param num_flows Number of plain texts to be hashed.
 * @param stream_id Stream index.
 */
void crypto_context_hmac_sha1(crypto_context_t *cry_ctx,
			    const void           *memory_start,
			    const unsigned long  in_pos,
			    const unsigned long  keys_pos,
			    const unsigned long  offsets_pos,
			    const unsigned long  lengths_pos,
			    const unsigned long  data_size,
			    unsigned char        *out,
			    const unsigned long  num_flows,
			    const unsigned int   stream_id);
/**
 * Synchronize/query the execution on the stream.
 * This function can be used to check whether the current execution
 * on the stream is finished or also be used to wait until
 * the execution to be finished.
 *
 * @param stream_id Stream index.
 * @param block Wait for the execution to finish or not. true by default.
 * @param copy_result If false, it will not copy result back to CPU.
 *
 * @return true if the current operation on the stream is finished
 * otherwise false.
 */
bool crypto_context_sync(device_context      *dev_ctx,
			const unsigned int  stream_id,
			const bool          block,
			const bool          copy_result);

#endif /*CRYPTO_CONTEXT_H*/
