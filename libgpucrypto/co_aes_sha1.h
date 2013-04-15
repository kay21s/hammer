#ifndef CO_AES_SHA1_H
#define CO_AES_SHA1_H
#if 0
extern "C" void co_aes_sha1_gpu(uint8_t	*in,
			uint8_t			*out,
			uint8_t			*aes_keys,
			uint8_t			*ivs,
			uint8_t			*hmac_keys,
			uint32_t		*pkt_offset,
			uint16_t		*actual_length,
			unsigned int	num_flows,
			uint8_t			*checkbits,
			unsigned		threads_per_blk,
			cudaStream_t	stream);

#endif
#endif
