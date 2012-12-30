#ifndef CRYPTO_KERNEL_H
#define CRYPTO_KERNEL_H

#include <stdint.h>
#include <cuda_runtime.h>

#define SHA1_THREADS_PER_BLK 32
#define MAX_KEY_SIZE 64
#define MAX_HASH_SIZE 20

void AES_cbc_128_encrypt_gpu(const uint8_t *in_d,
			     uint8_t *out_d,
			     const uint32_t* pkt_offset_d,
			     const uint8_t *keys_d,
			     uint8_t *ivs_d,
			     const unsigned int numFlows,
			     uint8_t *checkbits_d,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream = 0);

void hmac_sha1_gpu(char* buf, char* keys,  uint32_t *offsets, uint16_t *lengths,
		   uint32_t *outputs, int N, uint8_t * checkbits,
		   unsigned threads_per_blk, cudaStream_t stream=0);

#endif /* CRYPTO_KERNEL_H */
