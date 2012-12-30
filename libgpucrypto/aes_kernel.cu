#include "aes_core.h"
#include "crypto_kernel.h"

/*******************************************************************
  AES CBC kernel
******************************************************************/
__global__ void
AES_cbc_128_encrypt_kernel_SharedMem(const uint8_t       *in_all,
				     uint8_t             *out_all,
				     const uint32_t      *pkt_offset,
				     const uint8_t       *keys,
				     uint8_t             *ivs,
				     const unsigned int  num_flows,
				     uint8_t             *checkbits = 0)
{
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];

	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	/* initialize T boxes */
	for (unsigned i = 0 ; i *blockDim.x < 256 ; i++) {
		unsigned index = threadIdx.x + i * blockDim.x;
		if (index >= 256)
			break;
		shared_Te0[index] = Te0_ConstMem[index];
		shared_Te1[index] = Te1_ConstMem[index];
		shared_Te2[index] = Te2_ConstMem[index];
		shared_Te3[index] = Te3_ConstMem[index];
	}

	for(unsigned  i = 0;  i * blockDim.x < 10; i++){
		int index = threadIdx.x + blockDim.x * i;
		if(index < 10){
			shared_Rcon[index] = rcon[index];
		}
	}

	if (idx >= num_flows)
		return;

	/* make sure T boxes have been initialized. */
	__syncthreads();

	/* Locate data */
	const uint8_t *in  = pkt_offset[idx] + in_all;
	uint8_t *out       = pkt_offset[idx] + out_all;
	const uint8_t *key = idx * 16 + keys;
	uint8_t *ivec      = idx * AES_BLOCK_SIZE + ivs;

	/* Encrypt using cbc mode */
	unsigned long len = pkt_offset[idx + 1] - pkt_offset[idx];
	const unsigned char *iv = ivec;

	while (len >= AES_BLOCK_SIZE) {
		*((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)iv);
		*(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)iv) + 1);

		AES_128_encrypt(out, out, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
		iv = out;
		len -= AES_BLOCK_SIZE;
		in  += AES_BLOCK_SIZE;
		out += AES_BLOCK_SIZE;
	}


	if (len) {
		for(unsigned n = 0; n < len; ++n)
			out[n] = in[n] ^ iv[n];
		for(unsigned n = len; n < AES_BLOCK_SIZE; ++n)
			out[n] = iv[n];
		AES_128_encrypt(out, out, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
		iv = out;
	}

	*((uint4*)ivec) = *((uint4*)iv);

	__syncthreads();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;
}

/**************************************************************************
 Exported C++ function wrapper function for CUDA kernel
***************************************************************************/
void AES_cbc_128_encrypt_gpu(const uint8_t      *in_d,
			     uint8_t            *out_d,
			     const uint32_t     *pkt_offset_d,
			     const uint8_t      *keys_d,
			     uint8_t            *ivs_d,
			     const              unsigned int num_flows,
			     uint8_t            *checkbits_d,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream)
{
	unsigned int num_cuda_blks = (num_flows+threads_per_blk - 1) / threads_per_blk;
	if (stream == 0) {
		AES_cbc_128_encrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, pkt_offset_d, keys_d, ivs_d, num_flows, checkbits_d);
	} else {
		AES_cbc_128_encrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    in_d, out_d, pkt_offset_d, keys_d, ivs_d, num_flows, checkbits_d);
	}
}

