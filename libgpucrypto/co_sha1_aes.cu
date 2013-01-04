#include "sha1.h"
#include "aes_core.h"
#include "aes_kernel.h"

#include <stdint.h>
#include <assert.h>

__global__ void sha1_aes_kernel(
			const uint8_t	*input_buf,
			uint8_t		*output_buf,
			const uint8_t	*aes_keys,
			uint8_t		*ivs,
			const char	*hmac_keys,
			const uint32_t	*pkt_offset,
			const uint16_t  *length,
			const unsigned int num_flows,
			uint8_t		*checkbits=0)
{
/**************************************************************************
 SHA-1 Calculation is started first
***************************************************************************/
	uint32_t w_register[16];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_flows) {
		uint32_t *w = w_register;
		hash_digest_t h;
		uint32_t offset = pkt_offset[idx];
		unsigned long length = length[idx];

		uint16_t sha1_pad_len = (length + 63 + 9) & (~0x3F);
		uint32_t *sha1_out = intput_buf + offset + sha1_pad_len;

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x36363636;
		xorpads(w, (uint32_t*)(hmac_keys + 64 * idx));

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		//SHA1 compute on ipad
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA1 compute on mesage
		unsigned num_iter = (length + 63 + 9) / 64;
		for (unsigned i = 0; i < num_iter; i++)
			computeSHA1Block(input_buf + offset, w, i * 64, length, h);

		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		*(sha1_out+2) = swap(h.h3);
		*(sha1_out+3) = swap(h.h4);
		*(sha1_out+4) = swap(h.h5);

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x5c5c5c5c;

		xorpads(w, (uint32_t*)(hmac_keys + 64 * idx));

		//SHA 1 compute on opads
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA 1 compute on (hash of ipad|m)
		computeSHA1Block((char*)sha1_out, w, 0, 20, h);

		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		*(sha1_out+2) = swap(h.h3);
		*(sha1_out+3) = swap(h.h4);
		*(sha1_out+4) = swap(h.h5);
	}
        __syncthreads();

/**************************************************************************
 SHA-1 Calculation completed, Now we start AES encryption
***************************************************************************/
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];

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
	const uint8_t *in  = pkt_offset[idx] + input_buf;
	uint8_t *out       = pkt_offset[idx] + output_buf;
	const uint8_t *key = idx * 16 + aes_keys;
	uint8_t *ivec      = idx * AES_BLOCK_SIZE + ivs;

	/* Encrypt using cbc mode, this is the actual length for encryption
	 * which has already been padded in application */
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
		

	// Now we set the checkbits
	if (threadIdx.x == 0)
		*(checkbits + blockIdx.x) = 1;
}


void co_sha1_aes_gpu(
			const uint8_t	*in,
			uint8_t		*out,
			const uint8_t	*aes_keys,
			uint8_t		*ivs,
			const char	*hmac_keys,
			const uint32_t	*pkt_offset,
			const uint16_t	*actual_length,
			const unsigned int num_flows,
			uint8_t		*checkbits,
			unsigned	threads_per_blk,
			cudaStream_t	stream)
{
	int num_blks = (N + threads_per_blk - 1) / threads_per_blk;

	if (stream == 0) {
		sha1_aes_kernel<<<num_blks, threads_per_blk>>>(
		       in, out, aes_keys, ivs, hmac_keys, pkt_offset, actual_length, num_flows, checkbits);
	} else  {
		sha1_aes_kernel<<<num_blks, threads_per_blk, 0, stream>>>(
		       in, out, aes_keys, ivs, hmac_keys, pkt_offset, actual_length, num_flows, checkbits);
	}
}
