#include "sha1.h"
#include "aes_core.h"
#include "aes_kernel.h"

#include <stdint.h>
#include <assert.h>

/* AES counter mode + HMAC SHA-1, 
   the encryption of each block in AES counter mode is not parallelized in this implementation */
__global__ void aes_ctr_sha1_kernel(
			const uint8_t	*input_buf,
			uint8_t			*output_buf,
			const uint8_t	*aes_keys,
			uint8_t			*ivs,
			const char		*hmac_keys,
			const uint32_t	*pkt_offset,
			const uint16_t  *length,
			const unsigned int num_flows,
			uint8_t			*checkbits=0)
{
/**************************************************************************
 AES Encryption is started first
***************************************************************************/
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];

	/* Private counter 128 bits */
	uint64_t keystream[2];

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

	/* ----debug-----*/
	if (idx >= num_flows)
		return;

	/* make sure T boxes have been initialized. */
	__syncthreads();

	/* Encrypt using counter mode, this is the actual length of the packet */
	/* pkt_offset[idx + 1] - pkt_offset[idx] is used for "length[idx] + padding for HMAC + HMAC sha-1 tag" */
	unsigned long len	= length[idx];

	/* Skip RTP header to Locate the data to be encrypted */
	uint8_t *in			= pkt_offset[idx] + input_buf;
	uint8_t cc			= (in[0] & 0x80) >> 4; /* Get the number of CSRC identifiers */
	uint32_t header_len = (uint32_t *)((uint8_t *)in + 96 + 32 * cc + 4); /* Get the optional header length */
	header_len			= 128 + 32 * cc + header_len; /* Get the total header length */

	/* FIXME: optimization : copy the RTP header to output */
	for (i = 0; i < header_len; i ++) {
		((char *)out)[i] = ((char *)in)[i];
	}

	/* Jump to the parts need encryption */
	in					= in + header_len /* Get to the payload */
	uint8_t *out		= pkt_offset[idx] + output_buf;
	out					= out + header_len; /* Get to the payload */
	
	/* data length that needs encryption */
	len	-= header_len;

	/* ----debug----- */
	if (len <= 0) return;

	const uint8_t *key	= idx * 16 + aes_keys;
	uint64_t *iv		= (uint64_t *) (idx * AES_BLOCK_SIZE + ivs);

	while (len >= AES_BLOCK_SIZE) {

		/* for the ith block, its input is ((iv + i) mod 2^128)*/
		iv[0] ++;
		if (iv[0] == 0)
			iv[1] ++;

		/* Get the keystream here */
		AES_128_encrypt(iv, keystream, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);

		*((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)keystream);
		*(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)keystream) + 1);

		len -= AES_BLOCK_SIZE;
		in  += AES_BLOCK_SIZE;
		out += AES_BLOCK_SIZE;
	}

	if (len) {
		/* for the ith block, its input is ((iv + i) mod 2^128)*/
		iv[0] ++;
		if (iv[0] == 0)
			iv[1] ++;

		AES_128_encrypt(iv, keystream, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);

		for(unsigned n = 0; n < len; ++n)
			out[n] = in[n] ^ ((uint8_t *)keystream)[n];
	}

	__syncthreads();

/**************************************************************************
 AES Encryption completed, Now we start SHA-1 Calculation
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

		/* In SRTP, HMAC_KEY_SIZE is 160 bits = 20 bytes */
		xorpads(w, (uint32_t *)(hmac_keys + 20 * idx));

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		//SHA1 compute on ipad
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA1 compute on message
		unsigned num_iter = (length + 63 + 9) / 64;
		for (unsigned i = 0; i < num_iter; i ++)
			computeSHA1Block(input_buf + offset, w, i * 64, length, h);

		/* In SRTP, sha1_out has only 80 bits output 32+32+16 = 80 */
		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		uint32_t temp = swap(h.h3);
		*(uint16_t *)(sha1_out+2)  = ((uint16_t *)&temp)[0];
		//*(sha1_out+2) = swap(h.h3);
		//*(sha1_out+3) = swap(h.h4);
		//*(sha1_out+4) = swap(h.h5);

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x5c5c5c5c;

		xorpads(w, (uint32_t*)(hmac_keys + 20 * idx));

		//SHA 1 compute on opads
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA 1 compute on (hash of ipad|m)
		//HMAC_TAG_SIZE  = 10
		computeSHA1Block((char*)sha1_out, w, 0, 10, h);

		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		temp = swap(h.h3);
		*(uint16_t *)(sha1_out+2)  = ((uint16_t *)&temp)[0];
		//*(sha1_out+2) = swap(h.h3);
		//*(sha1_out+3) = swap(h.h4);
		//*(sha1_out+4) = swap(h.h5);
	}
	__syncthreads();

	// Now we set the checkbits
	if (threadIdx.x == 0)
		*(checkbits + blockIdx.x) = 1;
}


void co_aes_sha1_gpu(
			const uint8_t	*in,
			uint8_t			*out,
			const uint8_t	*aes_keys,
			uint8_t			*ivs,
			const char		*hmac_keys,
			const uint32_t	*pkt_offset,
			const uint16_t	*actual_length,
			const unsigned int num_flows,
			uint8_t			*checkbits,
			unsigned		threads_per_blk,
			cudaStream_t	stream)
{
	int num_blks = (N + threads_per_blk - 1) / threads_per_blk;

	if (stream == 0) {
		aes_ctr_sha1_kernel<<<num_blks, threads_per_blk>>>(
		       in, out, aes_keys, ivs, hmac_keys, pkt_offset, actual_length, num_flows, checkbits);
	} else  {
		aes_ctr_sha1_kernel<<<num_blks, threads_per_blk, 0, stream>>>(
		       in, out, aes_keys, ivs, hmac_keys, pkt_offset, actual_length, num_flows, checkbits);
	}
}
