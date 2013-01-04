#ifndef CRYPTO_SIZE_H
#define CRYPTO_SIZE_H

/* Define sizes in bytes */

#define AES_KEY_SIZE	16  // 128/8 = 16 bytes
#define AES_IV_SIZE	16  // 16 bytes
#define PKT_OFFSET_SIZE	4   // 32 bits = 4 bytes
#define HMAC_KEY_SIZE	64  // 64 bytes
#define LENGTH_SIZE	4   // 32 bits = 4 bytes

#define SHA1_OUTPUT_SIZE	20  // output of SHA1 is 20 bytes

#endif
