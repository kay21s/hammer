#include <stdlib.h>

#include "hammer_openssl.h"

#include "openssl/rand.h"
#include "openssl/ssl.h"
#include "openssl/aes.h"
#include "openssl/err.h"

void hammer_openssl_init(hammer_connection_t *c)
{
	// Register the error strings for libcrypto & libssl
	SSL_load_error_strings();
	// Register the available ciphers and digests
	SSL_library_init();

	// New context saying we are a server, and using SSL 2 or 3
	c->ssl_context = SSL_CTX_new(SSLv3_server_method ());
	if (c->sslContext == NULL)
		ERR_print_errors_fp (stderr);

	// Create an SSL struct for the connection
	c->ssl_handle = SSL_new(c->ssl_context);
	if (c->sslHandle == NULL)
		ERR_print_errors_fp (stderr);

	// Connect the SSL struct to our connection
	if (!SSL_set_fd (c->ssl_handle, c->socket))
		ERR_print_errors_fp (stderr);

	return;
}

void hammer_openssl_get_parameters(hammer_connection_t *c)
{
	/* Get AES Key */
	AES_KEY *key_st = (AES_KEY *)(c->ssl_handle->enc_write_ctx->cipher_data);
	struct aes_key_st aes_key_s = key_st->ks;
	memcpy((void *)c->aes_key, (void *)(aes_key_s->rd_key), AES_KEY_SIZE);

	c->aes_rounds = aes_key_st->rounds;
	memcpy((void *)c->iv, (void *)c->ssl_handle->enc_write_ctx->iv, AES_IV_SIZE);

	/* Get HMAC Key */
	memcpy((void *)c->hmac_key, (void *)(c->ssl_handle->s3->write_mac_secret), HMAC_KEY_SIZE);
}

int hammer_openssl_accept(hammer_connection_t *c)
{
	int ret;
	ret = SSL_accept(c->ssl_handle);
	if (ret < 0) {
		ERR_print_errors_fp (stderr);
	}

	return ret;
}

int hammer_openssl_read(hammer_connection_t *c, char *buffer, int read_size)
{
	int len;

	len = SSL_read(c->ssl_handle, buffer, read_size);
	return len;
}

int hammer_openssl_write(hammer_connection_t *c, char *buffer, int write_size)
{
	int len;

	len = SSL_write(c->ssl_handle, buffer, write_size);
	return len;
}

void hammer_openssl_close(hammer_connection_t *c)
{
	if (c->ssl_handle) {
		SSL_shutdown(c->ssl_handle);
		SSL_free(c->ssl_handle);
	}
	if (c->ssl_context)
		SSL_CTX_free(c->ssl_context);

	return;
}
