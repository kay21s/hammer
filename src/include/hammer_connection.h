#ifndef HAMMER_CONNECTION_H
#define HAMMER_CONNECTION_H

#include "hammer_list.h"
//#include "openssl/ssl.h"
#include "../../libgpucrypto/crypto_size.h"

#define HAMMER_EVENT_READ	0
#define HAMMER_EVENT_WRITE	1

#define HAMMER_CONN_CLIENT	3
#define HAMMER_CONN_SERVER	4

typedef struct hammer_connection_s
{
	int socket;
	int type; // client or server

	// FIXME:not use any more
	int body_length; // current length
	int body_size; // total size
	char *body_ptr;

	struct hammer_connection_s *rc; // its reverse direction connection
	struct hammer_list *job_list;

#if 0
	// SSL structs
	SSL *ssl_handle;
	SSL_CTX *ssl_context;
#endif

	char aes_key[AES_KEY_SIZE];
	char iv[AES_IV_SIZE];
	char hmac_key[HMAC_KEY_SIZE];

} hammer_connection_t;

// All events are forwarding event
typedef struct hammer_job_s
{
	int job_body_length; // padded length
	int job_actual_length;
	char *job_body_ptr;

	/* which connection this job belongs */
	hammer_connection_t *connection;

	struct hammer_list _head;
} hammer_job_t;

int hammer_conn_job_add(hammer_connection_t *c, int length);
int hammer_conn_job_del(hammer_job_t *job);
void hammer_init_connection(hammer_connection_t *c);
hammer_connection_t *hammer_get_connection();
void hammer_free_connection(hammer_connection_t *c);
int hammer_close_connection(hammer_connection_t *c);

#endif
