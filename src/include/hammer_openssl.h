#ifndef HAMMER_OPENSSL_H
#define HAMMER_OPENSSL

void hammer_openssl_init(hammer_connection_t *c);
void hammer_openssl_get_parameters(hammer_connection_t *c);
int hammer_openssl_accept(hammer_connection_t *c);
int hammer_openssl_read(hammer_connection_t *c, char *buffer, int read_size);
int hammer_openssl_write(hammer_connection_t *c, char *buffer, int write_size);
void hammer_openssl_close(hammer_connection_t *c);

#endif
