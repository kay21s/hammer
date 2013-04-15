#ifndef HAMMER_HANDLER_H
#define HAMMER_HANDLER_H

#include "hammer_connection.h"

int hammer_handler_connect(hammer_connection_t *c);
int hammer_handler_listen();
hammer_connection_t *hammer_handler_accept(int server_socket);
int hammer_handler_error(hammer_connection_t *c);
int hammer_handler_close(hammer_connection_t *c);
int hammer_handler_read(hammer_connection_t *c);
int hammer_handler_ssl_read(hammer_connection_t *c);
int hammer_handler_write(hammer_connection_t *c);
int hammer_handler_ssl_write(hammer_connection_t *c);


#endif
