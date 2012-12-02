#ifndef HAMMER_HANDLER_H
#define HAMMER_HANDLER_H

#include "hammer_connection.h"

int hammer_handler_connect(hammer_connection_t *conn);
int hammer_handler_accept();
int hammer_handler_error(hammer_connection_t *conn);
int hammer_handler_close(hammer_connection_t *conn);
int hammer_handler_read(hammer_connection_t *conn);
int hammer_handler_write(hammer_connection_t *conn);


#endif
