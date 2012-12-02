#ifndef HAMMER_SOCKET_H
#define HAMMER_SOCKET_H

#include <sys/uio.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "hammer_iov.h"

#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK 04000
#endif

/* Socket_Timeout() */
#define ST_RECV 0
#define ST_SEND 1

#define TCP_CORK_ON 1
#define TCP_CORK_OFF 0

int hammer_socket_set_cork_flag(int fd, int state);
int hammer_socket_set_tcp_nodelay(int sockfd);
int hammer_socket_set_tcp_defer_accept(int sockfd);
int hammer_socket_set_nonblocking(int sockfd);

int hammer_socket_create(void);
int hammer_socket_connect(char *host, int port);
int hammer_socket_accept(int server_fd);
int hammer_socket_write(int socket_fd, const void *buf, size_t count);
int hammer_socket_read(int socket_fd, void *buf, int count);
int hammer_socket_close(int socket);

//int hammer_socket_reset(int socket);
//int hammer_socket_server(int port, char *listen_addr);
//int hammer_socket_timeout(int s, char *buf, int len, int timeout, int recv_send);

//int hammer_socket_sendv(int socket_fd, struct hammer_iov *hammer_io);
//int hammer_socket_ip_str(int socket_fd, char **buf, int size, unsigned long *len);

#endif
