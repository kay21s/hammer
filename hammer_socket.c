#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/sendfile.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include "hammer.h"
#include "hammer_macros.h"
#include "hammer_socket.h"

int hammer_socket_create()
{
    int sockfd;

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        printf("client: socket");
        return -1;
    }

    return sockfd;
}

int hammer_socket_set_tcp_defer_accept(int sockfd)
{
	int timeout = 0;

	return setsockopt(sockfd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &timeout, sizeof(int));
}

int hammer_socket_accept(int server_fd)
{
	int remote_fd;
	struct sockaddr sock_addr;

	socklen_t socket_size = sizeof(struct sockaddr);
	remote_fd = accept(server_fd, &sock_addr, &socket_size);
	/* Set this socket non-blocking */
	hammer_socket_set_nonblocking(remote_fd);

	return remote_fd;
}

int hammer_socket_bind(int socket_fd, struct sockaddr *addr, socklen_t addrlen)
{
	ssize_t ret;
	ret = bind(socket_fd, addr, addrlen);

	return ret;
}

int hammer_socket_listen(int socket_fd, int backlog)
{
	ssize_t ret;
	ret = listen(socket_fd, backlog);

	return ret;
}

int hammer_socket_read(int socket_fd, void *buf, int count)
{
	ssize_t bytes_read;
	bytes_read = read(socket_fd, (void *)buf, count);

	return bytes_read;
}

int hammer_socket_write(int socket_fd, const void *buf, size_t count)
{
	ssize_t bytes_write;

	bytes_write = write(socket_fd, (void *)buf, count);

	return bytes_write;
}

int hammer_socket_connect(int socket_fd, struct sockaddr *addr, socklen_t addrlen)
{
	int ret;
	ret = connect(socket_fd, addr, addrlen);

	return ret;
}

int hammer_socket_close(int socket_fd)
{
	close(socket_fd);

	return 0;
}


/*
 * Example from:
 * http://www.baus.net/on-tcp_cork
 */
int hammer_socket_set_cork_flag(int fd, int state)
{

    HAMMER_TRACE("Socket, set Cork Flag FD %i to %s", fd, (state ? "ON" : "OFF"));

    return setsockopt(fd, SOL_TCP, TCP_CORK, &state, sizeof(state));
}

int hammer_socket_set_nonblocking(int sockfd)
{

    HAMMER_TRACE("Socket, set FD %i to non-blocking", sockfd);

    if (fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFD, 0) | O_NONBLOCK) == -1) {
        //hammer_err("Can't set to non-blocking mode socket %i", sockfd);
        return -1;
    }
    return 0;
}

int hammer_socket_set_tcp_nodelay(int sockfd)
{
    int on = 1;

    return setsockopt(sockfd, SOL_TCP, TCP_NODELAY, &on, sizeof(on));
}

int hammer_socket_reset(int socket)
{
    int status = 1;

    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &status, sizeof(int)) ==
        -1) {
        //perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    return 0;
}
