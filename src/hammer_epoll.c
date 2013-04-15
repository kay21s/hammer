#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>

#include "hammer_epoll.h"
#include "hammer_memory.h"
#include "hammer_handler.h"
#include "hammer_macros.h"


hammer_epoll_handlers_t *hammer_epoll_set_handlers(void (*client_read) (hammer_connection_t *),
                                         void (*server_read) (hammer_connection_t *),
                                         void (*client_write) (hammer_connection_t *),
                                         void (*server_write) (hammer_connection_t *),
                                         void (*error) (hammer_connection_t *),
                                         void (*close) (hammer_connection_t *),
                                         void (*timeout) (hammer_connection_t *))
{
	hammer_epoll_handlers_t *handler;

	handler = hammer_mem_malloc(sizeof(hammer_epoll_handlers_t));
	handler->client_read = (void *) client_read;
	handler->server_read = (void *) server_read;
	handler->client_write = (void *) client_write;
	handler->server_write = (void *) server_write;
	handler->error = (void *) error;
	handler->close = (void *) close;
	handler->timeout = (void *) timeout;

	return handler;
}

int hammer_epoll_create(int max_events)
{
	int efd;

	efd = epoll_create(max_events);
	if (efd == -1) {
		hammer_err("epoll_create() failed");
	}

	return efd;
}

int hammer_epoll_wait(int efd, struct epoll_event **events, int max_events)
{
	int num_fds;
	num_fds = epoll_wait(efd, *events, max_events, HAMMER_EPOLL_WAIT_TIMEOUT);

	return num_fds;
}

int hammer_epoll_add(int efd, int fd, int init_mode, int behavior, void *user_ptr)
{
	int ret;
	struct epoll_event event = {0, {0}};

	event.data.fd = fd;
	event.events = EPOLLERR | EPOLLHUP | EPOLLRDHUP;

	if (behavior == HAMMER_EPOLL_EDGE_TRIGGERED) {
		event.events |= EPOLLET;
	}

	switch (init_mode) {
		case HAMMER_EPOLL_READ:
			event.events |= EPOLLIN;
			break;
		case HAMMER_EPOLL_WRITE:
			event.events |= EPOLLOUT;
			break;
		case HAMMER_EPOLL_RW:
			event.events |= EPOLLIN | EPOLLOUT;
			break;
		case HAMMER_EPOLL_SLEEP:
			event.events = 0;
			break;
	}

	event.data.ptr = user_ptr;

	/* Add to epoll queue */
	ret = epoll_ctl(efd, EPOLL_CTL_ADD, fd, &event);
	if (hammer_unlikely(ret < 0 && errno != EEXIST)) {
		HAMMER_TRACE("[FD %i] epoll_ctl() %s", fd, strerror(errno));
		return ret;
	}

	return ret;
}

int hammer_epoll_del(int efd, int fd)
{
	int ret;

	ret = epoll_ctl(efd, EPOLL_CTL_DEL, fd, NULL);
	HAMMER_TRACE("Epoll, removing fd %i from efd %i", fd, efd);

	return ret;
}

int hammer_epoll_change_mode(int efd, int fd, int mode, int behavior)
{
	int ret;
	struct epoll_event event = {0, {0}};

	event.events = EPOLLERR | EPOLLHUP;
	event.data.fd = fd;

	switch (mode) {
		case HAMMER_EPOLL_READ:
			HAMMER_TRACE("[FD %i] EPoll changing mode to READ", fd);
			event.events |= EPOLLIN;
			break;
		case HAMMER_EPOLL_WRITE:
			HAMMER_TRACE("[FD %i] EPoll changing mode to WRITE", fd);
			event.events |= EPOLLOUT;
			break;
		case HAMMER_EPOLL_RW:
			HAMMER_TRACE("[FD %i] Epoll changing mode to READ/WRITE", fd);
			event.events |= EPOLLIN | EPOLLOUT;
			break;
		case HAMMER_EPOLL_SLEEP:
			HAMMER_TRACE("[FD %i] Epoll changing mode to DISABLE", fd);
			event.events = 0;
			printf("epoll sleep? \n");
			break;
		case HAMMER_EPOLL_WAKEUP:
			printf("epoll wakeup? \n");
			break;
	}

	if (behavior == HAMMER_EPOLL_EDGE_TRIGGERED) {
		event.events |= EPOLLET;
	}

	/* Update epoll fd events */
	ret = epoll_ctl(efd, EPOLL_CTL_MOD, fd, &event);

	return ret;
}
