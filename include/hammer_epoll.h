#ifndef HAMMER_EPOLL_H
#define HAMMER_EPOLL_H

#include <sys/epoll.h>
#include "hammer_connection.h"

#define HAMMER_EPOLL_READ     0
#define HAMMER_EPOLL_WRITE    1
#define HAMMER_EPOLL_RW       2
#define HAMMER_EPOLL_SLEEP    3
#define HAMMER_EPOLL_WAKEUP   4

/* Epoll timeout is 3 seconds */
#define HAMMER_EPOLL_WAIT_TIMEOUT 3000

#define HAMMER_EPOLL_LEVEL_TRIGGERED 2        /* default */
#define HAMMER_EPOLL_EDGE_TRIGGERED  EPOLLET

#ifndef EPOLLRDHUP
#define EPOLLRDHUP 0x2000
#endif

#define HAMMER_EPOLL_STATE_INDEX_CHUNK 64

typedef struct hammer_epoll_handlers_s
{
	int (*read) (hammer_connection_t *);
	int (*write) (hammer_connection_t *);
	int (*error) (hammer_connection_t *);
	int (*close) (hammer_connection_t *);
	int (*timeout) (hammer_connection_t *);
} hammer_epoll_handlers_t;

/* Monkey epoll calls */
int hammer_epoll_create(int max_events);
void *hammer_epoll_init(int efd, hammer_epoll_handlers_t *handler, int max_events);

hammer_epoll_handlers_t *hammer_epoll_set_handlers(void (*read) (hammer_connection_t *),
						void (*write) (hammer_connection_t *),
						void (*error) (hammer_connection_t *),
						void (*close) (hammer_connection_t *),
						void (*timeout) (hammer_connection_t *));

int hammer_epoll_add(int efd, int fd, int mode, int behavior, void *user_ptr);
int hammer_epoll_del(int efd, int fd);
int hammer_epoll_change_mode(int efd, int fd, int mode, int behavior);

#endif
