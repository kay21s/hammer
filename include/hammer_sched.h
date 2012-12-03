#ifndef HAMMER_SCHED_H
#define HAMMER_SCHED_H

#include <pthread.h>
#include "hammer_connection.h"

#define HAMMER_SCHED_WANT_NEW	0
#define HAMMER_SCHED_WANT_NO	1

#define HAMMER_SCHED_WORKER_CPU	2
#define HAMMER_SCHED_WORKER_GPU	3

pthread_key_t worker_sched_struct;

typedef struct hammer_sched_s {
	int epoll_fd;
	int epoll_max_events;

	int thread_id;

	unsigned char if_want_new;
	unsigned char initialized;

	unsigned char worker_type;

	unsigned long long accepted_connections;
	unsigned long long connected_connections;
	unsigned long long closed_connections;

} hammer_sched_t;

hammer_sched_t *sched_list;
extern pthread_mutex_t mutex_worker_init;

int hammer_init_sched_node(hammer_sched_t *sched, int epoll_fd, int thread_id);
int hammer_sched_want_new_conn(hammer_sched_t *sched);
int hammer_sched_want_no_conn(hammer_sched_t *sched);
inline hammer_sched_t *hammer_sched_get_sched_struct();
int hammer_sched_next_worker_id();
inline int hammer_sched_add_connection(int remote_fd, hammer_sched_t *sched, hammer_connection_t *r_conn);
inline int hammer_sched_del_connection(hammer_connection_t *conn);

#endif
