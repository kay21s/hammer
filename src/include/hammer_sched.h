#ifndef HAMMER_SCHED_H
#define HAMMER_SCHED_H

#include <pthread.h>
#include "hammer_connection.h"
#include "hammer_batch.h"

#define HAMMER_SCHED_WANT_NEW	0
#define HAMMER_SCHED_WANT_NO	1

#define HAMMER_SCHED_WORKER_CPU	2
#define HAMMER_SCHED_WORKER_GPU	3

extern pthread_key_t worker_sched_struct;

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

extern hammer_sched_t *sched_set;
extern pthread_mutex_t mutex_worker_init;

inline hammer_sched_t *hammer_sched_get_sched_struct();
inline hammer_batch_t *hammer_sched_get_batch_struct();
int hammer_sched_node_init(hammer_sched_t *sched, int epoll_fd, int thread_id);
int hammer_sched_want_new_conn(hammer_sched_t *sched);
int hammer_sched_want_no_conn(hammer_sched_t *sched);
int hammer_sched_add_connection(hammer_connection_t *c, hammer_sched_t *sched, int ctype);
int hammer_sched_del_connection(hammer_connection_t *c);

#endif
