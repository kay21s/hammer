#include <stdio.h>
#include <pthread.h>

#include "hammer_sched.h"
#include "hammer_epoll.h"
#include "hammer_macros.h"
#include "hammer_config.h"
#include "hammer_connection.h"
#include "hammer_batch.h"

pthread_mutex_t mutex_worker_init = PTHREAD_MUTEX_INITIALIZER;
pthread_key_t worker_sched_struct;

inline hammer_sched_t *hammer_sched_get_sched_struct()
{
    return pthread_getspecific(worker_sched_struct);
}

inline hammer_batch_t *hammer_sched_get_batch_struct()
{
    return pthread_getspecific(worker_batch_struct);
}

int hammer_init_sched_node(hammer_sched_t *sched, int epoll_fd, int thread_id)
{
	sched->epoll_fd = epoll_fd;
	sched->epoll_max_events = config->epoll_max_events;
	sched->thread_id = thread_id;

	sched->if_want_new = HAMMER_SCHED_WANT_NEW;

	sched->initialized = 0;

	sched->accepted_connections = 0;
	sched->closed_connections = 0;

	return 0;
}

int hammer_sched_want_new_conn(hammer_sched_t *sched)
{
	sched->if_want_new = HAMMER_SCHED_WANT_NEW;
	return 0;
}

int hammer_sched_want_no_conn(hammer_sched_t *sched)
{
	sched->if_want_new = HAMMER_SCHED_WANT_NO;
	return 0;
}

// simple dispatching algorithm
int hammer_sched_next_worker_id()
{
	static int id = -1;
	int i, pre_id;
	
	hammer_sched_t *sched;

	pre_id = id;
	id = -1;

	for (i = pre_id + 1; i < config->workers; i ++) {
		sched = &(sched_list[i]);
		if (sched->if_want_new == HAMMER_SCHED_WANT_NEW) {
			id = i;
			break;
		}
	}

	if (id == -1) {
		/* not find any available worker in previous search */
		for (i = 0; i <= pre_id; i ++) {
			sched = &(sched_list[i]);
			if (sched->if_want_new == HAMMER_SCHED_WANT_NEW) {
				id = i;
				break;
			}

		}
	}

	/* no available worker =( */
	if (id == -1) {
		printf("No available worker!!!\n");
	}

	return id;
}

void hammer_sched_add_connection(hammer_connection_t *c, hammer_sched_t *sched, hammer_connection_t *rc)
{
	int ret;

	ret = hammer_epoll_add(sched->epoll_fd, remote_fd, HAMMER_EPOLL_READ,
			HAMMER_EPOLL_LEVEL_TRIGGERED, (void *)c);
	if (hammer_likely(ret == 0)) {
		if (r_conn != NULL) {
			/* r_conn != NULL, this connection is added by connect(), to server */
			c->r_conn = rc;
			rc->r_conn = c;

			sched->connected_connections ++;
		} else {
			/* r_conn == NULL, this connection is added by accept(), from client */
			sched->accepted_connections ++;
		}
	} else {
		/* fails, close the connection */
		hammer_close_connection(c);
		hammer_err("epoll add fails\n");
		exit(0);
	}

	return;
}

// we delete both the two connections
int hammer_sched_del_connection(hammer_connection_t *c)
{
	hammer_connection_t *rc= c->r_conn;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

	/* remove this connection */
	hammer_epoll_del(sched->epoll_fd, c->socket);
	hammer_close_connection(c);
	sched->closed_connections ++;

	/* remove its corresponding connection */
	if (rc != NULL) {
		hammer_epoll_del(sched->epoll_fd, rc->socket);
		hammer_close_connection(rc);
		sched->closed_connections ++;
	}

	return 0;
}
