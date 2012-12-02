#include "hammer_sched.h"
#include "hammer_connection.h"

extern pthread_key_t worker_sched_struct;

int hammer_init_sched(hammer_sched_t *sched, int epoll_fd, int thread_id)
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

static inline struct hammer_sched_t *hammer_sched_get_sched_struct()
{
    return pthread_getspecific(worker_sched_struct);
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


inline int hammer_sched_add_connection(int remote_fd, hammer_sched_t *sched, hammer_connection_t *r_conn)
{
	int ret;
	hammer_connection_t *new_conn;

	/* Get a connection and associate with the epoll event */
	new_conn = hammer_get_connection();
	hammer_init_connection(new_conn);

	ret = hammer_epoll_add(sched->epoll_fd, remote_fd, HAMMER_EPOLL_READ,
			HAMMER_EPOLL_LEVEL_TRIGGERED, (void *)conn);

	/* If epoll has failed, decrement the active connections counter */
	if (hammer_likely(ret == 0)) {
		if (r_conn != NULL) {
			/* r_conn!=NULL, this connection is added by connect(), to server */
			new_conn->r_conn = r_conn;
			r_conn->r_conn = new_conn;

			sched->connected_connections ++;
		} else {
			/* r_conn == NULL, this connection is added by accept(), from client */
			sched->accepted_connections ++;
		}
	} else {
		/* fails, free the connection */
		hammer_free_connection(new_conn);
	}

	return ret;
}

// we delete both the two connections
inline int hammer_sched_del_connection(hammer_conn_t *conn)
{
	hammer_connection_t *r_conn = conn->r_conn;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

	/* remove this connection */
	hammer_epoll_del(sched->epoll_fd, conn->socket);
	hammer_close_connection(conn);
	sched->closed_connections ++;

	/* remove its corresponding connection */
	if (r_conn != NULL) {
		hammrer_epoll_del(sched->epoll_fd, r_conn->socket);
		hammer_close_connection(r_conn);
		sched->closed_connections ++;
	}

	return 0;
}
