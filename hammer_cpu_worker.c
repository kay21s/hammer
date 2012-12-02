#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>

#include "hammer.h"
#include "hammer_connection.h"
#include "hammer_scheduler.h"
#include "hammer_memory.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_utils.h"
#include "hammer_macros.h"

pthread_key_t worker_sched_node;

static pthread_mutex_t mutex_sched_init = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_worker_init = PTHREAD_MUTEX_INITIALIZER;

void *hammer_epoll_start(int efd, hammer_epoll_handlers_t *handler, int max_events)
{
	int i, fd, ret = -1;
	int num_events;
	// int fds_timeout;

	struct epoll_event *events;
	hammer_sched_t *sched;
	hammer_connection_t *conn;

	/* Get sched node */
	sched = hammer_sched_get_sched_struct();

	//fds_timeout = log_current_utime + config->timeout;
	events = hammer_mem_malloc(max_events * sizeof(struct epoll_event));

	/* Notify the dispatcher that this thread has been created */
	pthread_mutex_lock(&mutex_worker_init);
	sched->initialized = 1;
	pthread_mutex_unlock(&mutex_worker_init);

	while (1) {
		ret = -1;
		//FIXME: maybe problems in pointer &events
		num_events = hammer_epoll_wait(efd, &events, max_events);

		for (i = 0; i < num_events; i++) {
			conn = (hammer_connection_t *)events[i].data.ptr;
			fd = events[i].data.fd;

			if (events[i].events & EPOLLIN) {
				HAMMER_TRACE("[FD %i] EPoll Event READ", fd);
				ret = (*handler->read) (conn);
			}
			else if (events[i].events & EPOLLOUT) {
				HAMMER_TRACE("[FD %i] EPoll Event WRITE", fd);
				ret = (*handler->write) (conn);
			}
			else if (events[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) {
				HAMMER_TRACE("[FD %i] EPoll Event EPOLLHUP/EPOLLER", fd);
				ret = (*handler->error) (conn);
			}

			if (ret < 0) {
				HAMMER_TRACE("[FD %i] Epoll Event FORCE CLOSE | ret = %i", fd, ret);
				(*handler->close) (conn);
			}
		}

		// FIXME: enable timeout
		/* Check timeouts and update next one 
		   if (log_current_utime >= fds_timeout) {
		   hammer_sched_check_timeouts(sched);
		   fds_timeout = log_current_utime + config->timeout;
		   }*/
	}

	return NULL;
}

/* created thread, all this calls are in the thread context */
//FIXME: static function ? why ?
static void *hammer_cpu_worker_loop(void *thread_sched)
{
	hammer_sched_t *sched = thread_sched;
	hammer_epoll_handlers_t *handler;

	handler = hammer_epoll_set_handlers((void *) hammer_handler_read,
			(void *) hammer_handler_write,
			(void *) hammer_handler_error,
			(void *) hammer_handler_close,
			(void *) hammer_handler_close);

	/* Export known scheduler node to context thread */
	pthread_setspecific(worker_sched_struct, (void *) sched);

	__builtin_prefetch(sched);
	__builtin_prefetch(&worker_sched_struct);

	/* Init epoll_wait() loop */
	hammer_epoll_start(sched->epoll_fd, handler, sched->epoll_max_events);

	return 0;
}
