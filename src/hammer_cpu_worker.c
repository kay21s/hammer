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
#include "hammer_sched.h"
#include "hammer_handler.h"
#include "hammer_memory.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_macros.h"
#include "hammer_batch.h"

pthread_key_t worker_sched_struct;
pthread_key_t worker_batch;

void *hammer_epoll_start(int efd, hammer_epoll_handlers_t *handler, int max_events)
{
	int i, fd, ret = -1;
	int num_events;
	// int fds_timeout;

	struct epoll_event *events;
	hammer_sched_t *sched;
	hammer_connection_t *c;

	/* Get sched node */
	sched = hammer_sched_get_sched_struct();

	//fds_timeout = log_current_utime + config->timeout;
	events = hammer_mem_malloc(max_events * sizeof(struct epoll_event));

	/* Notify the dispatcher that this thread has been created */
	pthread_mutex_lock(&mutex_worker_init);
	sched->initialized = 1;
	pthread_mutex_unlock(&mutex_worker_init);

	while (1) {

		/* Each time, we first check if GPU has gave any indication for 
		   1) which buffer is taken,
		   2) which buffer has been processed */
		if (hammer_batch_if_gpu_processed_new(batch)) {
			hammer_batch_forwarding(batch);
		}

		//FIXME: maybe problems in pointer &events
		num_events = hammer_epoll_wait(efd, &events, max_events);

		for (i = 0; i < num_events; i++) {
			c = (hammer_connection_t *) events[i].data.ptr;
			fd = events[i].data.fd;

			if (events[i].events & EPOLLIN) {
				HAMMER_TRACE("[FD %i] EPoll Event READ", fd);
				if (c->ssl) {
					ret = (*handler->ssl_read) (c);
				} else {
					ret = (*handler->read) (c);
				}
			}
			else if (events[i].events & EPOLLOUT) {
				HAMMER_TRACE("[FD %i] EPoll Event WRITE", fd);
				if (c->ssl) {
					ret = (*handler->ssl_write) (c);
				} else {
					ret = (*handler->write) (c);
				}
			}
			else if (events[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) {
				HAMMER_TRACE("[FD %i] EPoll Event EPOLLHUP/EPOLLER", fd);
				ret = (*handler->error) (c);
			} else {
				hammer_err("What's up man, error here\n");
				exit(0);
			}

			if (ret < 0) {
				HAMMER_TRACE("[FD %i] Epoll Event FORCE CLOSE | ret = %i", fd, ret);
				(*handler->close) (c);
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
void *hammer_cpu_worker_loop(void *thread_sched)
{
	hammer_sched_t *sched = thread_sched;
	hammer_epoll_handlers_t *handler;

	handler = hammer_epoll_set_handlers((void *) hammer_handler_batch_read,
			(void *) hammer_handler_ssl_read,
			(void *) hammer_handler_write, 
			(void *) hammer_handler_ssl_write,
			(void *) hammer_handler_error,
			(void *) hammer_handler_close,
			(void *) hammer_handler_close);

	/* Export known scheduler node to context thread */
	pthread_setspecific(worker_sched_struct, (void *) sched);
	__builtin_prefetch(sched);
	__builtin_prefetch(&worker_sched_struct);

	// FIXME
	pthread_setspecific(worker_batch, (void *) batch);
	__builtin_prefetch(batch);
	__builtin_prefetch(&worker_batch);
	/* Allocate the batch buffers, each cpu worker has a set of buffers,
	 * two as input buffer, and two as output buffer. */
	hammer_batch_init();

	/* Init epoll_wait() loop */
	hammer_epoll_start(sched->epoll_fd, handler, sched->epoll_max_events);

	return 0;
}
