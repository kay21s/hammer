#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <sched.h>

#include "hammer.h"
#include "hammer_connection.h"
#include "hammer_sched.h"
#include "hammer_handler.h"
#include "hammer_memory.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_macros.h"
#include "hammer_batch.h"

void *hammer_epoll_start(int efd, hammer_epoll_handlers_t *handler, int max_events)
{
	int i, fd, ret = -1;
	int num_events;
	struct epoll_event *events;
	hammer_connection_t *c;
	// int fds_timeout;

	//fds_timeout = log_current_utime + config->timeout;
	events = hammer_mem_malloc(max_events * sizeof(struct epoll_event));

	while (1) {

		if (config->gpu) {
			/* Each time, we first check if GPU has gave any indication for 
			   1) which buffer is taken,
			   2) which buffer has been processed */
			if (hammer_batch_if_gpu_processed_new(batch)) {
				hammer_batch_forwarding(batch);
			}
		}

		//FIXME: maybe problems in pointer &events
		num_events = hammer_epoll_wait(efd, &events, max_events);

		for (i = 0; i < num_events; i ++) {
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
void *hammer_cpu_worker_loop(void *context)
{
	hammer_cpu_worker_context_t *my_context = (hammer_cpu_worker_context_t *)context;
	hammer_sched_t *sched = context->sched;
	hammer_batch_t *batch = context->batch;
	int core_id = context->core_id;
	hammer_epoll_handlers_t *handler;
	unsigned long mask = 0;

	/* Set affinity of this cpu worker */
	mask = 1 << core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), &mask) < 0) {
		hammer_err("Err set affinity in GPU worker\n");
		exit(0);
	}

	if (config->gpu) {
		assert(config->ssl); /* GPU must be used to accelerate ssl*/
		handler = hammer_epoll_set_handlers((void *) hammer_handler_batch_read,
						    (void *) hammer_handler_ssl_read,
						    (void *) hammer_handler_write, 
						    (void *) hammer_handler_write, // write directly, we have already encrypted the message
						    (void *) hammer_handler_error,
						    (void *) hammer_handler_close,
						    (void *) hammer_handler_close);
	else if (config->ssl) {
		/* This is a ssl proxy, without gpu acceleration */
		handler = hammer_epoll_set_handlers((void *) hammer_handler_read,
						    (void *) hammer_handler_ssl_read,
						    (void *) hammer_handler_write, 
						    (void *) hammer_handler_ssl_write,
						    (void *) hammer_handler_error,
						    (void *) hammer_handler_close,
						    (void *) hammer_handler_close);
	} else {
		/* This is just used for forwarding */
		handler = hammer_epoll_set_handlers((void *) hammer_handler_read,
						    (void *) hammer_handler_read,
						    (void *) hammer_handler_write, 
						    (void *) hammer_handler_write,
						    (void *) hammer_handler_error,
						    (void *) hammer_handler_close,
						    (void *) hammer_handler_close);
	}

	/* Export known scheduler node to context thread */
	pthread_setspecific(worker_sched_struct, (void *)sched);
	__builtin_prefetch(sched);
	__builtin_prefetch(&worker_sched_struct);

	pthread_setspecific(worker_batch_struct, (void *)batch);
	__builtin_prefetch(batch);
	__builtin_prefetch(&worker_batch_struct);

	if (config->gpu) {
		/* Allocate the batch buffers, each cpu worker has a set of buffers,
		 * two as input buffer, and two as output buffer. */
		hammer_batch_init();
	}

	/* Notify the dispatcher and the GPU worker that this thread has been created */
	pthread_mutex_lock(&mutex_worker_init);
	sched->initialized = 1;
	pthread_mutex_unlock(&mutex_worker_init);

	/* Init epoll_wait() loop */
	hammer_epoll_start(sched->epoll_fd, handler, sched->epoll_max_events);

	return 0;
}
