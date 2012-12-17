#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

#include "hammer_sched.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_socket.h"
#include "hammer_connection.h"
#include "hammer_dispatcher.h"
#include "hammer_macros.h"
#include "hammer_handler.h"

int hammer_dispatcher_loop(int server_fd)
{
	int ret, remote_fd, worker_id = 0;
	hammer_sched_t *sched;
	hammer_connection_t *c;

	/* Activate TCP_DEFER_ACCEPT */
	if (hammer_socket_set_tcp_defer_accept(server_fd) != 0) {
		hammer_warn("TCP_DEFER_ACCEPT failed\n");
	}

	/* Accept new connections */
	while (1) {
		/* accept first */
		c = hammer_handler_accept(server_fd);

		/* Next worker target */
		worker_id = hammer_sched_next_worker_id();
		if (hammer_unlikely(worker_id == -1)) {
			hammer_err("no worker available\n");
			exit(0);
		}
		sched = &(sched_list[worker_id]);

		/* Assign connection to worker thread */
		hammer_sched_add_connection(c, sched, NULL);
	}

	return 0;
}

int hammer_dispatcher()
{
	int i, ready = 0;
	int server_fd;

	// waiting for the launch of workers
	while (1) {
		pthread_mutex_lock(&mutex_worker_init);
		for (i = 0; i < config->cpu_worker_num + config->gpu_worker_num; i++) {
			if (sched_list[i].initialized)
				ready++;
		}
		pthread_mutex_unlock(&mutex_worker_init);

		if (ready == config->workers) break;
		usleep(10000);
	}

	server_fd = hammer_handler_listen();

	/* Server loop, let's listen for incomming clients */
	hammer_dispatcher_loop(server_fd);

	return 0;
}

