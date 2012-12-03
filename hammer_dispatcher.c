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

	/* Activate TCP_DEFER_ACCEPT */
	if (hammer_socket_set_tcp_defer_accept(server_fd) != 0) {
		hammer_warn("TCP_DEFER_ACCEPT failed\n");
	}

	/* Accept new connections */
	while (1) {
		remote_fd = hammer_socket_accept(server_fd);
		if (hammer_unlikely(remote_fd == -1)) {
			return -1;
		}

		/* Next worker target */
		worker_id = hammer_sched_next_worker_id();
		if (hammer_unlikely(worker_id == -1)) {
			return -1;
		}

		sched = &(sched_list[worker_id]);

		/* Assign socket to worker thread */
		ret = hammer_sched_add_connection(remote_fd, sched, NULL);
		if (ret == -1) {
			hammer_socket_close(remote_fd);
			return ret;
		}
	}
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

