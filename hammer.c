#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include "hammer.h"
#include "hammer_config.h"
#include "hammer_epoll.h"
#include "hammer_sched.h"
#include "hammer_memory.h"
#include "hammer_dispatcher.h"
#include "hammer_cpu_worker.h"

hammer_config_t *hammer_get_config()
{
	hammer_config_t *config;

	config = hammer_mem_calloc(sizeof(hammer_config_t));
	config->cpu_worker_num = 1;
	config->gpu_worker_num = 0;
	config->workers = 1; // cpu_worker_num + gpu_worker_num
	config->epoll_max_events = 128;

	config->server_ip = "219.219.216.11";
	config->server_port = 80;

	config->listen_ip = "0.0.0.0";
	config->listen_port = 80;

	config->conn_buffer_size = 4096;
	/*
	config = {
		1,
		0,
		1,
		128,
		"219.219.216.11",
		80,
		4096,
	};*/

	return config;
}


int hammer_sched_init()
{
	int i;

	sched_list = hammer_mem_malloc(config->workers * sizeof(hammer_sched_t));
	for (i = 0; i < config->workers; i ++) {
		hammer_init_sched_node((hammer_sched_t *)&(sched_list[i]), -1, -1);
	}
}

void hammer_thread_keys_init()
{
	pthread_key_create(&worker_sched_struct, NULL);
}
#if 0
int hammer_dispatcher_launch_gpu_workers()
{
	int efd;
	pthread_t tid;
	pthread_attr_t attr;
	int i, thread_id;

	hammer_sched_t *sched;

	for (i = 0; i < config->gpu_worker_num; i++) {
		/* Creating epoll file descriptor */
		efd = hammer_epoll_create(config->max_epoll_events);
		if (efd < 1) {
			return -1;
		}

		thread_id = config->cpu_worker_num + i;

		sched = &(sched_list[thread_id]);
		hammer_init_sched(sched, efd, thread_id);

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		if (pthread_create(&tid, &attr, hammer_gpu_worker_loop,
					(void *) sched) != 0) {
			perror("pthread_create");
			return -1;
		}

	}

	return 0;
}
#endif

int hammer_dispatcher_launch_cpu_workers()
{
	int efd;
	pthread_t tid;
	pthread_attr_t attr;
	int i, thread_id;

	hammer_sched_t *sched;

	for (i = 0; i < config->cpu_worker_num; i++) {
		/* Creating epoll file descriptor */
		efd = hammer_epoll_create(config->epoll_max_events);
		if (efd < 1) {
			return -1;
		}

		thread_id = i;

		sched = &(sched_list[thread_id]);
		hammer_init_sched_node(sched, efd, thread_id);

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		if (pthread_create(&tid, &attr, hammer_cpu_worker_loop, (void *) sched) != 0) {
			printf("pthread_create error!!\n");
			return -1;
		}

	}

	return 0;
}


int main()
{
	hammer_sched_init();
	hammer_thread_keys_init();
	config = hammer_get_config();

	/* Launch workers first*/
	hammer_dispatcher_launch_cpu_workers();
//	hammer_dispatcher_launch_gpu_workers();
	/* the main function becomes the dispatcher and enters the dispatcher loop*/
	hammer_dispatcher();
}
