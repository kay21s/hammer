#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#include "hammer.h"
#include "hammer_config.h"
#include "hammer_epoll.h"
#include "hammer_sched.h"
#include "hammer_memory.h"
#include "hammer_dispatcher.h"
#include "hammer_cpu_worker.h"

/*
  |                            |                              |
  |   hammer_handler_ssl_read  |                              |
  | ------------------------>  |                              |
  |                            |    hammer_handler_write      |
  |                            |  ------------------------>   |
  |                            |                              |
  |                            |                              |
  |                            |                              |
  |                            |    hammer_batch_handler_read |
  |                            |  <------------------------   |
  |                            |                              |
  |   hammer_handler_ssl_write |                              |
  | <------------------------  |                              |
  |                            |                              |
  |                            |                              |
Client        (SSL)          Proxy         (Socket)         Server

*/



int hammer_config_init()
{
	int length;

	config = hammer_mem_calloc(sizeof(hammer_config_t));

	config->cpu_worker_num = 1;
	config->gpu_worker_num = 0;
	config->workers = 1; // cpu_worker_num + gpu_worker_num
	config->epoll_max_events = 128;

	length = strlen("219.219.216.11");
	config->server_ip = malloc(length);
	memcpy(config->server_ip, "219.219.216.11", length);
	config->server_port = 80;

	length = strlen("127.0.0.1");
	config->listen_ip = malloc(length);
	memcpy(config->listen_ip, "127.0.0.1", length);
	config->listen_port = 80;

	config->conn_buffer_size = 4096;

	config->ssl = 0; // if this is a ssl proxy
	config->gpu = 0; // if this need batch processing by GPU

	config->time_interval = 40; // ms
	/* we take 40ms as parameter, for 10Gbps bandwidth,
	   40ms * 10Gbps = 400 * 10^3 bits ~= (<) 50 KB = 40 * 1.25 * 10^3.
	   Take 64 bytes minimum packet size, at most 782 jobs each batch,
	   we allocate 1000 jobs at most.
	   */
	config->batch_buf_max_size = config->time_interval * 1.25 * 1000; // byte
	config->batch_job_max_num = 1000;

	config->key_size = 128/8; // byte
	config->iv_size = 128/8; // byte

	/*

	config = {
		1, // cpu_worker_num
		0, // gpu_worker_num
		1, // total worker
		128, // epoll_max_events

		"219.219.216.11", // server_ip
		80, // server_port
		"127.0.0.1", // listen_ip
		80, // listen_port

		4096, // conn_buffer_size
	};
	*/
	return 0;
}


int hammer_sched_init()
{
	int i;

	sched_list = hammer_mem_malloc(config->workers * sizeof(hammer_sched_t));
	for (i = 0; i < config->workers; i ++) {
		hammer_init_sched_node((hammer_sched_t *)&(sched_list[i]), -1, -1);
	}

	return 0;
}

void hammer_thread_keys_init()
{
	pthread_key_create(&worker_sched_struct, NULL);
	pthread_key_create(&worker_buf, NULL);
}

int hammer_launch_cpu_workers()
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

int hammer_launch_gpu_workers()
{
	pthread_t tid;
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	if (pthread_create(&tid, &attr, hammer_gpu_worker_loop, NULL) != 0) {
		printf("pthread_create error!!\n");
		return -1;
	}

	return 0;
}

int main()
{
	hammer_config_init();

	hammer_sched_init();
	//hammer_connection_init();
	hammer_thread_keys_init();
	

	/* Launch workers first*/
	hammer_launch_cpu_workers();
	hammer_launch_gpu_workers();

	/* the main function becomes the dispatcher and enters the dispatcher loop*/
	hammer_dispatcher();

	return 0;
}
