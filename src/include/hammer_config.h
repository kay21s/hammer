#ifndef HAMMER_CONFIG_H
#define HAMMER_CONFIG_H

#include "hammer_memory.h"

typedef struct hammer_config_s {
	unsigned int cpu_worker_num;
	unsigned int gpu_worker_num;
	int workers; // cpu_worker_num + gpu_worker_num
	unsigned int epoll_max_events;

	char *server_ip;
	unsigned int server_port;

	char *listen_ip;
	unsigned int listen_port;

	unsigned int conn_buffer_size;

	unsigned int ssl;
	unsigned int gpu;
	unsigned int batch_buf_max_size;
	unsigned int batch_job_max_num;

	unsigned int *core_ids;

	// Most important argument for realtime scheduling algorithm
	unsigned int I; // 40ms, 30ms ...

	/* we currently not use these */
	unsigned int aes_key_size;
	unsigned int iv_size;
	unsigned int hmac_key_size
} hammer_config_t;

hammer_config_t *config;


#endif
