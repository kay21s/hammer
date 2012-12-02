#ifndef HAMMER_CONFIG_H
#define HAMMER_CONFIG_H

#include "hammer_memory.h"

typedef struct hammer_config_s {
	int cpu_worker_num;
	int gpu_worker_num;
	int workers; // cpu_worker_num + gpu_worker_num
	int epoll_max_events;

	char *server_ip;
	unsigned int server_port;

	char *listen_ip;
	unsigned int listen_port;

	int conn_buffer_size;
} hammer_config_t;

hammer_config_t *config;


#endif
