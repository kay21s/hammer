#ifndef HAMMER_CONFIG_H
#define HAMMER_CONFIG_H

typedef struct hammer_config_s {
	int cpu_worker_num;
	int gpu_worker_num;
	int workers; // cpu_worker_num + gpu_worker_num

	char *server_ip;
	unsigned int server_port;

	int conn_buffer_size;

	
} hammer_config_t;

hammer_config_t config = {
	1,
	0,
	1,
	"219.219.216.11",
	80,
	4096,
};

#endif
