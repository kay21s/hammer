#ifndef HAMMER_CPU_WORKER_H
#define HAMMER_CPU_WORKER_H

#include "hammer_batch.h"

typedef struct hammer_gpu_worker_s {
	hammer_batch_buf_t **buf_set;
	int *buf_size;

	int buf_set_id;
	crypto_context cry_ctx;

	int total_bytes;
} hammer_gpu_worker_t;


void *hammer_gpu_worker_loop(void *arg);

#endif
