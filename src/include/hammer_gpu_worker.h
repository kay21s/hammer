#ifndef HAMMER_GPU_WORKER_H
#define HAMMER_GPU_WORKER_H

#include "hammer_batch.h"
#include "../../libgpucrypto/crypto_context.h"

typedef struct hammer_gpu_worker_s {
	hammer_batch_buf_t **buf_set_A; /* All the buf_A in each CPU's batch */
	hammer_batch_buf_t **buf_set_B; /* All the buf_B in each CPU's batch */

	hammer_batch_buf_t **cur_buf_set; /* either buf_set_A or buf_set_B */
	int buf_set_id;

	crypto_context_t cry_ctx;
	int total_bytes;
} hammer_gpu_worker_t;

typedef struct hammer_gpu_worker_context_s {
	hammer_batch_t *cpu_batch_set;
	hammer_sched_t *sched;
	int core_id; /* which core should gpu worker run */
	/* Add more info passing to GPU worker here ... */
} hammer_gpu_worker_context_t;

void *hammer_gpu_worker_loop(void *context);
void hammer_gpu_get_batch(hammer_gpu_worker_t *g, hammer_batch_buf_t *batch_set);
void hammer_gpu_give_result(hammer_gpu_worker_t *g, hammer_batch_buf_t *batch_set);
void hammer_gpu_worker_init(hammer_gpu_worker_t *g, hammer_batch_t *batch_set, hammer_sched_t *sched_set);

#endif
