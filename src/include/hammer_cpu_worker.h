#ifndef HAMMER_CPU_WORKER_H
#define HAMMER_CPU_WORKER_H

typedef struct hammer_cpu_worker_context_s {
	int core_id;
	hammer_batch_t *batch;
	hammer_sched_t *sched;
} hammer_cpu_worker_context_t;

void *hammer_cpu_worker_loop(void *thread_sched);

#endif
