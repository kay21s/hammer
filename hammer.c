#include <pthread.h>

#include "hammer.h"
#include "hammer_config.h"
#include "hammer_sched.h"
#include "hammer_memory.h"
#include "hammer_dispatcher.h"
#include "hammer_cpu_worker.h"

hammer_sched_t *sched_list;

int hammer_sched_init()
{
	sched_list = hammer_mem_malloc(config->workers * sizeof(hammer_sched_t));
	for (i = 0; i < config->workers; i ++) {
		hammer_init_sched(&(sched_list[i]), -1, -1);
	}
}

void hammer_thread_key_init()
{
	pthread_key_create(&worker_sched_struct, NULL);
}

int hammer_dispatcher_launch_gpu_workers()
{
	int efd;
	pthread_t tid;
	pthread_attr_t attr;
	int i;

	hammer_sched_t *sched;

	for (i = 0; i < config->gpu_worker_num; i++) {
		/* Creating epoll file descriptor */
		efd = hammer_epoll_create(max_events);
		if (efd < 1) {
			return -1;
		}

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

int hammer_dispatcher_launch_cpu_workers()
{
	int efd;
	pthread_t tid;
	pthread_attr_t attr;
	int i;

	hammer_sched_t *sched;

	for (i = 0; i < config->cpu_worker_num; i++) {
		/* Creating epoll file descriptor */
		efd = hammer_epoll_create(max_events);
		if (efd < 1) {
			return -1;
		}

		sched = &(sched_list[thread_id]);
		hammer_init_sched(sched, efd, thread_id);

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		if (pthread_create(&tid, &attr, hammer_cpu_worker_loop,
					(void *) sched) != 0) {
			perror("pthread_create");
			return -1;
		}

	}

	return 0;
}


int main()
{

	hammer_sched_init();
	hammer_thread_keys_init();

	/* Launch workers first*/
	hammer_dispatcher_launch_cpu_workers();
	hammer_dispatcher_launch_gpu_workers();
	/* the main function becomes the dispatcher and enters the dispatcher loop*/
	hammer_dispatcher();
}
