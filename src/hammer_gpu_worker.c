#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>

#include "hammer.h"
#include "hammer_connection.h"
#include "hammer_sched.h"
#include "hammer_handler.h"
#include "hammer_memory.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_macros.h"
#include "hammer_batch.h"

hammer_batch_buf_t *hammer_gpu_take_buf(hammer_batch_t *batch)
{
	if (batch->buf_has_been_taken == -1) {
		batch->buf_has_been_taken = batch->cur_buf_index;
		return batch->cur_buf;
	} else {
		hammer_err("error in hammer_gpu_take_buf\n");
		exit(0);
	}
}

int hammer_gpu_give_result(hammer_batch_t *batch)
{
	if (batch->processed_buf_index == -1) {
		batch->processed_buf_index = 
	} else {
		hammer_err("error in hammer_gpu_take_buf\n");
		exit(0);
	}
}

/* created thread, all this calls are in the thread context */
void *hammer_gpu_worker_loop(void *thread_sched)
{
	return 0;
}
