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

hammer_batch_buf_t *hammer_gpu_get_batch_one(hammer_batch_t *batch, int *size)
{
	if (batch->buf_has_been_taken == -1) {

		pthread_mutex_lock(&mutex_batch_launch);
		batch->buf_has_been_taken = batch->cur_buf_index;
		pthread_mutex_unlock(&mutex_batch_launch);

		*size = batch->cur_buf->buf_length;

		return batch->cur_buf;
	} else {
		hammer_err("error in hammer_gpu_take_buf\n");
		exit(0);
	}

	return NULL;
}

/* Get the buffer of each CPU worker at each time interval I */
void hammer_gpu_get_batch(hammer_gpu_worker_t *g, hammer_batch_buf_t *batch_set)
{
	int i;
	hammer_batch_t *batch;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		batch = &(batch_set[i]);
		g->buf_set[i] = hammer_gpu_get_batch_one(batch, &(g->buf_size[i]));
		g->total_bytes += g->buf_size[i];
	}

	return ;
}

/* Tell the CPU worker that this batch has been completed */
int hammer_gpu_give_result_one(hammer_batch_t *batch)
{
	if (batch->processed_buf_index == -1) {
		/* just mark there is a buf been processed */
		pthread_mutex_lock(&mutex_batch_complete);
		batch->processed_buf_index = 1;
		pthread_mutex_unlock(&mutex_batch_complete);
	} else {
		hammer_err("error in hammer_gpu_take_buf\n");
		exit(0);
	}

	return 0;
}

void hammer_gpu_give_result(hammer_gpu_worker_t *g, hammer_batch_buf_t *batch_set)
{
	int i;
	hammer_batch_t *batch;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		batch = &(batch_set[i]);
		hammer_gpu_give_result_one(batch);
	}

	return ;
}

void hammer_gpu_worker_init(hammer_gpu_worker_t *g)
{
	g->buf_set = (hammer_batch_buf_t **)malloc(config->cpu_worker_num * sizeof(hammer_batch_buf_t *));
	
	/* Initialize variables in libgpucrypto */
	/* There is a one-to-one mapping from CPU Worker's HAMMER_BATCH_T
	 * to GPU Worker's CUDA_STREAM_T
	 * The pinned memory buffers are in $batch$, while their corresponding
	 * device buffers are in $stream$
	 */
	uint32_t input_size = config->batch_buf_max_size +
		config->batch_job_max_num * AES_KEY_SIZE +
		config->batch_job_max_num * PKT_OFFSET_SIZE + // input buffer
		config->batch_job_max_num * AES_IV_SIZE;
	uint32_t output_size = config->batch_buf_max_size;
	crypto_context_init(&(g->cry_ctx), input_size, output_size, config->cpu_worker_num);


	/* After waiting for I, we first take buffer set A (buf_id = 0), which has been filled with
	 * jobs by CPU workers*/
	 g->buf_set_id = 0;

	 return;
}

/* created thread, all this calls are in the thread context */
void *hammer_gpu_worker_loop(void *arg)
{
	hammer_timer_t t, counter, loopcounter;
	hammer_log_t log;

	/* Initialize GPU worker */
	hammer_gpu_worker_t g;
	hammer_gpu_worker_init(&g);

	/* Init timers */
	hammer_timer_init(&t);
	hammer_timer_init(&counter);
	hammer_timer_init(&loopcounter);
	hammer_log_init(&log);

	/* Timers for each kernel launch */
	hammer_timer_restart(&loopcounter);
	
	for (int i = 0; i < config->iterations; i ++) {
		timeLog->loopMarker();

		/* Counter for the whole loop, from the second loop */
		if (i == 2)	hammer_timer_restart(&counter);

		// Wait for 'I', synchronization point
		//////////////////////////////////////////
		/* This is a CPU/GPU synchronization point, as all commands in the
		 * in-order queue before the preceding cl*Unmap() are now finished.
		 * We can accurately sample the per-loop timer here.
		 */
		first = 1;
		do {
			elapsed_time = hammer_timer_get_elapsed_time(&loopcounter);
			if (first) {
				hammer_log_msg(&log, "\n%s %d\n", "<<<<<<<<Elapsed Time : ", elapsed_time);
				first = 0;
			}

			if (elapsed_time - time_point > 1) { // surpassed the time point more than 1 ms
				hammer_log_msg(&log, "\n%s %d\n", ">>>>>>>>Time point lost!!!! : ", elapsed_time);
				break;
			}
		} while (abs(elapsed_time - time_point) > 1);

		hammer_timeLog_msg(&log, "%s %d\n", ">>>>>>>>Time point arrived : ", elapsed_time);
		hammer_timer_restart(&loopcounter);



		// Get Buffers
		//////////////////////////////////////////
		hammer_timer_restart(&t);

		/* Get Input Buffer from CPU Workers */
		hammer_gpu_get_batch(g.buf_set);
		

		hammer_timer_stop(&t);
		hammer_log_msg(&log, "\n%s\n", "---------------------------", 0);
		hammer_log_timer(&log, "%s %f ms\n", "Get Input Time",
			hammer_timer_get_total_time(&t), 10, 1);
		hammer_log_msg(&log, "%s %d streams\n", " This time we have ", this_stream_num);
		hammer_log_msg(&log, "%s %d byte\n", " This buffer size is ", this_buffer_size);

		//Enqueue a kernel run call.
		//////////////////////////////////////////
		hammer_timer_restart(&t);

		/* We launch each cpu worker batch as a stream*/
		for (cuda_stream_id = 0; cuda_stream_id < config->cpu_worker_num; cuda_stream_id ++) {
			buf_t = buf_set[cuda_stream_id];

			crypto_context_sha1_aes_encrypt (
				&cry_ctx;
				buf_t->input_buf,
				buf_t->output_buf,
				0, // in_pos
				buf_t->aes_keys_pos,
				buf_t->pkt_offset_pos,
				buf_t->ivs_pos,
				buf_t->buf_size, // input buffer size
				buf_t->buf_length +, // output buffer size FIXME ???
				buf_t->job_num,
				cuda_stream_id,
				128);

			/* Wait for transfer completion */
			crypto_context_sync(&cry_ctx, cuda_stream_id, buf_t->output_buf, 1, 1);
		}

		hammer_timer_stop(&t);
		hammer_log_timer(&log, "%s %f ms\n", "Execution Time",
			hammer_timer_get_total_time(&t), 10, 1);

		hammer_log_msg(&log, "%s %dth iteration\n", "This is", i);
		//if (i > 1)	timeLog->Msg( "%s %f ms\n", "Time after is", counter.GetElapsedTime());
	}

	hammer_timer_stop(&counter);
	printf("End of execution, now the program costs : %d ms\n", hammer_timer_get_total_time(&counter));
	printf("Processing speed is %.2f Mbps\n", (bytes * 8) / (1e3 * hammer_timer_get_total_time(&counter)));

	uint64_t speed = (AVG_RATE/1000) * (STREAM_NUM/1000);
	printf("Theoretical speed is %lld Mbps\n", speed);

	return 0;
}
