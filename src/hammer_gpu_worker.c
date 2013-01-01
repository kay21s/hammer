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
		/* just mark there is a buf been processed */
		batch->processed_buf_index = 1;
	} else {
		hammer_err("error in hammer_gpu_take_buf\n");
		exit(0);
	}
}

void hammer_gpu_get_batch(hammer_batch_buf_t **buf_set)
{
	int i;
	hammer_batch_t *batch;

	for (i = 0; i < config->cpu_worker_num; i ++) {
		batch = &(batch_set[i]);
		buf_set[i] = hammer_gpu_take_buf(batch);
	}

	return ;
}

/* created thread, all this calls are in the thread context */
void *hammer_gpu_worker_loop(void *arg)
{
	hammer_timer_t t, counter, loopcounter;
	hammer_log_t log;
	hammer_batch_buf_t **buf_set;

	buf_set = (hammer_batch_buf_t **)malloc(config->cpu_worker_num * sizeof(hammer_batch_buf_t *));
	
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
	crypto_context_t cry_ctx;
	crypto_context_init(&cry_ctx, input_size, output_size, config->cpu_worker_num);

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
		bytes = hammer_gpu_get_batch(buf_set);
		
		/* Record the bytes processed for calculating the speed, from the second loop */
		if (i > 1)	bytes += this_buffer_size;

		/* We launch each cpu worker batch as a stream*/
		for (cuda_stream_id = 0; cuda_stream_id < config->cpu_worker_num; cuda_stream_id ++) {
			buf_t = buf_set[cuda_stream_id];

			crypto_context_sha1_aes_encrypt(
				&cry_ctx;
				buf_t->input_buf,
				stream->input_d,
				param.in_pos,
				param.key_pos,
				param.ivs_pos,
				param.pkt_offset_pos,
				param.tot_in_len,
				param.out,
				param.num_flows,
				param.tot_out_len,
				0);
			crypto_context_sync(0); /* wait for completion */
		}


		hammer_timer_stop(&t);
		hammer_log_msg(&log, "\n%s\n", "---------------------------", 0);
		hammer_log_timer(&log, "%s %f ms\n", "Get Input Time",
			hammer_timer_get_total_time(&t), 10, 1);
		hammer_log_msg(&log, "%s %d streams\n", " This time we have ", this_stream_num);
		hammer_log_msg(&log, "%s %d byte\n", " This buffer size is ", this_buffer_size);

		// Write buffers
		//////////////////////////////////////////
		hammer_timer_restart(&t);

		hammer_timer_stop(&t);
		hammer_log_timer(&log,"%s %f ms\n", "Input Data Transfer Time", 
			hammer_timer_get_total_time(&t), 10, 1);
		hammer_timer_restart(&t);

		// Set appropriate arguments to the kernel
		//////////////////////////////////////////


		//Enqueue a kernel run call.
		//////////////////////////////////////////

		hammer_timer_stop(&t);
		hammer_log_timer(&log, "%s %f ms\n", "Execution Time",
			hammer_timer_get_total_time(&t), 10, 1);
		hammer_timer_restart(&t);

		/* Enqueue the results to application pointer*/
		//////////////////////////////////////////

		hammer_timer_stop(&t);
		hammer_log_timer(&log,"%s %f ms\n", "Output Data Time", 
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
