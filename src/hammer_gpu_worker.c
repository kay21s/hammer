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

/* created thread, all this calls are in the thread context */
void *hammer_gpu_worker_loop(void *arg)
{
	int cpu_worker_num;
	hammer_timer_t t, counter, loopcounter;
	hammer_log_t log;

	hammer_timer_init(&t);
	hammer_timer_init(&counter);
	hammer_timer_init(&loopcounter);

	hammer_log_init(&log);

	// Counter for each kernel launch
	hammer_timer_reset(&loopcounter);
	hammer_timer_start(&loopcounter);
	
	for (int i = 0; i < config->iterations; i ++) {

		timeLog->loopMarker();

		hammer_timer_reset(&t);
		hammer_timer_start(&t);

		// Counter for the whole loop
		// From the second loop
		if (i == 2) {
			hammer_timer_reset(&counter);
			hammer_timer_start(&counter);
		}


		// Get Input Buffer from Client
		this_buffer_size = hammer_gpu_get_streams(input, buffer_size, 
			keys, ivs, pkt_offset, &this_stream_num);
		
		// Record the bytes processed for calculating the speed
		// From the second loop
		if (i > 1)
			bytes += this_buffer_size;

		if (this_stream_num > stream_num || this_stream_num < 256) {
			hammer_warn("What's the problem!!!!i\n");
			continue;
		}

		hammer_timer_stop(&t);
		hammer_log_msg(&log, "\n%s\n", "---------------------------", 0);
		hammer_log_timer(&log,
			"%s %f ms\n", "Get Streams Time",
			hammer_timer_get_total_time(&t);
			10,
			1);
		hammer_log_msg(&log, "%s %d streams\n", " This time we have ", this_stream_num);
		hammer_log_msg(&log, "%s %d byte\n", " This buffer size is ", this_buffer_size);


		//!!!!!!!!!!!!!!!!!!! BUG HERE: globalThreads cannot be greater than this_stream_num!!!!!!!!!!!!!!!!!!!
		// Get the reasonable global thread number
		if (this_stream_num / localThreads > 1) {
			if (this_stream_num % localThreads == 0) { // just equal
				globalThreads = this_stream_num;
			} else { // pad globalThreads to be a multiple of localThreads
				globalThreads = (this_stream_num / localThreads) * localThreads;
			} // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!why cannot +1 ? more than this_stream_num
		} else {
			globalThreads = localThreads;
		}

		//FIXME: 
		if (this_stream_num < 256) 
			globalThreads = localThreads = this_stream_num;

		/* -------------------------------------------------------------
		  This is a CPU/GPU synchronization point, as all commands in the
		in-order queue before the preceding cl*Unmap() are now finished.
		We can accurately sample the per-loop timer here.
		*/
		first = 1;
		do {
			elapsed_time = hammer_timer_get_elapsed_time(&loopcounter);
			if (first) {
				hammer_log_msg(&log, "\n%s %d\n", "<<<<<<<<Elapsed Time : ", elapsed_time);
				first = 0;
			}

			if (elapsed_time - time_point > 1) { // surpassed the time point more than 1 ms
				//std::cout << "Timepoint Lost! " << elapsed_time << "/" << time_point << std::endl;
				hammer_log_msg(&log, "\n%s %d\n", ">>>>>>>>Time point lost!!!! : ", elapsed_time);
				break;
			}
		} while(abs(elapsed_time - time_point) > 1);
		//std::cout << elapsed_time << "  " << time_point << std::endl;
		hammer_timeLog_msg(&log, "%s %d\n", ">>>>>>>>Time point arrived : ", elapsed_time);
		hammer_timer_reset(&loopcounter);
		hammer_timer_start(&loopcounter);

		
		/* ------------------------------------------------------------- */




		hammer_log_msg(&log, "%s %d\n", "global Threads:", globalThreads);
		hammer_log_msg(&log, "%s %d\n", "this_stream_number:", this_stream_num);
//		std::cout<<"Global threads: " << globalThreads 
//			<< "\n Local Thread: "<< localThreads 
//			<< "\n this_stream_num£º " << this_stream_num << std::endl;

		hammer_timer_reset(&t);
		hammer_timer_start(&t);

		// Write buffers
		//////////////////////////////////////////

		hammer_timer_stop(&t);
		hammer_log_timer(&log,
			"%s %f ms\n", "Input Data Transfer Time", 
			hammer_timer_get_total_time(&t), 
			10, 1);

		hammer_timer_reset(&t);
		hammer_timer_start(&t);

		// Set appropriate arguments to the kernel
		//////////////////////////////////////////


		//Enqueue a kernel run call.
		//////////////////////////////////////////

		hammer_timer_stop(&t);
		hammer_log_timer(&log,
			"%s %f ms\n", "Execution Time", 
			hammer_timer_get_total_time(&t), 
			10, 1);

		hammer_timer_reset(&t);
		hammer_timer_start(&t);

		/* Enqueue the results to application pointer*/
		//////////////////////////////////////////

		hammer_timer_stop(&t);
		hammer_log_timer(&log,
			"%s %f ms\n", "Output Data Time",
			hammer_timer_get_total_time(&t), 
			10, 1);

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
