#ifndef HAMMER_BATCH_JOB_H
#define HAMMER_BATCH_JOB_H

#include "hammer_connection.h"

typedef struct hammer_batch_buf_s
{
	char *buf_base;
	hammer_job_t *job_list;

	int buf_size;
	int buf_length;

	int job_num;

} hammer_batch_buf_t;

typedef struct hammer_batch_s
{
	hammer_batch_buf_t buf0;
	hammer_batch_buf_t buf1;
	hammer_batch_buf_t gpu_result_buf;

	hammer_batch_buf_t *cur_buf;

	int cur_buf_index;

	/* GPU worker notify CPU worker 
	   buf_has_been_taken tell CPU worker which buf has just been taken,
	   processed_buf_index tell CPU worker which buf has been processed.
	   they all should be -1, if there are no events.
	   GPU write it (0/1), and CPU clears it to -1 to claim its own action.
	*/
	int processed_buf_index;
	int buf_has_been_taken;

} hammer_batch_t;

#endif
