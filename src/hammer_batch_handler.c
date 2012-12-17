#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <unistd.h>

#include "hammer_connection.h"
#include "hammer_sched.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_socket.h"
#include "hammer_list.h"
#include "hammer_macros.h"
#include "hammer_batch.h"
#include "hammer.h"

int hammer_batch_init()
{
	hammer_batch_t *batch = hammer_sched_get_batch_struct();
	
	batch->buf0.buf_base = hammer_mem_malloc(config->batch_buf_max_size);
	batch->buf0.job_list = hammer_mem_malloc(config->batch_job_max_num * sizeof(hammer_job_t));
	batch->buf0.size = config->batch_buf_max_size;
	batch->buf0.length = 0;
	batch->buf0.job_num = 0;

	batch->buf1.buf_base = hammer_mem_malloc(config->batch_buf_size);
	batch->buf1.job_list = hammer_mem_malloc(config->batch_job_max_num * sizeof(hammer_job_t));
	batch->buf1.size = config->batch_buf_max_size;
	batch->buf1.length = 0;
	batch->buf1.job_num = 0;

	batch->gpu_result_buf.buf_base = hammer_mem_malloc(config->batch_buf_size);
	batch->gpu_result_buf.job_list = hammer_mem_malloc(config->batch_job_max_num * sizeof(hammer_job_t));
	batch->gpu_result_buf.size = config->batch_buf_max_size;
	batch->gpu_result_buf.length = 0;
	batch->gpu_result_buf.job_num = 0;

	batch->cur_buf = &(batch->buf0);
	batch->cur_buf_index = 0;

	batch->buf_can_fetch = 0; // currently buf0 are the first one we are working with
	batch->processed_buf_index = -1;
	batch->buf_has_been_taken = -1;

	return 0;
}

int hammer_batch_if_gpu_processed_new(hammer_batch_t *batch)
{
	if (batch->processed_buf_index == -1) {
		return 0;
	} else if (batch->processed_buf_index == 0 || batch->processed_buf_index == 1) {
		return 1;
	} else {
		hammer_err("error processed_buf_index\n");
		exit(0);
	}
}

int hammer_batch_if_current_buf_taken(hammer_batch_t *batch)
{
	if (batch->buf_has_been_taken != -1) {
		/* This buf has been taken */
		return 1;
	} else {
		return 0;
	}
}

int hammer_batch_switch_buffer(hammer_batch_t *batch)
{
	if (batch->cur_buf_index == 0) {
		batch->cur_buf = &(batch->buf1);
		batch->cur_buf_index = 1;
	} else {
		batch->cur_buf = &(batch->buf0);
		batch->cur_buf_index = 0;
	}

	/* mark this event has been processed, and buf is switched*/
	batch->buf_has_been_taken = -1;

	return 0;
}

int hammer_batch_job_add(hammer_batch_t *batch, hammer_connection_t *c, int length)
{
	int i = batch->cur_buf->job_num;
	hammer_job_t *new_job = &(batch->cur_buf->job_list[i]);
	
	new_job->job_body_ptr = batch->cur_buf->buf_base + batch->cur_buf->buf_length;
	new_job->job_body_length = length;
	new_job->connection = conn;

	batch->cur_buf->buf_length += length;
	batch->cur_buf->job_num ++;

	if (batch->cur_buf->buf_length >= batch->cur_buf->buf_size) {
		hammer_err("error in batch job add\n");
		exit(0);
	}

	return 0;
}

/* We don't batch read from clients, which all need decryption.
   And these are supposed to be only requests, therefore we do not bother GPU to handle this
   while CPU is competent for fast AES operation with small amount of data -- AES-NI 

   However, we batch the read from server, since they all need encryption
*/
int hammer_batch_handler_read(hammer_connection_t *c)
{
	int recv, available, if_gpu_fetched;
	hammer_connection_t *rc;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();
	hammer_batch_t *batch = hammer_sched_get_batch_struct();

//			hammer_epoll_state_set(sched->epoll_fd, socket,
//					HAMMER_EPOLL_READ,
//					HAMMER_EPOLL_LEVEL_TRIGGERED,
//					(EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLIN));

	/* we batch ssl encryption/decryption */
	if (!c->ssl) {
		hammer_err("this should be an ssl connection\n");
		exit(0);
	}

	/* if the GPU worker has fetched the data,
	   we will switch our buffer, a two-buffer strategy.
	   */ 
	if (hammer_batch_if_current_buf_taken(batch)) {
		hammer_batch_switch_buffer(batch);
	}

	available = batch->cur_buf->buf_size - batch->cur_buf->buf_length;
	if (available <= 0) {
		printf("small available buffer!\n");
		exit(0);
	}

	/* Read incomming data */
	recv = hammer_openssl_read_undecrypted(
			c->socket,
			batch->cur_buf->buf_base + batch->cur_buf->buf_length,
			available);
	if (recv <= 0) {
		// FIXME
		//if (errno == EAGAIN) {
		//	return 1;
		//} else {
			//hammer_session_remove(socket);
			printf("Hey!!!\n");
			return -1;
		//}

	}
	
	/* Batch this job */
	hammer_batch_job_add(batch, c, recv);

	/* Revoke its r_conn  */
	{
		if (c->r_conn == NULL) {
			// the connection has not been established, now we connect it
			hammer_handler_connect(c);
		}
		rc = c->r_conn;
	}

	return 0;
}

/* This function trigger write event of all the jobs in this batch */
int hammer_batch_forwarding(hammer_batch_t *batch)
{
	int i;
	hammer_connection_t *rc;
	hammer_job_t *this_job;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();
	hammer_batch_buf_t *buf;

	/* Get the buf that has been processed by GPU */
	if (batch->processed_buf_index == 0 || batch->processed_buf_index == 1) {
		buf = &(batch->gpu_result_buf)
	} else {
		hammer_err("error processed_buf_index\n");
		exit(0);
	}

	/* Set each connection to forward */
	for (i = 0; i < buf->job_num; i ++) {
		this_job = &(buf->job_list[i]);
		rc = this_job->connection->r_conn;

		hammer_epoll_change_mode(sched->epoll_fd,
				rc->socket,
				HAMMER_EPOLL_WRITE,
				HAMMER_EPOLL_LEVEL_TRIGGERED);
	}

	/* Mark this event has been processed */
	batch->processed_buf_index = -1;

	return 0;
}
