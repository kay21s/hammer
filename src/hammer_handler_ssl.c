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
#include "hammer.h"


int hammer_handler_ssl_accept(hammer_connection_t *c)
{
	int ret;
	ret = hammer_openssl_accept(c);

	return ret;
}

void hammer_handler_ssl_initialize(hammer_connection_t *c)
{
	hammer_openssl_initialize(c);
	return;
}

int hammer_handler_ssl_read(hammer_connection_t *c)
{
	int recv, available;
	hammer_connection_t *rc;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

	available = c->body_size - c->body_length;
	if (available <= 0) {
		printf("small available buffer!\n");
		exit(0);
	}

	buffer = c->body_ptr + c->body_length;

	recv = hammer_openssl_read (c, buffer, available);
	if (recv <= 0) {
		printf("Hey!!!\n");
		exit(0);

	} else if (recv > 0) {
		buffer[recv] = '\0';

		// add a job for forwarding, body_length + recv
		hammer_conn_job_add(c, recv);
		
		if (c->body_length + 1 >= c->body_size) {
			//hammer_session_remove(socket);
			printf("buffer full\n");
			return -1;
		}

		// activate the other socket to be write to
		if (c->rc == NULL) {
			// the connection has not been established, now we connect it
			hammer_handler_connect(c);
		}
		rc = c->rc;

		hammer_epoll_change_mode(sched->epoll_fd,
			rc->socket,
			HAMMER_EPOLL_WRITE, HAMMER_EPOLL_LEVEL_TRIGGERED);
	}

	return recv;
}

int hammer_handler_ssl_write()
{
	int send;
	hammer_connection_t *rc;

	// this is the socket to write to, now we get the socket that has read something
	rc = c->rc;

	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	job_list = rc->job_list;
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);

		send = hammer_openssl_write(
			c, 
			this_job->job_body_ptr, 
			this_job->job_body_length);

		if (send != this_job->job_body_length) {
			printf("Not all are send \n");
			exit(0);
		}

		hammer_conn_job_del(this_job);
	}

	return send;
}
