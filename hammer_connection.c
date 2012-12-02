#include "hammer_connection.h"
#include "hammer_macros.h"
#include "hammer_list.h"
#include "hammer_sched.h"
#include "hammer_memory.h"
#include "hammer_config.h"
#include "hammer_socket.h"

int hammer_conn_job_add(hammer_connection_t *conn, int length)
{
	hammer_job_t *new_job = hammer_mem_malloc(sizeof(hammer_job_t));
	
	new_job->job_body_ptr = conn->body_ptr + conn->body_length;
	new_job->job_body_length = length;

	hammer_list_add(&(new_job->_head), conn->job_list);

	conn->body_length += length;

	return 0;
}

int hammer_conn_job_del(hammer_job_t *job)
{
	hammer_list_del(&(job->_head));
	hammer_mem_free(job);

	return 0;
}



// connection management
void hammer_init_connection(hammer_connection_t *conn)
{
	conn->socket = 0;
	conn->ssl = 0; // ssl not enabled by default
	conn->body_ptr = hammer_mem_malloc(config->conn_buffer_size);
	conn->body_size = config->conn_buffer_size;
	conn->body_length = 0;
	conn->r_conn = NULL;
	conn->job_list = NULL;

	return;
}

hammer_connection_t *hammer_get_connection()
{
	return hammer_mem_malloc(sizeof(hammer_connection_t));
}

void hammer_free_connection(hammer_connection_t *conn)
{
	hammer_mem_free(conn);
	return;
}

int hammer_close_connection(hammer_connection_t *conn)
{
	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	if (conn == NULL) {
		return 0;
	}

	hammer_socket_close(conn->socket);

	job_list = conn->job_list;
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);
		hammer_conn_job_del(this_job);
	}

	hammer_free_connection(conn);

	return 0;
}

/*
int hammer_conn_get_socket(hammer_connection_t *conn)
{
	return conn->socket;
}

int hammer_conn_get_body_length(hammer_connection_t *conn)
{
	return conn->body_length;
}

int hammer_conn_get_body_size(hammer_connection_t *conn)
{
	return conn->body_size;
}

char *hammer_conn_get_body_ptr(hammer_connection_t *conn)
{
	return conn->body_ptr;
}

hamme_list_t *hammer_conn_get_job_list(hammer_connection_t *conn)
{
	return conn->job_list;
}
hammer_connection_t *hammer_conn_get_reverse_conn(hammer_connection_t *conn)
{
	return conn->r_conn;	
}

int hammer_conn_get_available_space(hammer_connection_t *conn)
{
	return conn->body_size - conn->body_length;
}

char *hammer_conn_get_current_ptr(hammer_connection_t *conn)
{
	return conn->body_ptr + conn->body_length;
}

void hammer_conn_set_body_length(hammer_connection_t *conn, int length)
{
	conn->body_length = length;
	return;
}
*/
