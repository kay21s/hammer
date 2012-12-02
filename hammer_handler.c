#include "hammer_connection.h"
#include "hammer_sched.h"
#include "hammer_epoll.h"
#include "hammer_config.h"
#include "hammer_socket.h"
#include "hammer_list.h"

int hammer_handler_connect(hammer_connection_t *conn)
{
	struct sockaddr address;
	int ret, socket;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

	socket = hammer_socket_create(AF_INET, SOCK_STREAM, 0);

	address.sin_family = PF_INET;
	address.sin_addr.s_addr = inet_addr(config->server_ip);
	address.sin_port = htons(config->server_port);

	ret = hammer_socket_connect(socket, &address, sizeof(address));
	if (ret != 0) {
		printf("connect error\n");
		return -1;
	}

	/* Assign socket to worker thread */
	ret = hammer_sched_add_connection(remote_fd, sched, NULL);
	if (ret == -1) {
		hammer_socket_close(remote_fd);
	}

	return 0;
}

/*
int hammer_handler_accept()
{
	return ret;
}*/

// we delete both the two connections
int hammer_handler_error(hammer_connection_t *conn)
{
	hammer_sched_del_connection(conn);

	return 0;
}

// we delete both the two connections
int hammer_handler_close(hammer_connection_t *conn)
{
	hammer_sched_del_connection(conn);

	return 0;
}

int hammer_handler_read(hammer_connection_t *conn)
{
	int ret;
	int bytes;
	struct hammer_connection *r_conn;

//			hammer_epoll_state_set(sched->epoll_fd, socket,
//					HAMMER_EPOLL_READ,
//					HAMMER_EPOLL_LEVEL_TRIGGERED,
//					(EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLIN));

	available = conn->body_size - conn->body_length;
	if (available <= 0) {
		printf("small available buffer!\n");
	}

	/* Read incomming data */
	bytes = hammer_socket_read(
			conn->socket,
			conn->body_ptr + conn->body_length;
			available);

	if (bytes <= 0) {
		// FIXME
		if (errno == EAGAIN) {
			return 1;
		} else {
			//hammer_session_remove(socket);
			printf("Hey!!!\n");
			return -1;
		}

	} else if (bytes > 0) {
		hammer_conn_job_add(conn, bytes);
		
		if (conn->body_length + 1 >= conn->body_size) {
			//hammer_session_remove(socket);
			printf("buffer full\n");
			return -1;
		}

		// activate the other socket to be write to
		r_conn = conn->r_conn;
		if (r_conn == NULL) {
			// the connection has not been established, now we connect it
			r_conn = hammer_proxy_connect(conn);
		}


		hammer_epoll_change_mode(sched->epoll_fd,
			r_conn->socket,
			HAMMER_EPOLL_WRITE, HAMMER_EPOLL_LEVEL_TRIGGERED);
	}

	return bytes;
}


int hammer_handler_write(hammer_connection_t *conn)
{
	int ret = -1;
	hammer_connection_t *r_conn

	// this is the socket to write to, now we get the socket that has read something
	r_conn = conn->r_conn;

	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	job_list = hammer_conn_get_job_list(r_conn);
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);

		bytes_send = hammer_socket_write(
			conn->socket, 
			this_job->job_body_ptr, 
			this_job->job_body_length);

		if (bytes_send != this_job->job_body_length) {
			printf("Not all are send \n")
				return -1;
		}

		hammer_conn_job_del(this_job);
	}

	return 0;
}

