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

/* connect to server */
int hammer_handler_connect(hammer_connection_t *conn)
{
	struct sockaddr_in address;
	int ret, socket;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

	socket = hammer_socket_create();

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = inet_addr(config->server_ip);
	address.sin_port = htons(config->server_port);

	ret = hammer_socket_connect(socket,(struct sockaddr *)&address, (socklen_t)sizeof(address));
	if (ret != 0) {
		printf("connect error\n");
		return -1;
	}

	/* Assign socket to worker thread */
	ret = hammer_sched_add_connection(socket, sched, conn);
	if (ret == -1) {
		hammer_socket_close(socket);
	}

	return 0;
}

int hammer_handler_listen()
{
	int socket, ret;
	struct sockaddr_in proxy_address;

	socket = hammer_socket_create();
	if (socket < 0) {
		hammer_warn("socket create failure\n");
		return -1;
	}

	proxy_address.sin_family = AF_INET;
	//proxy_address.sin_addr.s_addr = inet_addr(config->listen_ip);
	proxy_address.sin_port = htons(config->listen_port);

	ret = hammer_socket_bind(socket, (struct sockaddr *)&proxy_address, sizeof(proxy_address));
	if (ret == -1) {
		hammer_warn("error bind socket\n");
		return -1;
	}

	ret = hammer_socket_listen(socket, HAMMER_MAX_CONN);
	if (ret == -1) {
		hammer_warn("error listen socket\n");
		return -1;
	}
	return socket;
}

int hammer_handler_accept(int server_socket)
{
	int remote_socket;

	remote_socket = hammer_socket_accept(server_socket);
	if (remote_socket < 0) {
		hammer_warn("error socket accept\n");
		return -1;
	}

	/* Set this socket non-blocking */
	hammer_socket_set_nonblocking(remote_socket);

	return remote_socket;
}

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
	int recv, available;
	hammer_connection_t *r_conn;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

//			hammer_epoll_state_set(sched->epoll_fd, socket,
//					HAMMER_EPOLL_READ,
//					HAMMER_EPOLL_LEVEL_TRIGGERED,
//					(EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLIN));

	available = conn->body_size - conn->body_length;
	if (available <= 0) {
		printf("small available buffer!\n");
		exit(0);
	}

	/* Read incomming data */
	recv = hammer_socket_read(
			conn->socket,
			conn->body_ptr + conn->body_length,
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

	} else if (recv > 0) {
		hammer_conn_job_add(conn, recv);
		
		if (conn->body_length + 1 >= conn->body_size) {
			//hammer_session_remove(socket);
			printf("buffer full\n");
			return -1;
		}

		// activate the other socket to be write to
		if (conn->r_conn == NULL) {
			// the connection has not been established, now we connect it
			hammer_handler_connect(conn);
		}
		r_conn = conn->r_conn;

		hammer_epoll_change_mode(sched->epoll_fd,
			r_conn->socket,
			HAMMER_EPOLL_WRITE, HAMMER_EPOLL_LEVEL_TRIGGERED);
	}

	return recv;
}


int hammer_handler_write(hammer_connection_t *conn)
{
	int send;
	hammer_connection_t *r_conn;

	// this is the socket to write to, now we get the socket that has read something
	r_conn = conn->r_conn;

	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	job_list = r_conn->job_list;
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);

		send = hammer_socket_write(
			conn->socket, 
			this_job->job_body_ptr, 
			this_job->job_body_length);

		if (send != this_job->job_body_length) {
			printf("Not all are send \n");
			return -1;
		}

		hammer_conn_job_del(this_job);
	}

	return 0;
}

