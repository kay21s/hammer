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
int hammer_handler_connect(hammer_connection_t *c)
{
	struct sockaddr_in address;
	int ret, socket;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();
	hammer_connection_t *new_c;

	/* Get a connection and associate with the epoll event */
	new_c = hammer_get_connection();
	hammer_init_connection(new_c);

	new_c->socket = hammer_socket_create();
	new_c->type = HAMMER_CONN_RAW;
	new_c->rc = c;
	c->rc = new_c;

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = inet_addr(config->server_ip);
	address.sin_port = htons(config->server_port);

	ret = hammer_socket_connect(new_c->socket, (struct sockaddr *)&address, (socklen_t)sizeof(address));
	if (ret != 0) {
		printf("connect error\n");
		return -1;
	}

	/* Assign socket to worker thread */
	hammer_sched_add_connection(new_c, sched, HAMMER_CONN_CONNECTED);

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

hammer_connection_t *hammer_handler_accept(int server_socket)
{
	int remote_socket, ret;
	hammer_connection_t *c;

	remote_socket = hammer_socket_accept(server_socket);
	if (remote_socket < 0) {
		hammer_warn("error socket accept\n");
		return -1;
	}

	/* Set this socket non-blocking */
	hammer_socket_set_nonblocking(remote_socket);

	/* Get a connection and associate with the epoll event */
	c = hammer_get_connection();
	hammer_init_connection(c);

#if defined(SSL)
	if (config->ssl) {
		/* Accepted connection must be a SSL connection from client */
		c->ssl = 1;

		/* SSL initialization and accept */
		hammer_openssl_init(c);
		ret = hammer_openssl_accept(c);

		/* Get Parameters AES key, iv, hmac key, rounds */
		hammer_openssl_get_parameters(c);
	}
#endif

	return c;
}

// we delete both the two connections
int hammer_handler_error(hammer_connection_t *c)
{
	hammer_sched_del_connection(c);

	return 0;
}

// we delete both the two connections
int hammer_handler_close(hammer_connection_t *c)
{
	hammer_sched_del_connection(c);

	return 0;
}

/* Write to server, this is also used for writing to clients when we accelerate 
 * encryption and HMAC with GPU, and we just send the whole packet with this */
int hammer_handler_write(hammer_connection_t *c)
{
	int send;

	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	if (c->type != HAMMER_CONN_RAW) {
		hammer_err("What's up, this should not be a rtsp connection\n");
		exit(0);
	}

	// c->rc, this is the socket to write to, now we get the socket that has read something
	job_list = c->rc->job_list;
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);

		send = hammer_socket_write(
				c->socket, 
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

/* This read data from clients */
int hammer_handler_read(hammer_connection_t *c)
{
	int recv, available;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

//			hammer_epoll_state_set(sched->epoll_fd, socket,
//					HAMMER_EPOLL_READ,
//					HAMMER_EPOLL_LEVEL_TRIGGERED,
//					(EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLIN));

	available = c->body_size - c->body_length;
	if (available <= 0) {
		printf("small available buffer!\n");
		exit(0);
	}

	/* Read incomming data */
	if (c->type != HAMMER_CONN_RTSP) { 
		hammer_err("What's up, this should be a rtsp connection\n");
		exit(0)
	}

	recv = hammer_socket_read(
			c->socket,
			c->body_ptr + c->body_length,
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
		hammer_conn_job_add(c, recv);
		
		if (c->body_length + 1 >= c->body_size) {
			//hammer_session_remove(socket);
			printf("buffer full\n");
			return -1;
		}

		// activate the other socket to be write to
		if (c->r_conn == NULL) {
			// the connection has not been established, now we connect it
			hammer_handler_connect(c);
		}

		/* if we read packets from clients, we will forward it to server directly
		   which needs no batch operations. When GPU batch is needed, we will not  
		   forward packets received from server to client directly */
		if (config->gpu == 0) {
			hammer_epoll_change_mode(sched->epoll_fd,
					c->rc->socket,
					HAMMER_EPOLL_WRITE,
					HAMMER_EPOLL_LEVEL_TRIGGERED);
		}
	}

	return recv;
}

#if 0
/* read from clients, the SSL connection */
int hammer_handler_ssl_read(hammer_connection_t *c)
{
	int recv, available;
	hammer_sched_t *sched = hammer_sched_get_sched_struct();

//			hammer_epoll_state_set(sched->epoll_fd, socket,
//					HAMMER_EPOLL_READ,
//					HAMMER_EPOLL_LEVEL_TRIGGERED,
//					(EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLIN));

	available = c->body_size - c->body_length;
	if (available <= 0) {
		printf("small available buffer!\n");
		exit(0);
	}

	/* Read incomming data */
	if (!(c->ssl)) {
		hammer_err("What's up, this should be a ssl connection\n");
		exit(0);
	}

	recv = hammer_openssl_read(
			c,
			c->body_ptr + c->body_length,
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
		hammer_conn_job_add(c, recv);
		
		if (c->body_length + 1 >= c->body_size) {
			//hammer_session_remove(socket);
			printf("buffer full\n");
			return -1;
		}

		// activate the other socket to be write to
		if (c->r_conn == NULL) {
			// the connection has not been established, now we connect it
			hammer_handler_connect(c);
		}

		/* if we read packets from clients, we will forward it to server directly
		   which needs no batch operations. When GPU batch is needed, we will not  
		   forward packets received from server to client directly */
		if (config->gpu == 0) {
			hammer_epoll_change_mode(sched->epoll_fd,
					c->rc->socket,
					HAMMER_EPOLL_WRITE,
					HAMMER_EPOLL_LEVEL_TRIGGERED);
		}
	}

	return recv;
}

/* Write to clients, when we accelerate encryption with GPU, we do not use this function */
int hammer_handler_ssl_write(hammer_connection_t *c)
{
	int send;

	hammer_job_t *this_job;
	struct hammer_list *job_list, *job_head;

	if (!c->ssl) {
		hammer_err("What's up, this should be a ssl connection\n");
		exit(0);
	}

	// c->rc, this is the socket to write to, now we get the socket that has read something
	job_list = c->rc->job_list;
	hammer_list_foreach(job_head, job_list) {
		this_job = hammer_list_entry(job_head, hammer_job_t, _head);

		send = hammer_openssl_write(
				c, 
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
#endif

