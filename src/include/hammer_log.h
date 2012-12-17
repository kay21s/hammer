#ifndef HAMMER_LOG_H
#define HAMMER_LOG_H

typedef hammer_log_sample_s {
	bool          isMsg;
	bool          isErr;
	double        timer;
	unsigned int  nbytes;
	int           loops;
	char *        fmt;
	char *        msg;
	int           num;
} hammer_log_sample_t;

typedef hammer_log_s {
	unsigned int idx;
	unsigned int loops;
	unsigned int loop_entries;
	unsigned int loop_timers;
 
	hammer_log_sample_t *samples;
} hammer_log_t;

void hammer_log_init(hammer_log_t *log);
void hammer_log_loop_marker(hammer_log_t *log);
void hammer_log_msg(hammer_log_t *log, const char *format, const char *msg, const int num);
void hammer_log_timer(hammer_log_t *log, const char *format, const char *msg, double timer, unsigned int nbytes, int loops);
void hammer_log_print(hammer_log_t *log);
#endif
