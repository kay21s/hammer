#include "hammer_timer.h"

#include <sys/time.h>
#include <time.h>

int hammer_timer_init()
{
	freq = 1000;

	return 0;
}

int hammer_timer_start(hammer_timer_t *timer)
{
	struct timespec s;
	clock_gettime(CLOCK_REALTIME, &s);
	timer->start = (uint64_t)s.tv_sec * 1e9 + (uint64_t)s.tv_nsec;

	return 0;
}

int hammer_timer_restart(hammer_timer_t *timer)
{
	struct timespec s;
	clock_gettime(CLOCK_REALTIME, &s);
	timer->start = (uint64_t)s.tv_sec * 1e9 + (uint64_t)s.tv_nsec;

	_clocks = 0;

	return 0;
}

int hammer_timer_stop(hammer_timer_t *timer)
{
	uint64_t n;

	struct timespec s;
	clock_gettime(CLOCK_REALTIME, &s);
	n = (uint64_t)s.tv_sec * 1e9 + (uint64_t)s.tv_nsec;

	n -= timer->start;
	timer->start = 0;
	timer->clocks += n;

	return 0;
}

int hammer_timer_reset(hammer_timer_t *timer)
{
	_clocks = 0;

	return 0;
}

double hammer_timer_get_total_time(hammer_timer_t *timer)
{
	//returns millisecond as unit -- second * 1000
	return (double)(timer->clocks * 1000) / (double) 1e9;
}

double hammer_timer_get_elapsed_time(hammer_timer_t *timer)
{
	uint64_t n;

	struct timespec s;
	clock_gettime(CLOCK_REALTIME, &s);
	n = (uint64_t)s.tv_sec * 1e9 + (uint64_t)s.tv_nsec;

	return (double)((n - timer->start) * 1000) / (double) 1e9;
}

