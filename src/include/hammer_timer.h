#ifndef HAMMER_TIMER_H
#define HAMMER_TIMER_H

#include <stdint.h>

/**
 * \file Timer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 * FIXME:
 * 1s = 1000ms (millisecond)
 * 1ms = 1000us (microsecond)
 * 1us = 1000ns (nanosecond)
 * this counter returns in terms of us
 */


typedef hammer_timer_s {

    uint64_t freq;
    uint64_t clocks;
    uint64_t start;
} hammer_timer_t;

int hammer_timer_init();
int hammer_timer_start(hammer_timer_t *timer);
int hammer_timer_stop(hammer_timer_t *timer);
int hammer_timer_reset(hammer_timer_t *timer);
double hammer_timer_get_total_time(hammer_timer_t *timer);
double hammer_timer_get_elapsed_time(hammer_timer_t *timer);

#endif

