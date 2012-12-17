#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "hammer_log.h"
#include "hammer_memory.h"

// Add "num" variable based on original version
void hammer_sample_set_msg(hammer_log_sample_t *sample, const char *fmt, const char *msg, int num)
{
	sample->isMsg = true;

	sample->fmt = hammer_mem_malloc(strlen(fmt)+1);
	strcpy(sample->fmt, fmt);

	sample->msg = hammer_mem_malloc(strlen(msg)+1);
	strcpy(sample->msg, msg);

	sample->num = num;
}

void hammer_sample_set_timer(hammer_log_sample_t *sample, const char *fmt, const char *msg, double timer, unsigned int nbytes, int loops)
{
	sample->isMsg = false;
	sample->timer = timer;

	if (loops != 0)	sample->loops = loops;
	if (nbytes > 0)	sample->nbytes = nbytes;

	if (strlen(msg) > 0) {
		sample->fmt = hammer_mem_malloc(strlen( fmt ) + 1);
		strcpy(sample->fmt, fmt);
	}

	if (strlen(msg) > 0) {
		sample->msg = hammer_mem_malloc(strlen( msg ) + 1);
		strcpy(sample->msg, msg);
	}
}

void hammer_sample_print(hammer_log_sample_t *sample)
{
	if(sample->isMsg == true) {
		printf(sample->fmt, sample->msg, sample->num);
	} else {
		double bwd = (((double) sample->nbytes * sample->loops )/ sample->timer) / 1e9;
		printf(sample->fmt, sample->msg, sample->timer, bwd) ;
	}
}

/* ---------------------------------------------------------------------- */

void hammer_log_init(hammer_log_t *log)
{
	log->idx = 0;
	log->loops = 0;
	log->loop_entries = 0;
	log->loop_timers = 0;
	log->samples = hammer_mem_malloc(config->log_sample_num * sizeof(hammer_log_sample_t))
}

void hammer_log_loop_marker(hammer_log_t *log)
{
	log->loop_timers = 0;
	log->loops ++;
}

void hammer_log_msg(hammer_log_t *log, const char *format, const char *msg, const int num)
{
	hammer_sample_set_msg(&(log->samples[log->idx ++]), format, msg, num);
	log->loop_entries ++;
}

void hammer_log_timer(hammer_log_t *log, const char *format, const char *msg, double timer, unsigned int nbytes, int loops)
{
	hammer_sample_set_timer(&(log->samples[log->idx ++]), format, msg, timer, nbytes, loops);
	log->loop_entries ++;
	log->loop_timers ++;
}

void hammer_log_print(hammer_log_t *log)
{
	int i;

	for(i = 0; i < log->loop_entries; i++) {
		hammer_sample_print(log->samples[i]);
	}
}

#if 0
void hammer_log_print_summary(hammer_log_t *log, int skip)
{
	int i, nl, current;
	double sum;

	for(i = 0; i < log->loop_entries; i++)
	{
		if(log->samples[i].isMsg())
		{
			bool foundError = false;

			for(nl = 0; nl < log->loops; nl++)
			{
				current = i + nl * log->loop_entries;

				if(log->samples[current].isErr())
				{
					log->samples[current].printSample();
					foundError = true;
					break;
				}
			}

			if(!foundError)
				log->samples[i].printSample();
		}
		else
		{
			sum = 0;

			for(nl = skip; nl < log->loops; nl++)
			{
				sum += log->samples[i + nl * log->loop_entries].getTimer();
			}

			log->samples[i].setTimer("", "", sum / (log->loops-skip), 0, 0);
			log->samples[i].printSample();
		}
	}
}
#endif
