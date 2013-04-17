#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "hammer.h"
#include "hammer_config.h"
#include "hammer_memory.h"
#include "hammer_macros.h"

ALLOCSZ_ATTR(1)
void *hammer_mem_malloc(const size_t size)
{
	void *aux = malloc(size);

	if (hammer_unlikely(!aux && size)) {
		perror("malloc");
		return NULL;
	}

	return aux;
}

ALLOCSZ_ATTR(1)
void *hammer_mem_calloc(const size_t size)
{
	void *buf = calloc(1, size);
	if (hammer_unlikely(!buf)) {
		return NULL;
	}

	return buf;
}

ALLOCSZ_ATTR(2)
void *hammer_mem_realloc(void *ptr, const size_t size)
{
	void *aux = realloc(ptr, size);

	if (hammer_unlikely(!aux && size)) {
		perror("realloc");
		return NULL;
	}

	return aux;
}

void hammer_mem_free(void *ptr)
{
	free(ptr);
}
