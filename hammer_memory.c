#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "hammer.h"
#include "hammer_config.h"
#include "hammer_memory.h"
#include "hammer_http.h"
#include "hammer_macros.h"

inline ALLOCSZ_ATTR(1)
void *hammer_mem_malloc(const size_t size)
{
	void *aux = malloc(size);

	if (hammer_unlikely(!aux && size)) {
		perror("malloc");
		return NULL;
	}

	return aux;
}

inline ALLOCSZ_ATTR(1)
void *hammer_mem_calloc(const size_t size)
{
	void *buf = calloc(1, size);
	if (hammer_unlikely(!buf)) {
		return NULL;
	}

	return buf;
}

inline ALLOCSZ_ATTR(2)
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

hammer_pointer hammer_pointer_create(char *buf, long init, long end)
{
	hammer_pointer p;

	hammer_pointer_reset(&p);
	p.data = buf + init;

	if (init != end) {
		p.len = (end - init);
	}
	else {
		p.len = 1;
	}

	return p;
}

void hammer_pointer_free(hammer_pointer * p)
{
	hammer_mem_free(p->data);
	p->len = 0;
}

char *hammer_pointer_to_buf(hammer_pointer p)
{
	char *buf;

	buf = hammer_mem_malloc(p.len + 1);
	if (!buf) return NULL;

	memcpy(buf, p.data, p.len);
	buf[p.len] = '\0';

	return (char *) buf;
}

void hammer_pointer_print(hammer_pointer p)
{
	unsigned int i;

	printf("\nDEBUG HAMMER_POINTER: '");
	for (i = 0; i < p.len && p.data != NULL; i++) {
		printf("%c", p.data[i]);
	}
	printf("'");
	fflush(stdout);
}

void hammer_pointer_set(hammer_pointer *p, char *data)
{
	p->data = data;
	p->len = strlen(data);
}

void hammer_pointer_reset(hammer_pointer * p)
{
    p->data = NULL;
    p->len = 0;
}
