#ifndef HAMMER_MEM_H
#define HAMMER_MEM_H

typedef struct
{
    char *data;
    unsigned long len;
} hammer_pointer;

#if ((__GNUC__ * 100 + __GNUC__MINOR__) > 430)  /* gcc version > 4.3 */
# define ALLOCSZ_ATTR(x,...) __attribute__ ((alloc_size(x, ##__VA_ARGS__)))
#else
# define ALLOCSZ_ATTR(x,...)
#endif

inline void *hammer_mem_malloc(const size_t size);
inline void *hammer_mem_calloc(const size_t size);
inline void *hammer_mem_realloc(void *ptr, const size_t size);
void hammer_mem_free(void *ptr);
void hammer_mem_pointers_init(void);

/* hammer_pointer_* */
hammer_pointer hammer_pointer_create(char *buf, long init, long end);
void hammer_pointer_free(hammer_pointer * p);
void hammer_pointer_print(hammer_pointer p);
char *hammer_pointer_to_buf(hammer_pointer p);
void hammer_pointer_set(hammer_pointer * p, char *data);
void hammer_pointer_reset(hammer_pointer * p);

#define hammer_pointer_init(a) {a, sizeof(a) - 1}

#endif
