/***********************************************************
 *          mem_pool.h
 *  A lock-free memory management library for multi-core
 *  architecture. Designed for frequent memory allocate and 
 *  free operations of small-size control blocks.
 *  Linux uses the First-touch policy for allocating memory on
 *  a NUMA architecture: It allocates memory on the node the 
 *  thread is currently running on. This library also supports 
 *  libnuma.
 *  Note: Dynamicly allocate memory if current pool is empty.
 *  But do not free the allocated memory. Fit for real-time
 *  applications.
 *       Kai Zhang  Email:kay21s AT gmail DOT com
 *       2010   All rights Reserved
 * ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mem.h"

#define PERF_CODING 0
#if defined(PERF_CODING)
#  define PERF_PRINT(fmt, args...) printf("%s:%d: " fmt, __FUNCTION__ , __LINE__ , ## args)
#elif defined(PERF_DEBUG)
#  define PERF_PRINT(fmt, args...) printf("%s: " fmt, __FUNCTION__ , ## args)
#elif (PERF_LOG)
#  define PERF_PRINT(fmt, args...) printf(fmt, ## args)
#else
#  define PERF_PRINT(fmt, args...)
#endif


static char*mem_table[8][8] __attribute__((aligned(64)));
static unsigned short size_table[8];
static unsigned short num_table[8];

/********************************************************
* function name: init_mem_pool
* input:         void 
* output:        void
*
* init the two-dimension mem_table indexed by cpu_id and
* size_type. init size_table, it lists the size by given a
* size type. init num_table, it lists the number a size
* should be allocated one time.
*********************************************************/
void libpool_init()
{
	int i, j;
	int size_num = 8; // Note size_num should be a multiple of 8 -- for cache alignment

	for (i = 0; i < CPU_NUM; i ++) {
		for(j = 0; j < size_num; j ++) {
			mem_table[i][j] = NULL;
		}
	}

	for(i = 0; i < size_num; i ++)
		num_table[i] = 0;
}

/********************************************************
* function name: node_map
* return the node ID the core located on
* only available on the 8 core architecture E5530
* return 1 when cpu_id = 1,3,5,7 
* return 0 when cpu_id = 0,2,4,6.
*********************************************************/
int node_map(int cpu_id)
{
	return (cpu_id & 0x1);
}

/********************************************************
* function name: libpool_init_size
* input:         size_type, block_num, cpu_id
* output:        void
* 
* find the mem_table[cpu_id][size_type], it is the entry
* of the block link list. malloc block_num * size bytes
* and init the link list.
*********************************************************/
void libpool_init_size(int size_type, int block_num, int size, int cpu_id)
{
	int i;

	size_table[size_type] = size;
	if (mem_table[cpu_id][size_type] != NULL)
		return;

	mem_table[cpu_id][size_type] = (char *)malloc(block_num * size);
	for (i = 0; i < block_num - 1; i ++) {
		*((char **)(mem_table[cpu_id][size_type] + i * size)) = mem_table[cpu_id][size_type] + (i + 1) * size;
	}
	*((char **)(mem_table[cpu_id][size_type] + i * size)) = NULL;
	num_table[size_type] = block_num;
}

/********************************************************
* function name: mem_realloc
* input:         size_type, cpu_id
* output:        void 
*  
* called by mem_alloc, when there is no space in the entry
*********************************************************/

static void mem_realloc(int size_type, int cpu_id)
{
	int i, size = size_table[size_type], block_num = num_table[size_type];
//	PERF_PRINT("SIZE_REALLOC: size is %d, block_num is \n", size);
	if (mem_table[cpu_id][size_type] != NULL)
		return;
	
	mem_table[cpu_id][size_type] = (char *)malloc(block_num * size);
    
	for (i = 0; i < block_num - 1; i ++) {
		*((char **)(mem_table[cpu_id][size_type] + i * size)) = mem_table[cpu_id][size_type] + (i + 1) * size;
	}
	*((char **)(mem_table[cpu_id][size_type] + i * size)) = NULL;
}

/********************************************************
* function name: libpool_alloc
* input:         size_type, cpu_id
* output:        void * 
* 
* alloc a block of memory in the corresponding entry
*********************************************************/
void *libpool_alloc(int size_type, int cpu_id)
{
	void *p_block;
	if (mem_table[cpu_id][size_type] == NULL)
		mem_realloc(size_type, cpu_id);
	p_block = mem_table[cpu_id][size_type];
	mem_table[cpu_id][size_type] = *((char **)(mem_table[cpu_id][size_type]));
	return p_block;
}

/********************************************************
* function name: libpool_free
* input:         p_block, size_type, cpu_id
* output:        void 
*  
* return a block to its link list.
*********************************************************/
void libpool_free(void *p_block, int size_type, int cpu_id)
{
	*((char **)p_block) = mem_table[cpu_id][size_type];
	mem_table[cpu_id][size_type] = p_block;
}

