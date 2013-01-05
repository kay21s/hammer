/***********************************************************
 *                      libpool
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

#ifndef __MEM_LL_H
#define __MEM_LL_H

/***********************************************************
 *          Add more types
 * 1. add the symbolic name of size type in enum size_type
 * 2. add real size map in the size_table in mem_init()
 * ********************************************************/


enum size_type{
#if 0
    // Take these for example
    SIZE_HTTP_CONNECTION=0,
    SIZE_HTTP_REQUEST,
    SIZE_TCP_STREAM,
    SIZE_FIX_NODE,
    SIZE_FIX_SLIDE,
    SIZE_YY_BUFFER_STATE,
#endif
    SIZE_JOB,
    SIZE_CONN,
    SIZE_LIST_ELEM,
    SIZE_COUNT
};


// Modify These to Suit Your Application
#define CPU_NUM 8
#define CACHE_LINE_SIZE 64

void libpool_init();
void libpool_init_size(int size_type, int block_num, int size, int cpu_id);
void *libpool_alloc(int size_type, int cpu_id);
void libpool_free(void *p_block, int size_type,  int cpu_id);

#endif

