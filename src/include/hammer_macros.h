#ifndef HAMMER_MACROS_H
#define HAMMER_MACROS_H

#include <stdlib.h>

/* Boolean */
#define HAMMER_FALSE 0
#define HAMMER_TRUE  !HAMMER_FALSE
#define HAMMER_ERROR -1

/* Architecture */
#define INTSIZE sizeof(int)

/* Print macros */
#define HAMMER_INFO     0x1000
#define HAMMER_ERR      0X1001
#define HAMMER_WARN     0x1002
#define HAMMER_BUG      0x1003


//#define hammer_info(...)  hammer_print(HAMMER_INFO, __VA_ARGS__)
//#define hammer_err(...)   hammer_print(HAMMER_ERR, __VA_ARGS__)
//#define hammer_warn(...)  hammer_print(HAMMER_WARN, __VA_ARGS__)
//#define hammer_trace(...)  hammer_print(HAMMER_WARN, __VA_ARGS__)
#define hammer_info  printf
#define hammer_err   printf
#define hammer_warn  printf
#define hammer_trace  printf

/* Transport type */
#ifndef ARRAY_SIZE
# define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifdef __GNUC__ /* GCC supports this since 2.3. */
 #define PRINTF_WARNINGS(a,b) __attribute__ ((format (printf, a, b)))
#else
 #define PRINTF_WARNINGS(a,b)
#endif

#ifdef __GNUC__ /* GCC supports this since 2.7. */
 #define UNUSED_PARAM __attribute__ ((unused))
#else
 #define UNUSED_PARAM
#endif

/*
 * Validation macros
 * -----------------
 * Based on article http://lwn.net/Articles/13183/
 *
 * ---
 * ChangeSet 1.803, 2002/10/18 16:28:57-07:00, torvalds@home.transmeta.com
 *
 *	Make a polite version of BUG_ON() - WARN_ON() which doesn't
 *	kill the machine.
 *
 *	Damn I hate people who kill the machine for no good reason.
 * ---
 *
 */

#define hammer_unlikely(x) __builtin_expect((x),0)
#define hammer_likely(x) __builtin_expect((x),1)
#define hammer_prefetch(x, ...) __builtin_prefetch(x, __VA_ARGS__)

#define hammer_is_bool(x) ((x == HAMMER_TRUE || x == HAMMER_FALSE) ? 1 : 0)

#define hammer_bug(condition) do {                                          \
        if (hammer_unlikely((condition)!=0)) {                              \
            hammer_print(HAMMER_BUG, "Bug found in %s() at %s:%d",              \
                     __FUNCTION__, __FILE__, __LINE__);                 \
            abort();                                                    \
        }                                                               \
    } while(0)

/*
 * Macros to calculate sub-net data using ip address and sub-net prefix
 */

#define HAMMER_NET_IP_OCTECT(addr,pos) (addr >> (8 * pos) & 255)
#define HAMMER_NET_NETMASK(addr,net) htonl((0xffffffff << (32 - net)))
#define HAMMER_NET_BROADCAST(addr,net) (addr | ~HAMMER_NET_NETMASK(addr,net))
#define HAMMER_NET_NETWORK(addr,net) (addr & HAMMER_NET_NETMASK(addr,net))
#define HAMMER_NET_WILDCARD(addr,net) (HAMMER_NET_BROADCAST(addr,net) ^ HAMMER_NET_NETWORK(addr,net))
#define HAMMER_NET_HOSTMIN(addr,net) net == 31 ? HAMMER_NET_NETWORK(addr,net) : (HAMMER_NET_NETWORK(addr,net) + 0x01000000)
#define HAMMER_NET_HOSTMAX(addr,net) net == 31 ? HAMMER_NET_BROADCAST(addr,net) : (HAMMER_NET_BROADCAST(addr,net) - 0x01000000);

#if __GNUC__ >= 4
 #define HAMMER_EXPORT __attribute__ ((visibility ("default")))
#else
 #define HAMMER_EXPORT
#endif

// TRACE
#define HAMMER_TRACE(...) do {} while (0)

#endif
