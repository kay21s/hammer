#ifndef   	HAMMER_LIST_H_
#define   	HAMMER_LIST_H_

#include <stddef.h>

#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
#endif

#define container_of(ptr, type, member) ({                      \
      const typeof( ((type *)0)->member ) *__mptr = (ptr);      \
      (type *)( (char *)__mptr - offsetof(type,member) );})


struct hammer_list
{
	struct hammer_list *prev, *next;
};

static inline void hammer_list_init(struct hammer_list *list)
{
	list->next = list;
	list->prev = list;
}

static inline void __hammer_list_add(struct hammer_list *new, struct hammer_list *prev,
                                 struct hammer_list *next)
{
	next->prev = new;
	new->next = next;
	new->prev = prev;
	prev->next = new;
}

static inline void hammer_list_add(struct hammer_list *new, struct hammer_list *head)
{
	__hammer_list_add(new, head->prev, head);
}

static inline void __hammer_list_del(struct hammer_list *prev, struct hammer_list *next)
{
	prev->next = next;
	next->prev = prev;
}

static inline void hammer_list_del(struct hammer_list *entry)
{
	__hammer_list_del(entry->prev, entry->next);
	entry->prev = NULL;
	entry->next = NULL;
}

static inline int hammer_list_is_empty(struct hammer_list *head)
{
	if (head->next == head) return 0;
	else return -1;
}

#define hammer_list_foreach(curr, head) for( curr = (head)->next; curr != (head); curr = curr->next )
#define hammer_list_foreach_safe(curr, n, head) \
    for (curr = (head)->next, n = curr->next; curr != (head); curr = n, n = curr->next)

#define hammer_list_entry( ptr, type, member ) container_of( ptr, type, member )

/* First node of the list */
#define hammer_list_entry_first(ptr, type, member) container_of((ptr)->next, type, member)

/* Last node of the list */
#define hammer_list_entry_last(ptr, type, member) container_of((ptr)->prev, type, member)

/* Next node */
#define hammer_list_entry_next(ptr, type, member, head)                     \
    (ptr)->next == (head) ? container_of((head)->next, type, member) :  \
        container_of((ptr)->next, type, member);

#endif /* !HAMMER_LIST_H_ */
