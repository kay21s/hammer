CC = gcc

CFLAGS = -Wall -g
LIBS = -lpthread

INCLUDE_DIR = ./include
OBJS_DIR = ./objs

vpath % objs

TARGET = hammer
SOURCES = $(wildcard *.c)
#OBJS = $(pathsubst %.c, $(OBJS_DIR)/%.o, $(SOURCES))
OBJS = hammer.o hammer_sched.o hammer_connection.o hammer_dispatcher.o hammer_cpu_worker.o \
	hammer_socket.o hammer_epoll.o hammer_memory.o hammer_handler.o

%.o: %.c
	$(CC) $(CFLAGS) -I $(INCLUDE_DIR) -c $< -o $@
#	$(CC) $(CFLAGS) -I $(INCLUDE_DIR) -c $< -o $@

#OBJS = $(OBJS_DIR)/$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))

$(TARGET) : $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LIBS)

clean:
	rm -rf *.o  hammer
