CC = gcc

CFLAGS = -Wall -O -lpthread

INCLUDE = include

TARGET = project

%.o: %.c
	$(CC) $(CFLAGS) -I $(INCLUDE) -c $< -o $@

SOURCES = $(wildcard *.c *.cpp)
OBJS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))

$(TARGET) : $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

clean:
	rm -rf *.o hammer
