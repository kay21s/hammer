CC = gcc

OPENSSL_DIR = ../openssl-1.0.1c
ifneq ($(OPENSSL_DIR), )
	OPENSSL_LIBDIR = -L$(OPENSSL_DIR)/lib64
	OPENSSL_INCDIR = -I$(OPENSSL_DIR)/include
endif
INC = $(OPENSSL_INCDIR)

HEADER_DIR = include
OBJECT_DIR = objs
TARGET_DIR = ..
TARGET_FILE = hammer
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

SOURCE_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)
OBJECT_FILE = $(SOURCE_FILES:.c=.o)

OBJECT = $(addprefix $(OBJECT_DIR)/, $(OBJECT_FILE))

$(OBJECT_DIR):
	mkdir $(OBJECT_DIR)

CFLAGS = -Wall -g -Wall
LIBS = -lpthread -lcrypto

$(OBJECT_DIR)/%.o : %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

all: $(TARGET)

$(TARGET): $(OBJS) | $(TARGET_DIR) $(OBJECT_DIR)
	$(CC) $(OBJECT) -o $@ $(LIBS)

.PHONY : clean
clean:
	rm -rf $(TARGET) $(OBJECT)
