CC = gcc

CFLAGS = -Wall -g -Wall
LIBS = -lpthread -lcrypto

OPENSSL_DIR = ../openssl-1.0.1c
ifneq ($(OPENSSL_DIR), )
	OPENSSL_LIBDIR = -L$(OPENSSL_DIR)/lib64
	OPENSSL_INCDIR = -I$(OPENSSL_DIR)/include
endif

################################################################
CUDA_PATH       ?= /usr/local/cuda-5.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
################################################################

HEADER_DIR = ./include
LIBCRYPTO_DIR = ../libgpucrypto
OBJECT_DIR = objs
TARGET_DIR = ..
INC = $(HEADER_DIR)

TARGET_FILE = hammer
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

SOURCE_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)
OBJECT_FILE = $(SOURCE_FILES:.c=.o)

OBJECT = $(addprefix $(OBJECT_DIR)/, $(OBJECT_FILE))

all: $(TARGET)

$(TARGET): $(OBJECT) | $(TARGET_DIR) $(OBJECT_DIR)
	$(CC) $(OBJECT) -o $@ $(LIBS)

$(OBJECT_DIR):
	mkdir $(OBJECT_DIR)


$(OBJECT_DIR)/%.o : %.c
	$(CC) $(CFLAGS) -I $(INC) -I $(CUDA_INC_PATH) -I $(LIBCRYPTO_DIR) -c $< -o $@

.PHONY : clean
clean:
	rm -rf $(TARGET) $(OBJECT)
