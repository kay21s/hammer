GPUCRYPTO_DIR = ./

include Makefile.in

OBJS_DIR = objs
TARGET_DIR = lib
TARGET_FILE = libgpucrypto.a
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

.SUFFIXES : .cu .c .o

CU_SRC_FILES = $(wildcard *.cu)
CC_SRC_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)

SRC_FILES = $(CU_SRC_FILES) $(CC_SRC_FILES)
OBJS_FILE = $(CU_SRC_FILES:.cu=.o) $(CC_SRC_FILES:.c=.o)

OBJS = $(addprefix $(OBJS_DIR)/, $(OBJS_FILE))
DEPS = Makefile.dep

all: $(TARGET) test

$(TARGET): $(DEPS) $(OBJS) | $(TARGET_DIR) $(OBJS_DIR)
	ar rcs $@ $(OBJS)

$(TARGET_DIR):
	mkdir $(TARGET_DIR)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)

$(DEPS): $(SRC_FILES) $(HEADER_FILES)
	$(CC) -MM -MP -x c++ $(CU_SRC_FILES) $(CC_SRC_FILES) | sed 's![^:]*.o:!objs/&!g' > Makefile.dep

$(OBJS_DIR)/%.o : %.c
	$(CC) $(CFLAGS) $(INC) $(NVINC) -c $< -o $@

$(OBJS_DIR)/%.o : %.cu
	$(NVCC) $(NVCFLAGS) $(NVINC) -c $< -o $@

.PHONY : test clean

test: $(TARGET)
	make -C ./test

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
	make clean -C ./test

ifneq ($(MAKECMDGOALS), clean)
-include $(DEPS)
endif
