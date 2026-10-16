# definitelynotpython - build system
#
#   make run EX=examples/test_mlp.c    build & run an example
#   make bench                         matrix multiplication benchmark
#   make MODE=debug run EX=...         -O0 -g + AddressSanitizer/UBSan
#   make BLAS=0 run EX=...             build without OpenBLAS
#   make data / make mnist             download datasets
#   make model                         download TinyStories 15M + tokenizer
#   make clean

CC    ?= gcc
MODE  ?= release
BLAS  ?= 1

CFLAGS   := -std=c11 -Wall -Wextra -fopenmp -MMD -MP
CPPFLAGS := -Iinclude
LDLIBS   := -lm -fopenmp

ifeq ($(MODE),debug)
  CFLAGS  += -O0 -g -fsanitize=address,undefined
  LDFLAGS += -fsanitize=address,undefined
else
  CFLAGS  += -O3 -march=native
endif

ifeq ($(BLAS),1)
  LDLIBS += -lopenblas
else
  CPPFLAGS += -DNO_BLAS
endif

BUILD := build/$(MODE)$(if $(filter 0,$(BLAS)),-noblas)
SRCS  := $(wildcard src/*.c src/backends/*.c)
OBJS  := $(SRCS:%.c=$(BUILD)/%.o)

BOSTON_CSV := boston.csv
PIMA_CSV   := pima-indians-diabetes.csv
DATA       := $(BOSTON_CSV) $(PIMA_CSV)

MODEL_DIR  := models
MODEL      := $(MODEL_DIR)/stories15M.bin
TOKENIZER  := $(MODEL_DIR)/tokenizer.bin

.PHONY: all run bench examples data mnist model clean help
.DEFAULT_GOAL := help

help:
	@sed -n '3,9p' Makefile | sed 's/^# \{0,1\}//'

# Library objects (compiled once, rebuilt only when changed)
$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Any example / benchmark -> build/<mode>/bin/<name>
$(BUILD)/bin/%: $(BUILD)/%.o $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

EXAMPLES := $(wildcard examples/*.c examples/ops/*.c)
examples: $(EXAMPLES:%.c=$(BUILD)/bin/%)
all: examples $(BUILD)/bin/benchmark

run: $(BUILD)/bin/$(basename $(EX)) $(DATA)
	@test -n "$(EX)" || { echo "Usage: make run EX=examples/<file>.c"; exit 1; }
	./$<

bench: $(BUILD)/bin/benchmark
	./$<

data: $(DATA)

$(BOSTON_CSV):
	wget -q --show-progress https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv -O $@

$(PIMA_CSV):
	wget -q --show-progress https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv -O $@

mnist: mnist_train.csv mnist_test.csv
mnist_%.csv:
	wget -q --show-progress https://pjreddie.com/media/files/$@ -O $@

model: $(MODEL) $(TOKENIZER)

$(MODEL):
	@mkdir -p $(MODEL_DIR)
	wget -q --show-progress https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O $@

$(TOKENIZER):
	@mkdir -p $(MODEL_DIR)
	wget -q --show-progress https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin -O $@

clean:
	rm -rf build

.SECONDARY:
-include $(shell find build -name '*.d' 2>/dev/null)
