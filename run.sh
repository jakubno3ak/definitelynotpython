#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <source_file.c>"
    echo "Example: ./run.sh examples/test_layer.c"
    exit 1
fi

SOURCE_FILE=$1
OUTPUT_NAME=$(basename "$SOURCE_FILE" .c)

echo "Building and running $SOURCE_FILE..."

gcc -O3 "$SOURCE_FILE" \
    src/matrix.c \
    src/ops.c \
    src/layer.c \
    src/backends/ops_naive.c \
    src/backends/ops_omp.c \
    -I. -I./include \
    -lm -fopenmp \
    -o "$OUTPUT_NAME"

if [ $? -eq 0 ]; then
    echo "------------------------------------"
    ./"$OUTPUT_NAME"
    rm "$OUTPUT_NAME"
else
    echo "Compilation failed."
    exit 1
fi