#!/bin/bash

# Check if source file is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh examples/<filename>.c"
    exit 1
fi

SOURCE_FILE=$1
OUTPUT_NAME=$(basename "$SOURCE_FILE" .c)

# Automatic Data Downloader
# Boston Housing for linear models
if [[ "$SOURCE_FILE" == *"linear"* && ! -f "boston.csv" ]]; then
    echo "Dataset missing: boston.csv. Downloading..."
    wget https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv -O boston.csv
fi

# Pima Diabetes for logistic/layer models
if [[ ("$SOURCE_FILE" == *"logistic"* || "$SOURCE_FILE" == *"layer"*) && ! -f "pima-indians-diabetes.csv" ]]; then
    echo "Dataset missing: pima-indians-diabetes.csv. Downloading..."
    wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
fi

# MNIST for future digit classification
if [[ "$SOURCE_FILE" == *"mnist"* && ! -f "mnist_train.csv" ]]; then
    echo "Dataset missing: MNIST. Downloading (large files)..."
    wget https://pjreddie.com/media/files/mnist_train.csv
    wget https://pjreddie.com/media/files/mnist_test.csv
fi

echo "--- Compiling $SOURCE_FILE ---"

# Compile with all sources in src/ and backends/
# Includes OpenMP and math library
gcc -O3 "$SOURCE_FILE" \
    src/*.c \
    src/backends/*.c \
    -I. -I./include \
    -lm -fopenmp -lopenblas \
    -o "$OUTPUT_NAME"

# Execute if compilation succeeded
if [ $? -eq 0 ]; then
    echo "--- Running $OUTPUT_NAME ---"
    ./"$OUTPUT_NAME"
    rm "$OUTPUT_NAME"
    echo "--- Done ---"
else
    echo "--- Compilation Failed ---"
    exit 1
fi