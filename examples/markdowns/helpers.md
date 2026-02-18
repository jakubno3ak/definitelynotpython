# Data Helpers Implementation

Utility functions for dataset preprocessing, normalization, and splitting. These tools bridge the gap between raw CSV files and tensor-ready matrices.

## Core Preprocessing Flow

Before feeding data into a Neural Layer, it must be loaded, separated into features and labels, and normalized.

```c
#include "include/matrix.h"
#include "include/ops.h"

// 1. Loading from CSV
Matrix RawData = create_matrix_from_csv("data.csv");

// 2. Slicing X and Y
// Assuming last column is the target (Y)
int input_cols = RawData.columns - 1;
Matrix X = create_matrix(RawData.rows, input_cols);
Matrix y = create_matrix(RawData.rows, 1);
slice_matrix(input_cols, 1, &RawData, &X, &y);

// 3. Normalization (Min-Max Scaling)
// Scales every feature to [0, 1] range to prevent gradient explosion
normalize_matrix_min_max(&X);

// 4. Train/Test Split
// 80% Training, 20% Testing, with Shuffling enabled
DataSplit split = train_test_split(&X, &y, 0.2, true);

```

## Key Functions Detailed

#### Matrix Slicing
The slice_matrix function allows you to partition a large matrix into sub-matrices. This is essential for isolating labels from features.

#### Min-Max Normalization
Neural Networks are sensitive to the scale of input data. Our implementation ensures that all features contribute equally to the loss calculation:

`x_normalized = (x - x_min) / (x_max - x_min)`

#### Train/Test Split and Shuffling
The train_test_split function performs two critical tasks:

**Shuffling**: Randomizes the order of rows to ensure the model doesn't learn patterns based on the data's entry order.

**Partitioning**: Allocates a percentage of data for validation to monitor the model's performance on unseen data.

#### Softmax

* [Explanation](./ops/softmax.md)
* [Definition](../../include/ops.h)
* [Implementation](../../src/ops.c)