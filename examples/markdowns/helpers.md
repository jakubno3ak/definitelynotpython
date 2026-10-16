# Data Helpers Implementation

Utility functions for dataset preprocessing, normalization, and splitting. These tools bridge the gap between raw CSV files and tensor-ready matrices.

## Core Preprocessing Flow

Before feeding data into a Neural Layer, it must be loaded, separated into features and labels, and normalized.

```c
#include "data.h"
#include "matrix.h"
#include "ops.h"

// 1. Loading from CSV
Matrix RawData = data_load_csv("data.csv");

// 2. Slicing X and Y
// Assuming last column is the target (Y)
int input_cols = RawData.columns - 1;
Matrix X = mat_create(RawData.rows, input_cols);
Matrix y = mat_create(RawData.rows, 1);
data_split_xy(input_cols, 1, &RawData, &X, &y);

// 3. Train/Test Split
// 80% Training, 20% Testing, with Shuffling enabled
DataSplit split = data_train_test_split(&X, &y, 0.2, true);

// 4. Normalization (Min-Max Scaling)
// Scale fitted on the training set only, then applied to both sets
MinMax scaler = minmax_fit(split.X_train);
minmax_apply(scaler, &split.X_train);
minmax_apply(scaler, &split.X_test);
minmax_free(scaler);

```

## Key Functions Detailed

#### Matrix Slicing
The data_split_xy function allows you to partition a large matrix into sub-matrices. This is essential for isolating labels from features.

#### Min-Max Normalization
Neural Networks are sensitive to the scale of input data. Our implementation ensures that all features contribute equally to the loss calculation:

`x_normalized = (x - x_min) / (x_max - x_min)`

`x_min` and `x_max` come from the training set only. Computing them on all data would leak
information about the test set into training (data leakage). Test values can therefore fall
slightly outside [0, 1].

#### Train/Test Split and Shuffling
The data_train_test_split function performs two critical tasks:

**Shuffling**: Randomizes the order of rows to ensure the model doesn't learn patterns based on the data's entry order.

**Partitioning**: Allocates a percentage of data for validation to monitor the model's performance on unseen data.

#### Softmax

* [Explanation](./ops/softmax.md)
* [Definition](../../include/ops.h)
* [Implementation](../../src/ops.c)