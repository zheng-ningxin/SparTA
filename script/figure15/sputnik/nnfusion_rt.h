#pragma once

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" int kernel_entry(int m, int k, int n, int fine_nnz, int*d_row_swizzle, float*d_values, int*d_row_idx, int*d_col_idx, float* dB, float * dC, int beta);
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.