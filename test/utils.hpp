#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "bcsr.hpp"
#include <stdint.h>

void cal_block(bcsr*, float* );
void generate_bcsr(bcsr*, float* );

void cal_block(bcsr* mat, float* data) {
    // m_block : number of K axis block
    // n_block : number of N axis block
    for ( int i = 0 ; i < mat->m_block * mat->n_block ; i ++ ) {
        mat->is_block_present[i] = 0;
    }
    for ( int i = 0 ; i < mat->m * mat->n ; i ++ ) {
        if (data[i] != 0) {
            // 计算属于哪一个block
            int m_block_idx = i % mat->m / mat->m_block_sz;     // block index of K axis
            int n_block_idx = i / mat->m / mat->n_block_sz;     // block index of N axis
            /*
            if (mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] == 0) {
                mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] = 1;
                mat->nnz_block_num += 1;
            }
            */
            // NOTE: is_block_present is column major!
            if (mat->is_block_present[n_block_idx * mat->m_block + m_block_idx] == 0) {
                mat->is_block_present[n_block_idx * mat->m_block + m_block_idx] = 1;
                mat->nnz_block_num += 1;
            }
        }
    }
}

void cal_block_row(bcsr* mat, float* data) {
    // m_block : number of K axis block
    // n_block : number of N axis block
    for ( int i = 0 ; i < mat->m_block * mat->n_block ; i ++ ) {
        mat->is_block_present[i] = 0;
    }
    for ( int i = 0 ; i < mat->m * mat->n ; i ++ ) {
        if (data[i] != 0) {
            // 计算属于哪一个block
            int n_block_idx = i % mat->n / mat->n_block_sz;     // block index of K axis
            int m_block_idx = i / mat->n / mat->m_block_sz;     // block index of N axis

            if (mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] == 0) {
                mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] = 1;
                mat->nnz_block_num += 1;
            }

            // NOTE: is_block_present is column major!
            /*if (mat->is_block_present[n_block_idx * mat->m_block + m_block_idx] == 0) {
                mat->is_block_present[n_block_idx * mat->m_block + m_block_idx] = 1;
                mat->nnz_block_num += 1;
            }*/
        }
    }
}

void generate_bcsr(bcsr* mat, float* data) {
    int ptr = 0;
    int block_ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr ++ ] = block_ptr;
    for( int i = 0; i < mat->n_block; i += 1){
        for( int j = 0; j < mat->m_block; j += 1){
            if(mat->is_block_present[i * mat->m_block + j] == 1){
                mat->col_idx[block_ptr ++] = j;
                // B, row-major, N consecutive
                //for (int i_block = 0; i_block < mat->m_block_sz; i_block += 1){
                //    for(int j_block = 0; j_block < mat->n_block_sz; j_block += 1){
                //        mat->val[ptr++] = data[(j * mat->m_block_sz+i_block) * mat->n + (i * mat->n_block_sz + j_block)];
                //    }
                //}
                // B, column-major, K consecutive
                for (int i_block = 0; i_block < mat->n_block_sz; i_block += 1){
                    for(int j_block = 0; j_block < mat->m_block_sz; j_block += 1){
                        mat->val[ptr++] = data[(i * mat->n_block_sz+i_block) * mat->m + (j * mat->m_block_sz + j_block)];
                    }
                }
            }
        }
        mat->row_ptr[row_ptr ++] = block_ptr;
    }
    /*
    for ( int i = 0 ; i < mat->m_block ; i += 1) {
        for ( int j = 0 ; j < mat->n_block ; j += 1) {
            if ( mat->is_block_present[i * mat->n_block + j] == 1) {
                mat->col_idx[block_ptr ++ ] = j;
                // copy whole block into val
                for (int i_block = 0 ; i_block < mat->m_block_sz ; i_block ++ ) {
                    for ( int j_block = 0 ; j_block < mat->n_block_sz ; j_block ++) {
                        mat->val[ptr ++ ] = data[ (i * mat->m_block_sz + i_block) * mat->n + (j * mat->n_block_sz + j_block)];
                    }
                }
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr ++ ] = block_ptr;
    }
    */
}

void generate_bcsr_row(bcsr* mat, float* data) {
    int ptr = 0;
    int block_ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr ++ ] = block_ptr;
    for( int i = 0; i < mat->m_block; i += 1){
        for( int j = 0; j < mat->n_block; j += 1){
            if(mat->is_block_present[i * mat->n_block + j] == 1){
                mat->col_idx[block_ptr ++] = j;
                // B, row-major, N consecutive
                for (int i_block = 0; i_block < mat->m_block_sz; i_block += 1){
                    for(int j_block = 0; j_block < mat->n_block_sz; j_block += 1){
                        mat->val[ptr++] = data[(i * mat->m_block_sz+i_block) * mat->n + (j * mat->n_block_sz + j_block)];
                    }
                }
                // B, column-major, K consecutive
                /*for (int i_block = 0; i_block < mat->n_block_sz; i_block += 1){
                    for(int j_block = 0; j_block < mat->m_block_sz; j_block += 1){
                        mat->val[ptr++] = data[(i * mat->n_block_sz+i_block) * mat->m + (j * mat->m_block_sz + j_block)];
                    }
                }*/
            }
        }
        mat->row_ptr[row_ptr ++] = block_ptr;
    }
    /*
    for ( int i = 0 ; i < mat->m_block ; i += 1) {
        for ( int j = 0 ; j < mat->n_block ; j += 1) {
            if ( mat->is_block_present[i * mat->n_block + j] == 1) {
                mat->col_idx[block_ptr ++ ] = j;
                // copy whole block into val
                for (int i_block = 0 ; i_block < mat->m_block_sz ; i_block ++ ) {
                    for ( int j_block = 0 ; j_block < mat->n_block_sz ; j_block ++) {
                        mat->val[ptr ++ ] = data[ (i * mat->m_block_sz + i_block) * mat->n + (j * mat->n_block_sz + j_block)];
                    }
                }
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr ++ ] = block_ptr;
    }
    */
}

void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void init_mask(int * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0;
        }
        else
        {
            ptr[i] = 1;
        }
    }
}

#endif
