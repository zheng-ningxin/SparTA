#ifndef BCSR_H
#define BCSR_H

#include <stdlib.h>
#include <fstream>
#include <iostream>

using namespace std;

class bcsr {
public:
    float* val;
    int* is_block_present;
    int* col_idx;
    int* row_ptr;
    int m, n, m_block_sz, n_block_sz, m_block, n_block, nnz_block_num;

    bcsr(int m, int n, int m_block_sz, int n_block_sz): m(m), n(n), m_block_sz(m_block_sz), n_block_sz(n_block_sz) {
        m_block = m / m_block_sz;       // number of K axis block
        n_block = n / n_block_sz;       // number of N axis block
        nnz_block_num = 0;
        is_block_present = (int*) malloc(sizeof(int) * m_block * n_block);
        val = NULL;
        col_idx = NULL;
        row_ptr = NULL;
    }
    
    ~bcsr() {
        free(is_block_present);
        if (val != NULL) free(val);
        if (col_idx != NULL) free(col_idx);
        if (row_ptr != NULL) free(row_ptr);
    }

    void print() {
        printf("is block present: \n");
        for ( int i = 0 ; i < m_block ; i ++ ) {
            for ( int j = 0 ; j < n_block ; j ++ ) {
                printf("%d ", is_block_present[i * n_block + j]);
            }
            printf("\n");
        }

        printf("row_ptr: \n");
        for ( int i = 0 ; i < m_block + 1 ; i ++ ) {
            printf("%d ", row_ptr[i]);
        }
        printf("\n");
        printf("col_idx: \n");
        for ( int i = 0 ; i < nnz_block_num ; i ++ ) {
            printf("%d ", col_idx[i]);
        }
    }

    void load_val(){
        size_t size_val = nnz_block_num * m_block_sz * n_block_sz;
        size_t size_row = m_block + 1;
        size_t size_col = nnz_block_num;
        ifstream f_val("val.bin", ios::out | ios::binary);
        ifstream f_row("row.bin", ios::out | ios::binary);
        ifstream f_col("col.bin", ios::out | ios::binary);
        f_val.read((char *)val, sizeof(float) * size_val);
        f_row.read((char *)row_ptr, sizeof(int) * size_row);
        f_col.read((char *)col_idx, sizeof(int) * size_col);
        f_val.close();
        f_row.close();
        f_col.close();
    }

    void export_val(){
        size_t size_val = nnz_block_num * m_block_sz * n_block_sz;
        size_t size_row = m_block + 1;
        size_t size_col = nnz_block_num;
        ofstream f_val("val.bin", ios::out | ios::binary);
        ofstream f_row("row.bin", ios::out | ios::binary);
        ofstream f_col("col.bin", ios::out | ios::binary);
        f_val.write((char *)val, sizeof(float) * size_val);
        f_row.write((char *)row_ptr, sizeof(int) * size_row);
        f_col.write((char *)col_idx, sizeof(int) * size_col);
        f_val.close();
        f_row.close();
        f_col.close();
    }
};
#endif