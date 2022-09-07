


__global__ void scan_non_zeros( int * mask, int M, int N, int block_h, int block_w,
                                int *extra_buffer, int dim)
{
    // mask is 0,1 matrix with a shape of (M, N)
    int m_idx = blockIdx.y;
    int n_idx = blockIdx.x;
    int have_non_zero = 0;
    int h_start, w_start, h_stride, w_stride;
    calculate_start_stride(&h_start, &w_start, &h_stride, &w_stride, block_h, block_w, blockIdx.x);    
    int global_offset = calculate_global_offset(m_idx, n_idx, block_h, block_w, M, N, dim);
    // main loop: scan corresponding data tile
    for(int i=h_start;i<block_h;i+=h_stride){
        for(int j=w_start; j<block_w; j+=w_stride){
            int block_offset = calculate_block_offset(i, j, block_h, block_w);
            have_non_zero += mask[global_offset+block_offset];
        }
    }
    reduce_across_thread_block(have_non_zero);
    if(have_non_zero){
        if(dim==0){
            // the skipping index is along N
            int pos = atomicAdd(extra_buffer[m_idx], 1);
            write_pos_into_buffer(pos, m_idx, n_idx);
        }else if(dim==0){
            int pos = atomicAdd(extra_buffer[m_idx], 1);
            write_pos_into_buffer(pos, m_idx, n_idx);
        }
    }
}
__global__ void copy_non_zeros(float *values, int block_h, int block_w, int * extra_buffer,
                               int* csr_row, int* csr_col, float* csr_val)
{
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int blk_idx = read_block_pos(extra_buffer, by, bx);
    int write_offset = blk_idx * block_h * block_w;
    int h_start, w_start, h_stride, w_stride;
    calculate_start_stride(&h_start, &w_start, &h_stride, &w_stride, block_h, block_w, blockDim.x);
    int global_offset = calculate_global_offset(by, bx, block_h, block_w, M, N, dim);
    for(int i=h_start; i<block_h; i+=h_stride){
        for(int j=w_start; j<block_w; j+=w_stride){
            int block_offset = calculate_block_offset(i, j, block_h, block_w);
            csr_val[write_offset+block_offset] = values[global_offset+block_offset];           
        }
    }
}