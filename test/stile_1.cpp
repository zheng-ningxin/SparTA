// Dense Tile 1: (BLOCK_M x K) * (K * BLOCK_N)
for k_loop = 0 -> (K / BLOCK_K)
    k_start = k_loop * BLOCK_K
    k_end = k_start + BLOCK_K
    // Load to Shared memory
    for m = 0 -> BLOCK_M
        for k = k_start -> k_end
            load(A_shared[m][k], A_global[m][k])
    for n = 0 -> BLOCK_N
        for k = k_start -> k_end
            load(B_shared[n][k], B_global[n][l])
                
    // Dense tile calculation of BLOCK_M x BLOCK_K x BLOCK_N
    for m = 0 -> BLOCK_M
        for n = 0 -> BLOCK_N
            C_shared[m][n] = 0    
            for k = k_start -> k_end
                C_local[m][n] += A_shared[m][k] * B_shared[k][n]
            C_shared[m][n] = C_local[m][n]
    // Write back to global memory
    for m = 0 -> BLOCK_M
        for n = 0-> BLOCK_N
            write(C_gloab[m][n], C_shared[m][n])