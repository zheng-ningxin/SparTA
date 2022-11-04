// Dense Tile 1: (BLOCK_M x K) * (K * BLOCK_N)
for k_loop = 0 -> (K / BLOCK_K)
    k_start = k_loop * BLOCK_K
    k_end = k_start + BLOCK_K
    for m = 0 -> BLOCK_M
        for n = 0 -> BLOCK_N
            C_shared[m][n] = 0    
            for k = k_start -> k_end 0 3 4
                C_shared[m][n] += A_shared[m][k] * B_shared[k][n]




for k_loop = 0 -> (K / BLOCK_K)
    k_start = k_loop * BLOCK_K
    k_end = k_start + BLOCK_K
    for m_loop = 0 -> 2
        m_start = m_loop * (BLOCK_M/2)
        m_end = m_start + (BLOCK_M/2)
        for m in m_start - > m_end
            for n = 0 -> BLOCK_N
                C_shared[m][n] = 0    
                for k = k_start -> k_end
                    C_shared[m][n] += A_shared[m][k] * B_shared[k][n]




for k_loop = 0 -> (K / BLOCK_K)
    k_start = k_loop * BLOCK_K
    k_end = k_start + BLOCK_K
    m_loop = 0
    m_start = m_loop * (BLOCK_M/2)
    m_end = m_start + (BLOCK_M/2)
    for m in m_start - > m_end
        for n = 0 -> BLOCK_N
            C_shared[m][n] = 0    
            for k = k_start -> k_end
                C_shared[m][n] += A_shared[m][k] * B_shared[k][n]

    m_loop = 1
    m_start = m_loop * (BLOCK_M/2)
    m_end = m_start + (BLOCK_M/2)
    for m in m_start - > m_end
        for n = 0 -> BLOCK_N
            C_shared[m][n] = 0    
            for k = k_start -> k_end
                C_shared[m][n] += A_shared[m][k] * B_shared[k][n]


// RELU
Sload(n)
for m in 0 -> BLOCK_M // Sparse
    for n in 0 -> BLOCK_N // Sparse
        Out[m][n] = relu(in[m][n])
Swrite(n)

for m in 0 -> BLOCK_M/2 // Sparse
    for n in 0 -> BLOCK_N // Sparse
        Out[m][n] = relu(in[m][n])
for m in BLOCK_M/2 -> BLOCK_M // Sparse
    for n in 0 -> BLOCK_N // Sparse
        Out[m][n] = relu(in[m][n])



for ic_loop = 0 -> IC/BLOCK_IC
    for batch = 0 -> BLOCK_BATCH
        for oc = 0 -> BLOCK_OC
            O_shared
            ic_start = ic_loop * BLOCK_IC
            ic_end = ic_start + BLOCK_IC
            for ic = ic_start -> ic_end
                for h = 0 - > H 
                    for w = 0 -> W 
                        for kh = 0 -> 3
                            for kw = 0 -> 3
                                    O_shared[batch, oc, h, w] += I[batch, ic,h+kh, w+kw] * W[oc, ic, kh, kw]