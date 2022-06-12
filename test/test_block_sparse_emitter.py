from sparta.codegen.emitter.block_sparse_emmiter import GPUBlockSparseEmitter


if __name__ == '__main__':
    emitter = GPUBlockSparseEmitter(1024, 1024, 1024, './tmp')
    import ipdb; ipdb.set_trace()
    best_cfg = emitter.tunning_kernel_cfg(emitter.space)
    