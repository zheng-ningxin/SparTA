__all__ = ['specialize_matmul']

def specialize_matmul(in_tesa: tuple, weight_tesa: tuple, out_t_tesa: tuple):
    """
    Generate the kernels and profile the combined latency
    """
    # mocked for now
    latency = 0.2
    kernels = ['']
    aggr_type = ''
    return latency, kernels, aggr_type