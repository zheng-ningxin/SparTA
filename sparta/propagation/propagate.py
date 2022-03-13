import torch
import torch.nn as nn

__all__ = ['propagate_sparsity', 'extract_sparsity']

def extract_sparsity(model: nn.Module):
    ...

def propagate_sparsity(model: nn.Module):
    pre_sparsity = extract_sparsity(model)
    ...