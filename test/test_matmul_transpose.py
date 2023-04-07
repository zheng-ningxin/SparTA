import torch
import time

A = torch.rand(1024,1024).cuda()
B = torch.rand(1024,1024).cuda()
RUNTIME = 10000
torch.cuda.synchronize()
t_start = time.time()
for i in range(RUNTIME):
    torch.matmul(A, B)
torch.cuda.synchronize()
t_end = time.time()
print('No trans Time: ', t_end-t_start)


torch.cuda.synchronize()
t_start = time.time()
for i in range(RUNTIME):
    torch.matmul(A.t(), B)
torch.cuda.synchronize()
t_end = time.time()
print('Trans Time: ', t_end-t_start)
