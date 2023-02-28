import torch
import time
a = torch.rand(10240)
c = a.cuda()
RUNTIME = 10240
t_start = time.time()
for i in range(RUNTIME):
    b = torch.cumsum(a, dim=0)
t_end = time.time()

print('CPU Time: {}'.format((t_end-t_start)*1000/RUNTIME))
torch.cuda.synchronize()
t_start = time.time()
for i in range(RUNTIME):
    d = torch.cumsum(c, dim=0)
torch.cuda.synchronize()
t_end = time.time()
print('GPU Time: {}'.format((t_end-t_start)*1000/RUNTIME))

print(b)
print(b.size())
print(d)
print(d.size())