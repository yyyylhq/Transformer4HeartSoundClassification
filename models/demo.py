import torch


N = 3
L = 10
D = 7
len_keep = int(L * (1 - 0.4))
print(len_keep)
x = torch.rand(N, L, D)

noise = torch.rand(N, L)

ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#print(ids_shuffle)
#print(ids_shuffle.shape)

ids_restore = torch.argsort(ids_shuffle, dim=1)
#print(ids_restore)
#print(ids_restore.shape)

ids_keep = ids_shuffle[:, :len_keep]
#print(ids_keep)
#print(ids_keep.shape)
print(ids_keep.unsqueeze(-1).repeat(1, 1, D))
print(ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)

x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
print(x_masked)
print(x)
print(x.shape)
print(x_masked.shape)

mask = torch.ones([N, L])
mask[:, :len_keep] = 0
mask = torch.gather(mask, dim=1, index=ids_restore)

print(mask)
print(mask.shape)
