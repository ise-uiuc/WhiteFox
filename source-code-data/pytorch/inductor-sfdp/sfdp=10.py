q = query.permute(0, 2, 1, 3)
k = key.permute(0, 2, 1, 3)
v = value.permute(0, 2, 1, 3)
torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)