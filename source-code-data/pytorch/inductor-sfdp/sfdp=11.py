q = query.permute(0, 2, 1, 3)
k = key.permute(0, 2, 1, 3)
v = value.permute(0, 2, 1, 3)
torch.nn.functional.dropout(
    torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
    p=dropout_p,
).matmul(v)