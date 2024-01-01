(
    torch.matmul(query, key.transpose(-2, -1))
    .div(inv_scale)
    .softmax(dim=-1)
    .matmul(value)
)
