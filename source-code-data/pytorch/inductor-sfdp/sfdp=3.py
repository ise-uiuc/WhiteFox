torch.nn.functional.dropout(
    torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
    p=dropout_p,
).matmul(value)
