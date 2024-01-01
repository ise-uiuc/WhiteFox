attn_weight = torch.softmax(
    (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
)
attn_weight @ value
