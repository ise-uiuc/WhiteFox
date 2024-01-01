attn_weight = torch.softmax(
    (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
)
attn_weight = torch.dropout(attn_weight, dropout_p, True)
attn_weight @ value
