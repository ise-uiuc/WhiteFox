q = query.permute([0, 2, 1, 3])
k = key.permute([0, 2, 1, 3])
v = value.permute([0, 2, 1, 3])
bs = q.size(0)
k_len = k.size(-2)
scores = q @ k.transpose(-2, -1)
scores = scores.div(inv_scale)
fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
(
    torch.nn.functional.dropout(
        torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p
    )
    @ v
)