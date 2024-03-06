q = query.permute(0, 2, 1, 3)
k = key.permute(0, 2, 1, 3)
v = value.permute(0, 2, 1, 3)
div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
div = div.to(torch.float32)
attn_weight = torch.softmax(div, dim=-1)
attn_weight = attn_weight.to(torch.float16)
attn_weight @ v