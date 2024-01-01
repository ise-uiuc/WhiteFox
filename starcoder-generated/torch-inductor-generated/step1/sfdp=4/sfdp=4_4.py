
c = m.conv.weight
attn_mask = torch.zeros_like(x)[:,:1]
query = c[:,:1]
key = c
def scaled_dot_product(query, key, value, attn_mask):
    