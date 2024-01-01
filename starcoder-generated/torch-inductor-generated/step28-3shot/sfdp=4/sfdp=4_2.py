
def attention_head_with_mask(q, k, v, mask=None):
    if mask is not None:
        # Compute a mask such that mask[i, j] = 1.0 iff i!= j and mask is broadcastable to q @ k
        n_spatial = mask.ndim - q.ndim
        for _ in range(n_spatial):
            mask = mask.unsqueeze(-2)
        assert mask.shape == q.shape

    _dot_product = q @ k.transpose(-2, -1)
    attn = _dot_product / math.sqrt(k.size(-1))
    if mask is not None:
        assert mask.shape == attn.shape
        attn += mask
    attn_weights = torch.softmax(attn, dim=-1)
    out = attn_weights @ v
    return out

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(4, 30, 10)
key = torch.randn(4, 20, 10)
value = torch.randn(4, 20, 10)
mask = torch.abs(torch.randn(4, 30, 20))

